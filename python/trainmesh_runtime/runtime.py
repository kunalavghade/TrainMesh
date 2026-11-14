from __future__ import annotations

import json
import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("configuration must be a YAML mapping with version: 1")
    execution = config.get("execution", {})
    if execution.get("strategy") != "data_parallel":
        raise ValueError("only execution.strategy=data_parallel is implemented")
    backend = execution.get("backend", {})
    if backend.get("device", "cpu") != "cpu" or backend.get("communication", "gloo") != "gloo":
        raise ValueError("this release supports CPU with Gloo only; CUDA/NCCL is not available")
    training = config.get("training", {})
    if training.get("precision", "fp32") != "fp32":
        raise ValueError("only fp32 precision is currently supported")
    if training.get("dataset", {}).get("type") != "synthetic":
        raise ValueError("only deterministic synthetic datasets are supported")
    batch = training.get("batch", {})
    if int(batch.get("per_rank", 0)) <= 0 or int(batch.get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("training.batch.per_rank and gradient_accumulation_steps must be positive")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    return config


def _max_across_ranks(value: float, world_size: int, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    bare_model = model.module if isinstance(model, DistributedDataParallel) else model
    for name, tensor in sorted(bare_model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(
    destination: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, step: int, config: dict[str, Any], seed: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    bare_model = model.module if isinstance(model, DistributedDataParallel) else model
    temporary = destination / ".checkpoint.tmp"
    torch.save(
        {
            "model": bare_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "seed": seed,
            "config": config,
        },
        temporary,
    )
    temporary.replace(destination / "checkpoint.pt")
    summary = {
        "version": 1,
        "epoch": epoch,
        "global_step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "saved_at_unix": time.time(),
        "dataset": config["training"]["dataset"],
    }
    digest = hashlib.sha256((destination / "checkpoint.pt").read_bytes()).hexdigest()
    summary["checkpoint_sha256"] = digest
    (destination / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def train(config_path: str, output_path: str | None = None, resume_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if resume_path:
        config.setdefault("run", {})["resume"] = resume_path
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("installed PyTorch does not provide the Gloo process-group backend")
        dist.init_process_group(backend="gloo", init_method="env://")

    seed = int(config.get("run", {}).get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

    training = config["training"]
    model_config = training.get("model", {})
    dataset_config = training["dataset"]
    count = int(dataset_config.get("samples", 4096))
    input_size = int(model_config.get("input_size", dataset_config.get("input_size", 32)))
    hidden_size = int(model_config.get("hidden_size", 64))
    output_size = int(model_config.get("output_size", dataset_config.get("output_size", 4)))
    # Identical deterministic dataset on every rank; DistributedSampler assigns disjoint indices.
    generator = torch.Generator().manual_seed(seed + 1)
    features = torch.randn(count, input_size, generator=generator)
    target = torch.randint(output_size, (count,), generator=generator)
    dataset = TensorDataset(features, target)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if world_size > 1 else None
    batch_size = int(training["batch"]["per_rank"])
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)

    model = MLP(input_size, hidden_size, output_size)
    if world_size > 1:
        model = DistributedDataParallel(model)
    optimizer_config = training.get("optimizer", {})
    optimizer_name = str(optimizer_config.get("name", "adam")).lower()
    learning_rate = float(optimizer_config.get("learning_rate", 0.001))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    criterion = nn.CrossEntropyLoss(reduction="sum")

    resume = config.get("run", {}).get("resume")
    start_epoch, global_step = 0, 0
    if resume:
        checkpoint_path = Path(resume) / "checkpoint.pt"
        metadata_path = Path(resume) / "metadata.json"
        if not checkpoint_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"checkpoint is incomplete: expected {checkpoint_path} and metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if metadata.get("checkpoint_sha256") != actual_digest:
            raise ValueError("checkpoint checksum mismatch; refusing to resume corrupted state")
        if int(metadata.get("world_size", -1)) != world_size:
            raise ValueError("checkpoint world size does not match this run; change membership through a planned restart")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_training = checkpoint.get("config", {}).get("training", {})
        for key in ("model", "dataset", "batch", "optimizer"):
            if saved_training.get(key) != training.get(key):
                raise ValueError(f"checkpoint training.{key} differs from current configuration")
        base = model.module if isinstance(model, DistributedDataParallel) else model
        base.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

    started = time.perf_counter()
    epoch_losses: list[float] = []
    for epoch in range(start_epoch, int(training["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum, samples = 0.0, 0
        accumulation = int(training["batch"].get("gradient_accumulation_steps", 1))
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for batch_index, (x, y) in enumerate(loader):
            logits = model(x)
            loss = criterion(logits, y)
            (loss / (len(y) * accumulation)).backward()
            pending += 1
            if pending == accumulation or batch_index == len(loader) - 1:
                if pending < accumulation:
                    correction = accumulation / pending
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                optimizer.step()
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            loss_sum += float(loss.detach().item())
            samples += len(y)
        totals = torch.tensor([loss_sum, samples], dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        epoch_loss = float(totals[0] / max(1, totals[1]))
        epoch_losses.append(epoch_loss)
        if rank == 0:
            print(json.dumps({"event": "epoch", "epoch": epoch + 1, "loss": epoch_loss, "samples": int(totals[1]), "world_size": world_size}), flush=True)
        ckpt = training.get("checkpoint", {})
        if ckpt.get("enabled", True) and rank == 0:
            checkpoint_dir = Path(ckpt.get("directory", "./checkpoints"))
            _save_checkpoint(checkpoint_dir / "latest", model, optimizer, epoch + 1, global_step, config, seed)

    elapsed = time.perf_counter() - started
    elapsed = _max_across_ranks(elapsed, world_size, torch.device("cpu"))
    digest = _model_digest(model)
    if world_size > 1:
        digests: list[str | None] = [None] * world_size
        dist.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise RuntimeError(f"DDP model parameters diverged across ranks: {digests}")
    result = {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "epochs": int(training["epochs"]) - start_epoch,
        "global_step": global_step,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "duration_seconds": elapsed,
        "samples_per_second": (count * max(0, int(training["epochs"]) - start_epoch)) / max(elapsed, 1e-9),
        "model_sha256": digest,
        "parameters_consistent": True,
    }
    destination = output_path or config.get("optimizer", {}).get("results", "")
    if rank == 0 and destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return result
