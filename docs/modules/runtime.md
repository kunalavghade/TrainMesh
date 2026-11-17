# Training runtime (Python)

**Path:** `python/trainmesh_runtime/`. Started once per rank, by torchrun or directly with the
`RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` environment.

| File | Responsibility |
|------|----------------|
| `__main__.py` | CLI entry (`--config`, `--output`, `--resume`); maps `TrainMeshError` to exit 2 |
| `config.py` | Load and validate the `JobSpec`; `TrainingSettings` with defaults ([config.md](config.md)) |
| `distributed.py` | `DistContext` from env, process-group setup (Gloo or NCCL), tensor-only collectives |
| `devices.py` | Device, precision and collective backend for this rank ([devices.md](devices.md)) |
| `sync.py` | Gradient sync: DDP, or one flattened all-reduce (staged through host memory for MPS) |
| `balance.py` | Throughput balancing: largest-remainder split, per-epoch re-split from median step time |
| `diloco.py` | DiLoCo outer loop: anchor weights, pseudo-gradients through `tmcomm`, Nesterov outer step ([diloco.md](diloco.md)) |
| `data.py` | Deterministic `synthetic` / `teacher` datasets; `GlobalBatchSampler` (same global batches for any split), batched loaders |
| `runtime.py` | The trainer: build, train epochs, validate, checkpoint, report |
| `evaluation.py` | Validation loss and accuracy, reduced across ranks |
| `checkpoint.py` | Format v2 checkpoints (safetensors + JSON), integrity and compatibility checks |
| `errors.py` | `TrainMeshError` hierarchy with stable codes |
| `probe.py` | Capability report used by `trainmesh doctor`; `--benchmark` measures each device for `tm-agent` |

## One run, step by step (`train()` → `_run_job()`)

1. `load_config` → `training_settings` (all defaults applied).
2. `context_from_env` → `DistContext(rank, world_size, local_rank, local_world_size)`;
   `devices.choose` resolves device, precision and collective backend; `init_process_group`
   (Gloo or NCCL, `env://`) when `world_size > 1`.
3. Seed Python and torch with `run.seed`; set intra-op threads from `OMP_NUM_THREADS`.
4. **Data** (`data.py`): every rank builds the same tensors from the seed, so no data moves
   between machines.
   - `synthetic`: random features, random labels. Nothing to learn; used for throughput.
   - `teacher`: random features; label = `argmax(x · W)` for a fixed random teacher matrix `W`.
     Learnable, so validation accuracy shows whether training works.
   - Training loader: `GlobalBatchSampler` + `BatchLoader`: shuffled per epoch, step *k* is the
     same global batch for any world size and any per-rank split; each batch is fetched with
     one tensor index (no per-sample `DataLoader` path).
   - Validation loader: rank *r* takes samples `r, r+W, r+2W, …` — disjoint, no padding, so
     global counts are exact.
5. **Model:** `MLP(input → hidden → ReLU → output)` on the chosen device. `choose_sync` (after
   all-gathering every rank's device kind) wraps it in `DistributedDataParallel`, or uses one
   flattened all-reduce per step for MPS, mixed devices or unequal batches. Optimizer: Adam or
   SGD.
6. **Resume** (optional): `load_checkpoint` verifies and restores model, optimizer and this
   rank's RNG; training continues from the saved epoch.
7. **Each epoch:**
   - `train_epoch`: micro-batches inside `model.no_sync()` except the one that closes an
     accumulation window, so gradients are reduced **once per optimizer step** (either
     strategy). Loss is summed on the device and read once per epoch. Compute and sync time are
     measured per step.
   - With mixed device kinds: measure `max_parameter_drift`, then re-broadcast rank 0's
     parameters. With `balance: throughput`: re-split the next epoch's batch from the median
     step times ([devices.md](devices.md)).
   - Epoch loss and sample count are all-reduced (sum) to global values.
   - `evaluate` (if `training.validation.samples > 0`): each rank scores its shard; loss sum,
     correct count and sample count are all-reduced → global `val_loss`, `val_accuracy`.
   - Rank 0 prints one JSON event line, e.g.
     `{"event": "epoch", "epoch": 3, "loss": 0.10, "samples": 65536, "world_size": 4,
     "val_loss": 0.20, "val_accuracy": 0.917}`.
   - `save_checkpoint` (if enabled) — a collective: every rank sends its RNG state, rank 0 writes.
8. **Consistency check:** each rank hashes its parameters (SHA-256); the 32-byte digests are
   all-gathered as tensors (no pickle). Any difference raises `DivergenceError`.
9. Rank 0 writes the result JSON (to `--output` or `run.results_path`) and prints it.

## Result JSON (rank 0)

| Field | Meaning |
|-------|---------|
| `world_size`, `rank`, `local_rank`, `threads_per_rank` | Topology of the run |
| `epochs`, `global_step` | Epochs run in this invocation; optimizer steps in total |
| `epoch_losses`, `final_loss` | Global, sample-weighted training loss |
| `samples_processed`, `duration_seconds`, `samples_per_second` | Throughput (duration = slowest rank) |
| `model_sha256`, `parameters_consistent` | Replica consistency proof |
| `dataset_type` | `synthetic` or `teacher` |
| `device`, `precision`, `collective_backend`, `gradient_sync` | What rank 0 resolved |
| `balance`, `rank_batch_sizes`, `balance_history`, `ranks`, `compute_imbalance`, `mixed_devices`, `max_parameter_drift` | Per-rank devices, splits and timings ([devices.md](devices.md)) |
| `strategy`, `compute_utilization`, `diloco` | Strategy; compute ÷ (compute + communication), `null` under DDP; outer-round statistics ([diloco.md](diloco.md)) |
| `validation` | `null`, or `{samples, final_loss, final_accuracy, accuracy_history, loss_history}` |

## Checkpoint format v2 (`checkpoint.py`)

A directory (default `checkpoints/latest/`), written to `latest.staging/` and swapped in, keeping
`latest.previous/` until the new one is in place:

| File | Content |
|------|---------|
| `model.safetensors` | Model state dict |
| `optimizer.safetensors` | Optimizer tensors, keyed `state.<param_id>.<name>` |
| `rng.safetensors` | Torch CPU RNG per rank, keyed `torch_cpu.rank<r>` |
| `state.json` | Job config dict, optimizer param groups and scalar state |
| `metadata.json` | `format_version: 2`, epoch, global step, world size, seed, a SHA-256 per file |

Loading checks, in order: metadata present → format version 2 (v1 pickle files are refused) →
every file's SHA-256 (hashed in 8 MiB chunks) → world size matches → `training.model`, `dataset`,
`batch`, `optimizer` match the current config. Nothing is unpickled.

## Distributed helpers (`distributed.py`)

All collectives move fixed-size tensors: `all_reduce_sum_`, `max_across_ranks`,
`all_gather_fixed`, `assert_replicas_identical`. Single-process runs skip them, so the same code
runs with or without a process group.

## Tests

`python/tests/test_accumulation.py` (one all-reduce per optimizer step; accumulation equals one
large batch), `test_checkpoint.py` (round trip, tampering, legacy format, compatibility),
`test_learning.py` (schema defaults and errors, deterministic teacher data, disjoint validation
shards, perfect accuracy for the teacher itself, **training reaches ≥ 85% validation accuracy**),
`test_runtime.py` (two real ranks over Gloo, checkpoint and resume, uneven and re-split batches
give the same model, cpu+mps ranks), `test_devices.py`, `test_balance.py`.
