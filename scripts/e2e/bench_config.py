"""Write a TrainMesh job config whose global batch stays fixed across topologies.

With a fixed global batch, DistributedSampler gives every optimizer step the same samples no
matter how many ranks share them, so single- and multi-machine runs must reach the same loss.
The default `teacher` dataset is learnable, so validation accuracy proves training worked.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_GLOBAL_BATCH = 256
DEFAULT_SAMPLES = 65_536
DEFAULT_INPUT_SIZE = 256
DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_OUTPUT_SIZE = 10
DEFAULT_EPOCHS = 3
DEFAULT_DATASET = "teacher"
DEFAULT_VALIDATION_SAMPLES = 4096

TEMPLATE = """version: 1
run:
  name: {name}
  seed: 42
  results_path: "{results_path}"
execution:
  strategy: data_parallel
  backend: {{device: cpu, communication: gloo}}
  workers: {machines}
  processes_per_worker: {ranks_per_machine}
training:
  model:
    type: mlp
    input_size: {input_size}
    hidden_size: {hidden_size}
    output_size: {output_size}
  dataset:
    type: {dataset}
    samples: {samples}
    input_size: {input_size}
    output_size: {output_size}
  optimizer: {{name: adam, learning_rate: 0.001}}
  batch: {{per_rank: {per_rank}, gradient_accumulation_steps: 1}}
  epochs: {epochs}
  precision: fp32
  checkpoint: {{enabled: {checkpoint_enabled}, directory: {checkpoint_dir}}}
  validation: {{samples: {validation_samples}}}
"""


def per_rank_batch(global_batch: int, world_size: int) -> int:
    if global_batch % world_size:
        raise ValueError(f"global batch {global_batch} is not divisible by world size {world_size}")
    return global_batch // world_size


def render(args: argparse.Namespace) -> str:
    world_size = args.machines * args.ranks_per_machine
    if args.samples % args.global_batch:
        raise ValueError(f"samples {args.samples} must be a multiple of {args.global_batch}")
    return TEMPLATE.format(
        name=args.name,
        machines=args.machines,
        ranks_per_machine=args.ranks_per_machine,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        samples=args.samples,
        dataset=args.dataset,
        validation_samples=args.validation_samples,
        per_rank=per_rank_batch(args.global_batch, world_size),
        epochs=args.epochs,
        checkpoint_enabled="true" if args.checkpoint_dir else "false",
        checkpoint_dir=args.checkpoint_dir or "./checkpoints",
        results_path=args.results_path,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="trainmesh-e2e")
    parser.add_argument("--machines", type=int, required=True)
    parser.add_argument("--ranks-per-machine", type=int, required=True)
    parser.add_argument("--global-batch", type=int, default=DEFAULT_GLOBAL_BATCH)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--dataset", choices=["teacher", "synthetic"], default=DEFAULT_DATASET)
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--checkpoint-dir", default="", help="empty disables checkpoints")
    parser.add_argument("--results-path", default="", help="rank 0 writes its result JSON here")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.stdout.write(render(parse_args(sys.argv[1:])))
    except ValueError as exc:
        sys.exit(f"bench_config: {exc}")
