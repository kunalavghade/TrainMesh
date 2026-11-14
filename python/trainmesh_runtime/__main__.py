import argparse
import json
import logging
import sys

from .errors import TrainMeshError
from .runtime import train


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="TrainMesh CPU/Gloo training runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", help="verified checkpoint directory")
    args = parser.parse_args()
    try:
        result = train(args.config, args.output, args.resume)
    except TrainMeshError as exc:
        print(f"trainmesh-runtime: [{exc.code}] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
    if result["rank"] == 0:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
