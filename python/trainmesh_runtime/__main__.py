import argparse
import json

from .runtime import train


def main() -> None:
    parser = argparse.ArgumentParser(description="TrainMesh CPU/Gloo training runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", help="verified checkpoint directory")
    args = parser.parse_args()
    result = train(args.config, args.output, args.resume)
    if result["rank"] == 0:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
