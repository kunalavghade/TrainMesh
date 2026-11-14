"""Print the local PyTorch capability report as JSON; used by `trainmesh doctor`."""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any


def capability_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch": "missing",
        "distributed": False,
        "gloo": False,
        "torchrun": False,
        "safetensors": importlib.util.find_spec("safetensors") is not None,
        "cpu_count": os.cpu_count(),
    }
    if importlib.util.find_spec("torch") is None:
        return report

    import torch
    import torch.distributed as dist

    report["torch"] = torch.__version__
    report["distributed"] = dist.is_available()
    report["gloo"] = dist.is_available() and dist.is_gloo_available()
    report["torchrun"] = importlib.util.find_spec("torch.distributed.run") is not None
    return report


if __name__ == "__main__":
    print(json.dumps(capability_report()))
