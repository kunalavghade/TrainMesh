# Contributing

TrainMesh is CPU-first. Keep tensor and gradient communication inside PyTorch distributed backends; the Go control plane must never relay tensor payloads.

Before opening a change:

```bash
gofmt -w cmd internal
go test ./...
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q
```

Use the CPU/Gloo example for runtime changes. Do not describe local multi-process execution as multi-machine validation. Update `docs/IMPLEMENTATION_STATUS.md` when implementation or test coverage changes, and report hardware-specific features as unsupported until executed on that hardware.
