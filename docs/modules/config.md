# Job schema and config loading

**Paths:** `proto/trainmesh/v1/jobspec.proto` (source of truth), `internal/config` (Go loader),
`python/trainmesh_runtime/config.py` (Python loader), generated code in
`internal/gen/trainmesh/v1/` and `python/trainmesh/v1/`, `python/buf/validate/`.

## Why one schema

Before Phase 0 the Go CLI and the Python runtime validated the YAML separately and disagreed
(Go ignored `model` and `optimizer`; Python ignored `workers`). Now there is one protobuf schema
that defines both the **shape** of a job and its **rules**, using
[protovalidate](https://protovalidate.com) annotations (`buf.validate`). Go and Python generate
types from it and evaluate the same rules, so a config is accepted or rejected identically
wherever it is loaded, and error messages are the same.

## The schema at a glance

```text
JobSpec
├── version            must be 1
├── cluster.name
├── run                name, seed (optional, default 42), resume, results_path
├── execution          strategy=data_parallel, workers ≥ 1, processes_per_worker ≥ 1,
│                      cpu_threads_per_rank ≥ 0, backend{device=cpu, communication=gloo},
│                      rendezvous{backend, endpoint} (informational)
├── training
│   ├── model          type ∈ {"", mlp}, input/hidden/output sizes ≥ 0 (0 = default)
│   ├── dataset        type ∈ {synthetic, teacher}, samples > 0, input/output sizes
│   ├── optimizer      name ∈ {"", adam, sgd}, learning_rate ≥ 0 (0 = 0.001)
│   ├── batch          per_rank > 0, gradient_accumulation_steps ≥ 0 (0 = 1)
│   ├── epochs > 0, precision ∈ {"", fp32}
│   ├── checkpoint     enabled (optional, default true), directory (default ./checkpoints)
│   └── validation     samples ≥ 0 (0 = no validation)
└── planner            free-form (google.protobuf.Struct), reserved for the Phase 6 planner
```

Rules with a custom message use CEL, for example:

```protobuf
string device = 1 [(buf.validate.field).cel = {
  id: "execution.backend.device"
  message: "this release supports device=cpu only; CUDA/NCCL is not available"
  expression: "this == 'cpu'"
}];
```

Zero values mean "use the default" unless a field is `optional`; defaults are applied in one
place per language (`training_settings()` in Python, accessor methods in Go).

## Loading, step by step (both languages)

1. Read YAML into a generic map.
2. **Migrate legacy keys:** a top-level `optimizer:` block becomes `planner:` and its `results`
   entry becomes `run.results_path`. The Go CLI prints a deprecation warning.
3. **Decode strictly** into the generated `JobSpec` (Go: YAML → JSON → `protojson.Unmarshal`;
   Python: `json_format.ParseDict`). Unknown keys and wrong types are errors.
4. **Validate** with protovalidate. *All* violations are reported together, each as
   `field.path: message`.

Go API (`internal/config`):

| Function / method | Purpose |
|-------------------|---------|
| `Load(path) (Config, error)` / `Parse(raw)` | Steps 1–4; errors wrap `config.ErrInvalid` |
| `Validate(spec) error` | Step 4 only |
| `Config.Spec` | The generated `*trainmeshv1.JobSpec`; read fields with generated getters |
| `WorldSize()`, `AccumulationSteps()`, `GlobalBatchSize()` | Derived values with defaults |
| `UsesLegacyOptimizerKey()` | Whether step 2 migrated anything |

Python API (`trainmesh_runtime.config`):

| Function | Purpose |
|----------|---------|
| `load_config(path) -> JobSpec` / `parse_config(document)` | Steps 1–4; raises `ConfigError` |
| `training_settings(spec) -> TrainingSettings` | Frozen dataclass with every default applied; the trainer reads only this |
| `to_dict(spec)` | Plain dict stored in checkpoints and compared on resume |

## Changing the schema

1. Edit `proto/trainmesh/v1/jobspec.proto` (add fields; never renumber or reuse a field number).
2. `make generate` (runs `buf lint` and `buf generate`; needs `buf`, `protoc`, `protoc-gen-go`).
3. Apply the new default in `training_settings()` and, if the CLI needs it, a Go accessor.
4. Add tests on both sides (`internal/config/config_test.go`, `python/tests/test_learning.py`).
5. `make lint && make test`.

Generated files are committed so that building does not require the protobuf toolchain. They are
excluded from ruff/black and must never be edited by hand.

## Example

[`examples/cpu-ddp.yaml`](../../examples/cpu-ddp.yaml) is loaded by a Go test
(`TestExampleConfigIsValid`), so the example can never drift from the schema.
