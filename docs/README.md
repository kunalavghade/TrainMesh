# TrainMesh documentation

Start with the architecture document, then the module map. Each document has one job; when the
code changes, update the document that owns that topic.

| Document | Read it to learn | Kept up to date when |
|----------|------------------|----------------------|
| [TRAINMESH_ARCHITECTURE.md](TRAINMESH_ARCHITECTURE.md) | Vision, review of the original code, target architecture (control plane, agents, islands, `tmcomm`, `tmdata`), device matrix, performance design, roadmap P0–P6, ADRs | Design decisions change or a phase finishes |
| [modules/README.md](modules/README.md) | What exists today, module by module, and how a run flows through them | Any module is added or its behaviour changes |
| [modules/cli.md](modules/cli.md) | `trainmesh` commands and what each one does | CLI changes |
| [modules/config.md](modules/config.md) | The single job schema (`jobspec.proto`), its rules, and loading in Go and Python | The schema changes |
| [modules/launcher.md](modules/launcher.md) | How ranks are started, supervised and stopped | Launch behaviour changes |
| [modules/runtime.md](modules/runtime.md) | The Python training runtime: data, DDP training, validation, checkpoints, results | Runtime changes |
| [modules/mesh.md](modules/mesh.md) | The Phase 1 mesh: `meshd`, `tm-agent`, joining, liveness, job flow, failure handling, security, how to operate it | Coordinator, agent or security changes |
| [CPU_BACKEND.md](CPU_BACKEND.md) | Installing and running on CPU (macOS and Linux), tuning, config changes per release | Setup or defaults change |
| [E2E_TESTING.md](E2E_TESTING.md) | End-to-end tests and the 1-machine vs multi-machine proof (local, minikube, physical) | Tests or measured results change |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | What is implemented, with evidence, and what is not | Every merged change |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Step-by-step handoff for the next contributor or agent | A task finishes or is added |

Diagrams are Mermaid and render on GitHub. Generated protobuf code is documented in
[modules/config.md](modules/config.md), not here.
