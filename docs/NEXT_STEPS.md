# TrainMesh — Next Steps (handoff)

> Written 2026-11-14 for the next agent picking up this repo. It assumes **no memory of earlier
> sessions**. Follow the tasks **in order**, do **one task at a time**, and **stop and report to
> the user after each task**. When this file and the code disagree, trust the code and say so.

Read these first, in this order (each is short):

1. This file, completely.
2. [`docs/TRAINMESH_ARCHITECTURE.md`](TRAINMESH_ARCHITECTURE.md) §1, §2.3, §9 and §10. Skip the rest
   until a task points you there.
3. [`docs/IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

---

## 0. Ground rules (do not skip)

| Rule | Why |
|------|-----|
| **Never** add `Co-Authored-By`, "Generated with Claude Code" or any AI attribution to commits or PRs. | The user requires it, in every project. |
| Commit as the repo-local identity that is already configured (`git config --local user.email` → `67228832+kunalavghade@users.noreply.github.com`). Never change it, never use the work email. | The repo is public. |
| **Never** force-push, rewrite history, or delete a branch you did not create, without asking the user first. | `main` is public. |
| Commit or push **only** when the user asks, or when a task below explicitly says to. | User preference. |
| Before saying a task is done, run `make lint` and `make test` from the repo root. Both must pass. Paste the last lines of output in your report. | Evidence, not claims. |
| Change **one thing at a time** when debugging. Write down what you changed and what happened. | You will otherwise lose track of the cause. |
| If a step says "ask the user", stop and ask. Do not guess. | Some steps need installs or decisions. |
| Do not rewrite files you were not asked to touch. Do not "clean up" unrelated code. | Small, reviewable changes. |

Style: the user's coding rules live in `~/.claude/skills/kunal-coding-style/SKILL.md` and
`~/.claude/skills/kunal-best-practices/SKILL.md`. Read both before writing code. Short version:
guard clauses first, small functions, typed errors, config instead of magic numbers, tests next to
source, comments only for a non-obvious *why*.

---

## 1. Where things stand

### Repo layout (only the parts you will touch)

```text
cmd/trainmesh/main.go          CLI entry: wires signals, calls internal/cli
internal/cli/                  cobra commands: doctor, plan, run, benchmark, cluster
internal/config/               Go config loader + validator (v1 YAML)
internal/launcher/             builds torchrun args/env, runs ranks in a process group
python/trainmesh_runtime/      Python training runtime (config, distributed, checkpoint, runtime)
python/tests/                  pytest suites (accumulation, checkpoint, runtime)
.github/workflows/ci.yml       CI: Go + Python on ubuntu-latest and macos-latest
Makefile                       build, test, lint, fmt targets
```

### Commands you will use

```bash
make build        # go build -o bin/trainmesh ./cmd/trainmesh
make test         # go test -race ./... and pytest
make lint         # gofmt, go vet, golangci-lint, ruff, black --check
make fmt          # gofmt -w and black
make e2e-local    # real CLI end to end on this machine (~1 min)
make e2e-minikube # 2-node minikube cluster: 1 vs 2 machine comparison (slow, needs Docker)
./bin/trainmesh run --config examples/cpu-ddp.yaml   # real 2-rank CPU training
```

How to test end to end, and how the 1-vs-multi machine comparison works: [`docs/E2E_TESTING.md`](E2E_TESTING.md).
For any change to the CLI, launcher or Python runtime, also run `make e2e-local` before reporting.

The Python venv is `.venv/` at the repo root. Use `.venv/bin/python`, never the system `python3`,
for anything that imports torch.

### Git state at hand-off

- `main` = `4983fc4` "fix: complete Phase 0 runtime and launcher hardening" (pushed).
- Branch `ci/macos-hang` → PR [kunalavghade/TrainMesh#1](https://github.com/kunalavghade/TrainMesh/pull/1)
  (pushed, Auto-fix on). Fixes the macOS CI hang (Task 1). Its history contains debug commits; it
  should be **squash-merged** so they do not land on `main` individually.
- Branch `feat/e2e-multinode` (based on `ci/macos-hang`): e2e scripts, minikube multi-node test,
  1-vs-multi machine comparison, `docs/E2E_TESTING.md`, CI `e2e` job. Open its PR after PR #1 is
  merged (rebase it onto `main` first).

### Done (do not redo)

Phase 0 from the architecture doc: `no_sync` gradient accumulation, safetensors checkpoints
(format v2), per-rank RNG, process-group cleanup on Ctrl-C, bounded port retry, CLI split into
`internal/cli`, lint + CI config. See `docs/IMPLEMENTATION_STATUS.md`.

End-to-end tooling (branch `feat/e2e-multinode`): `scripts/e2e/local.sh`, `scripts/e2e/minikube.sh`,
`scripts/e2e/manual-node.sh`, `scripts/e2e/bench_config.py`, `scripts/e2e/compare.py` (+ tests),
`deploy/docker/runtime.Dockerfile`. See `docs/E2E_TESTING.md`.

### Traps already hit — avoid them

1. **Do not use `torchrun --standalone`.** On macOS its c10d store ignores `USE_LIBUV=0` and hangs.
   The launcher deliberately uses `--master_addr=127.0.0.1 --master_port=<port>` with a bounded
   retry (`internal/launcher/launcher.go`).
2. **Keep `USE_LIBUV=0` and, on macOS, `GLOO_SOCKET_IFNAME=lo0`** for local runs. Removing them
   breaks macOS.
3. **Threads per rank default to 1 on purpose.** Measured: more threads were slower for small
   models. Do not change the default without a benchmark (`./bin/trainmesh benchmark`).
4. In `python/tests/test_accumulation.py` there is **no** `from __future__ import annotations`.
   DDP checks the comm hook's real annotations at runtime; adding that import breaks the test.
5. `subprocess.TimeoutExpired.stdout/.stderr` are **bytes** even with `text=True`.
6. The shell is **zsh**: `$var` holding several words is *not* split. Use `${=var}` or arrays.
7. `gofmt` realigns struct fields, so an exact-text edit can fail after formatting. Re-read the
   file before editing it.
8. When benchmarking, use a config that actually contains the key you vary. A `sed` on a missing
   key silently does nothing.
9. **torchrun's agent calls `socket.getfqdn()` at startup.** Where reverse DNS is broken (GitHub's
   macOS runners: 35 s per lookup) torchrun stalls at `Rendezvous'ing worker group`. pytest
   therefore starts ranks directly (`python/tests/test_runtime.py`), and the CI `e2e` job that goes
   through torchrun runs on Linux only.
10. For a fair 1-vs-multi machine comparison keep the **global batch** fixed
    (`scripts/e2e/bench_config.py` does this). Then every topology must reach the same final loss;
    a difference means a real bug, not noise.

---

## Task 1 — Land the macOS CI fix (PR #1) and the e2e branch

**Root cause (proven).** On GitHub's macOS runner the DNS server never answers *reverse* lookups:
`gethostbyaddr('127.0.0.1')` and `gethostbyaddr('::1')` take 35.0 s each. torchrun's agent calls
`socket.getfqdn()` at startup, so the two-rank pytest timed out. An `/etc/hosts` workaround only
passed by accident (the diagnostic lookups had warmed the resolver cache).

**Fix (commit `aa8166c` on PR #1).** `test_two_rank_gloo_training_and_checkpoint` now starts both
ranks directly with the env torchrun would set (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, …), logging to
files. The CI hostname workaround was removed. The torchrun path is covered by
`scripts/e2e/local.sh` in the Linux-only CI `e2e` job (on `feat/e2e-multinode`).

Steps:

1. `gh pr checks 1 --repo kunalavghade/TrainMesh`. All four jobs must pass on the latest commit.
   If the macOS Python job fails, read its log (`gh run view <RUN_ID> --log-failed`) and report it
   with the excerpt. Do not re-add `/etc/hosts` hacks.
2. Ask the user: "PR #1 is green. Squash-merge it and delete the branch?" On yes:
   `gh pr merge 1 --repo kunalavghade/TrainMesh --squash --delete-branch`, then
   `git switch main && git pull --ff-only && git branch -D ci/macos-hang`.
3. Rebase the e2e branch onto the new `main`: `git switch feat/e2e-multinode && git rebase main`.
   If git reports the PR #1 commits as already applied, that is expected. Run `make lint`,
   `make test`, `make e2e-local`.
4. Ask the user before pushing it and opening its PR. PR text: what was added (see
   `docs/E2E_TESTING.md`) and the measured minikube table from that doc. No AI attribution.
5. Optional, ask first: make `trainmesh doctor` time `socket.getfqdn()` and warn above 2 s, with the
   fix from the troubleshooting table in `docs/E2E_TESTING.md` (`python/trainmesh_runtime/probe.py`
   + `internal/cli/doctor.go` + tests).

**Done when:** both branches are on `main` and the `ci` workflow on `main` is green for all jobs
(`go` ×2, `python` ×2, `e2e`).

---

## Task 2 — Remove the Node 20 deprecation warnings in CI

GitHub warns that these actions run on deprecated Node 20: `actions/checkout@v4`,
`actions/setup-go@v5`, `astral-sh/setup-uv@v6`, `golangci/golangci-lint-action@v8`.

1. For each action, find the latest major release. **Do not guess version numbers**:
   ```bash
   gh api repos/actions/checkout/releases/latest --jq .tag_name
   gh api repos/actions/setup-go/releases/latest --jq .tag_name
   gh api repos/astral-sh/setup-uv/releases/latest --jq .tag_name
   gh api repos/golangci/golangci-lint-action/releases/latest --jq .tag_name
   ```
2. Open each release's notes (`gh release view <tag> --repo <owner/repo>`) and confirm it runs on
   Node 24 and has no breaking input changes we use (`go-version-file`, `version`).
3. Edit `.github/workflows/ci.yml`: change only the `@vN` suffixes to the new majors.
4. Ask the user before pushing. After the push, confirm the run has no "Node.js 20 is deprecated"
   annotation: `gh run view <RUN_ID> --repo kunalavghade/TrainMesh | grep -i node`.

**Done when:** CI is green and the Node 20 annotation is gone.

---

## Task 3 — One config schema for Go and Python (protobuf)

**Why.** `internal/config/config.go` (Go) and `python/trainmesh_runtime/config.py` (Python)
validate the same YAML separately and will drift. Architecture doc §5.13 and ADR-008.

This task needs a new tool. **Ask the user first:** "Task 3 needs `buf` (protobuf toolchain).
OK to install it with `brew install bufbuild/buf/buf`?" Do nothing until they answer.

Break it into these sub-steps. Commit after each one only if the user wants per-step commits.

1. **Schema only.** Create `proto/trainmesh/v1/jobspec.proto` describing exactly the fields
   that `examples/cpu-ddp.yaml` uses today (version, run, execution, training, planner). Use the
   sketch in architecture doc §5.13 as a guide, but only add fields that exist in the YAML now.
   Add `buf.yaml` and run `buf lint`. **Done when** `buf lint` passes.
2. **Go generation.** Add `buf.gen.yaml` producing Go code into `internal/gen/trainmesh/v1/`.
   Run `buf generate`. **Done when** `go build ./...` passes. Do not use the generated types yet.
3. **Python generation.** Generate Python into `python/trainmesh_runtime/gen/`. **Done when**
   `.venv/bin/python -c "import trainmesh_runtime.gen"` works.
4. **Switch Go to the generated types** in `internal/config`, keeping every existing test passing
   unchanged. Keep the `optimizer:` → `planner:` legacy migration and its warning.
5. **Switch Python** `config.py` to parse into the generated type, keeping the `ConfigError`
   messages that the tests match ("CPU with Gloo", etc.).
6. Add a test that loads `examples/cpu-ddp.yaml` in both languages and checks they agree on
   world size and global batch.

**Done when:** `make lint && make test` pass, both languages use the generated types, and
`docs/IMPLEMENTATION_STATUS.md` says the schema is implemented.

---

## Task 4 — Phase 1, the LAN mesh (large; do in small pieces)

This is the first step that actually connects PCs. Read architecture doc §4, §5.1, §5.3 and §5.14
first. **Before starting, show the user this sub-step list and ask which one to start with.** Do
each sub-step as its own small change with tests, and report after each.

| # | Sub-step | Where | Done when |
|---|----------|-------|-----------|
| 4.1 | Node lifecycle state machine (JOINING → PROFILING → READY → BUSY / DEGRADED → LOST, DRAINING) as **pure** Go with a transition table and typed errors for illegal transitions | `internal/membership/state.go` + `state_test.go` | Table-driven tests cover every legal and illegal transition |
| 4.2 | Join tokens: generate a 256-bit token, store only its SHA-256, single use, TTL from config, revoke | `internal/security/token.go` + tests (use an injected clock, no `time.Sleep`) | Tests for expired, reused, revoked and valid tokens |
| 4.3 | Coordinator state store on bbolt behind a small `Store` interface | `internal/store/` | Tests use a temp-dir DB; interface allows an in-memory fake |
| 4.4 | `coordinator.proto` with `Join`, `Heartbeat`, `ReportCapabilities` only | `proto/trainmesh/v1/` (needs Task 3's `buf`) | `buf lint` passes, Go code generates |
| 4.5 | `meshd` binary: gRPC server for 4.4 using 4.1–4.3, **without TLS yet**, bound to localhost | `cmd/meshd/`, `internal/coordinator/` | An in-process test joins a fake node and sees it reach READY |
| 4.6 | `tm-agent` binary: join with a token, heartbeat on an interval from config | `cmd/tm-agent/`, `internal/agent/` | Integration test: meshd + agent on one machine, node becomes READY, stop agent → node becomes LOST after the configured timeout |
| 4.7 | mTLS (coordinator CA issues node certs at join) | `internal/security/` | Test: a node without a valid cert is rejected |
| 4.8 | Agent-launched multi-node training (one rank per machine) | `internal/agent/`, `internal/launcher/` | Manual two-machine test from `docs/MULTI_MACHINE_MANUAL_TEST.md` passes; ask the user to run it |

Rules for Task 4: no new dependency without telling the user which and why; every network call
has a timeout from config; every retry has a max attempt count; every goroutine stops on context
cancel; and no placeholder or mock implementations in non-test code.

---

## Reporting template (use after every task)

```text
Task: <number and name>
Changed: <files>
Verified: <commands run + last lines of output>
Not done / not verified: <honest list>
Question for you: <only if needed>
```
