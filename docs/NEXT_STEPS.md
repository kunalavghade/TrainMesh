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
make generate     # regenerate Go/Python code from proto/ after editing jobspec.proto
make e2e-local    # real CLI end to end on this machine, incl. learning check (~1 min)
make e2e-minikube # 2-node minikube cluster: 1 vs 2 machine comparison (slow, needs Docker)
./bin/trainmesh run --config examples/cpu-ddp.yaml   # real 2-rank CPU training
```

How to test end to end, and how the 1-vs-multi machine comparison works: [`docs/E2E_TESTING.md`](E2E_TESTING.md).
For any change to the CLI, launcher or Python runtime, also run `make e2e-local` before reporting.

The Python venv is `.venv/` at the repo root. Use `.venv/bin/python`, never the system `python3`,
for anything that imports torch.

### Git state at hand-off

History was rewritten on 2026-11-16 so every commit is authored by `kunalavghade` (no-reply
address); do not reintroduce the work email or any AI attribution.

- `main` = `8ff4ef5` (pushed).
- `feat/p0-jobspec` → PR [kunalavghade/TrainMesh#2](https://github.com/kunalavghade/TrainMesh/pull/2):
  Phase 0 (e2e tooling, single schema, proof of learning, module docs). Auto-fix is on.
- `feat/p1-lan-mesh` (based on `feat/p0-jobspec`) → its own PR: Phase 1 (coordinator, agents,
  mTLS, CLI mesh commands, mesh e2e). Merge it **after** PR #2; then rebase it onto `main`
  (`git rebase --onto main feat/p0-jobspec feat/p1-lan-mesh`).
- `feat/e2e-multinode` (local only) is contained in `feat/p0-jobspec`; delete it once PR #2 merges.
- `gh`'s active account is `kunalavghade`; if a `gh` call is refused, check `gh auth status`.

### Done (do not redo)

Phase 0 from the architecture doc: `no_sync` gradient accumulation, safetensors checkpoints
(format v2), per-rank RNG, process-group cleanup on Ctrl-C, bounded port retry, CLI split into
`internal/cli`, lint + CI config. See `docs/IMPLEMENTATION_STATUS.md`.

Phase 0 is complete (see `docs/TRAINMESH_ARCHITECTURE.md` §9): single protobuf schema with
shared validation rules, learnable `teacher` dataset with validation accuracy, per-module docs
in `docs/modules/`.

Phase 1 is implemented and verified on one Mac and on minikube (see §10 status and
`docs/modules/mesh.md`): `meshd`, `tm-agent`, tokens, CA + mTLS, liveness, scheduling,
`trainmesh run --coordinator`, failure propagation.

End-to-end tooling: `scripts/e2e/local.sh`, `scripts/e2e/minikube.sh`,
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
    (`scripts/e2e/bench_config.py` does this). Then every topology must reach the same final loss
    and validation accuracy; a difference means a real bug, not noise.
11. **Stay on `protovalidate` 1.x for Python.** `protovalidate` 2.x depends on Buf's new
    `protobuf-py` runtime, not Google's `protobuf`, and does not work with the protoc-generated
    code here. `python/pyproject.toml` pins `protovalidate>=1.2,<2`.
12. **Generated code** (`internal/gen/`, `python/buf/`, `python/trainmesh/`) is never edited or
    reformatted; ruff/black exclude it. Change the `.proto` and run `make generate`.
13. **minikube shares the Docker VM's memory with every other container** on this Mac (the user
    runs other stacks there). `scripts/e2e/minikube.sh` checks free VM memory before each scenario;
    if it refuses, report it to the user. Never stop or remove containers you did not create.

---

## Task 1 — Land the Phase 0 PR

1. Check the PR's CI: `GH_TOKEN=$(gh auth token --user kunalavghade) gh pr checks <N> --repo
   kunalavghade/TrainMesh`. All jobs (`go` ×2, `python` ×2, `e2e`) must pass. If one fails, read
   its log (`gh run view <RUN_ID> --log-failed`), fix the cause on `feat/p0-jobspec`, re-run
   `make lint && make test && make e2e-local`, push.
2. Ask the user: "The Phase 0 PR is green. Squash-merge it?" On yes:
   `gh pr merge <N> --repo kunalavghade/TrainMesh --squash --delete-branch`, then
   `git switch main && git pull --ff-only && git branch -D feat/e2e-multinode feat/p0-jobspec`.
3. Optional, ask first: make `trainmesh doctor` time `socket.getfqdn()` and warn above 2 s
   (slow reverse DNS stalls torchrun; see the troubleshooting table in `docs/E2E_TESTING.md`).

**Done when:** Phase 0 is on `main` and the `ci` workflow on `main` is green for every job.

Background on the macOS CI fix already on `main`: GitHub's macOS runner never answers reverse DNS
lookups for `127.0.0.1`/`::1` (35 s each), and torchrun's agent calls `socket.getfqdn()`. The
pytest two-rank test therefore starts ranks directly; torchrun is covered by the Linux `e2e` job.

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

## Task 3 — Done: one config schema for Go and Python

Implemented in Phase 0: `proto/trainmesh/v1/jobspec.proto` with protovalidate rules, generated
Go and Python code, both loaders on the generated types. Read `docs/modules/config.md` before
touching configs. To change the schema: edit the `.proto`, `make generate`, add tests, update
`docs/modules/config.md`.

---

## Task 4 — Finish Phase 1 (small items) and accept it on real machines

Phase 1 is built. Read `docs/modules/mesh.md` first. Remaining, each a small PR with tests:

| # | Item | Where | Done when |
|---|------|-------|-----------|
| 4.1 | **Physical acceptance** (ask the user to run it; you cannot) | `docs/E2E_TESTING.md` "Phase 1 acceptance on physical machines" | Two real machines join, one job runs one rank per machine with the same accuracy as one machine, killing one machine fails the job and marks it LOST; results recorded in the doc |
| 4.2 | Token revocation | `RevokeToken` RPC (admin) + `trainmesh token revoke` + store flag already exists (`TokenRecord.Revoked`) | Revoked token is refused at Join (test) |
| 4.3 | Drain / remove a node | `DrainNode`, `RemoveNode` RPCs; `DRAINING` state already in `internal/membership` | A drained node gets no new jobs; a removed node's heartbeats get NotFound and the agent exits with `ErrRemoved` (tests) |
| 4.4 | Node certificate renewal | `RenewCertificate` RPC (node role); agent renews when < 1/3 of validity remains | Agent keeps working past its first certificate's expiry with an injected clock (test) |
| 4.5 | Optional: mDNS discovery | `tm-agent --discover` finds `meshd` on the LAN | Only if the user wants it; join strings already carry the address |

Rules: every new RPC goes into `methodRoles` (`internal/coordinator/auth.go`) or it is denied;
regenerate with `make generate`; update `docs/modules/mesh.md`.

## Task 5 — Phase 2 (GPU backends, profiler, heterogeneous batches)

Start only after the user reviews Phase 1. Read architecture §5.4–5.6, §6 and §10 (P2). This Mac
has Apple MPS (no NVIDIA/AMD): build and test the device abstraction for `cpu` and `mps`, write
`cuda`/`rocm` paths with unit tests, and mark them "not verified on hardware". Ask the user
before choosing the order of sub-steps.

## Reporting template (use after every task)

```text
Task: <number and name>
Changed: <files>
Verified: <commands run + last lines of output>
Not done / not verified: <honest list>
Question for you: <only if needed>
```
