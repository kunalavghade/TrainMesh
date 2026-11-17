.PHONY: build native generate test test-go test-python test-rust lint fmt demo e2e-local e2e-mesh-local e2e-devices-local e2e-wan e2e-minikube e2e-minikube-mesh e2e-minikube-down

# cargo from rustup, also when only Homebrew's rustup (not ~/.cargo/bin) is installed.
export PATH := $(HOME)/.cargo/bin:/opt/homebrew/opt/rustup/bin:$(PATH)

# trainmesh CLI, meshd coordinator and tm-agent.
build:
	go build -o bin/trainmesh ./cmd/trainmesh
	go build -o bin/meshd ./cmd/meshd
	go build -o bin/tm-agent ./cmd/tm-agent

# Build the tmcomm Rust extension (DiLoCo transport) into .venv; needs cargo and maturin.
native:
	cd native/tmcomm && VIRTUAL_ENV=$(CURDIR)/.venv ../../.venv/bin/maturin develop --release --uv

# Regenerate Go and Python code from proto/ (needs buf, protoc, protoc-gen-go; see docs/CPU_BACKEND.md).
generate:
	buf lint
	PATH="$$(go env GOPATH)/bin:$$PATH" buf generate

test: test-go test-python test-rust

test-go:
	go test -race ./...

test-python:
	PYTHONPATH=python:scripts/e2e .venv/bin/python -m pytest python/tests scripts/e2e -q

test-rust:
	cd native && cargo test

lint:
	test -z "$$(gofmt -l cmd internal)"
	go vet ./...
	golangci-lint run ./...
	.venv/bin/ruff check python scripts
	.venv/bin/black --check python scripts
	cd native && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features python -- -D warnings

fmt:
	gofmt -w $$(find cmd internal -name '*.go')
	.venv/bin/black python scripts
	cd native && cargo fmt

demo:
	go run ./cmd/trainmesh run --config examples/cpu-ddp.yaml

# Single-machine CLI end-to-end test (build, doctor, plan, run, resume, benchmark, rejections).
e2e-local:
	scripts/e2e/local.sh

# Multi-node dev test on a 2-node minikube cluster, with 1-vs-multi machine comparison.
e2e-minikube:
	scripts/e2e/minikube.sh all

e2e-minikube-down:
	scripts/e2e/minikube.sh down

# Phase 1 on one machine: meshd + 2 tm-agents, a job through `trainmesh run --coordinator`, failover.
e2e-mesh-local:
	scripts/e2e/mesh-local.sh

# Phase 2 on an Apple Silicon Mac: one mps agent + one cpu agent, uniform vs throughput balance.
e2e-devices-local:
	scripts/e2e/devices-local.sh

# Phase 3: DiLoCo across 3 islands over tc netem (50 ms RTT, 100 Mbit/s) in a Linux container.
e2e-wan:
	scripts/e2e/wan-docker.sh

# Phase 1 on minikube: meshd + one tm-agent pod per node (needs the cluster from e2e-minikube).
e2e-minikube-mesh:
	scripts/e2e/minikube.sh mesh
