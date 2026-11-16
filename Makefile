.PHONY: build generate test test-go test-python lint fmt demo e2e-local e2e-mesh-local e2e-devices-local e2e-minikube e2e-minikube-mesh e2e-minikube-down

# trainmesh CLI, meshd coordinator and tm-agent.
build:
	go build -o bin/trainmesh ./cmd/trainmesh
	go build -o bin/meshd ./cmd/meshd
	go build -o bin/tm-agent ./cmd/tm-agent

# Regenerate Go and Python code from proto/ (needs buf, protoc, protoc-gen-go; see docs/CPU_BACKEND.md).
generate:
	buf lint
	PATH="$$(go env GOPATH)/bin:$$PATH" buf generate

test: test-go test-python

test-go:
	go test -race ./...

test-python:
	PYTHONPATH=python:scripts/e2e .venv/bin/python -m pytest python/tests scripts/e2e -q

lint:
	test -z "$$(gofmt -l cmd internal)"
	go vet ./...
	golangci-lint run ./...
	.venv/bin/ruff check python scripts
	.venv/bin/black --check python scripts

fmt:
	gofmt -w $$(find cmd internal -name '*.go')
	.venv/bin/black python scripts

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

# Phase 1 on minikube: meshd + one tm-agent pod per node (needs the cluster from e2e-minikube).
e2e-minikube-mesh:
	scripts/e2e/minikube.sh mesh
