.PHONY: build generate test test-go test-python lint fmt demo e2e-local e2e-minikube e2e-minikube-down

build:
	go build -o bin/trainmesh ./cmd/trainmesh

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
