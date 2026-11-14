.PHONY: build test test-go test-python lint fmt demo

build:
	go build -o bin/trainmesh ./cmd/trainmesh

test: test-go test-python

test-go:
	go test -race ./...

test-python:
	PYTHONPATH=python .venv/bin/python -m pytest python/tests -q

lint:
	test -z "$$(gofmt -l cmd internal)"
	go vet ./...
	golangci-lint run ./...
	.venv/bin/ruff check python
	.venv/bin/black --check python

fmt:
	gofmt -w $$(find cmd internal -name '*.go')
	.venv/bin/black python

demo:
	go run ./cmd/trainmesh run --config examples/cpu-ddp.yaml
