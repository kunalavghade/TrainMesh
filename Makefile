.PHONY: build test test-go test-python fmt demo

build:
	go build -o bin/trainmesh ./cmd/trainmesh

test: test-go test-python

test-go:
	go test ./...

test-python:
	PYTHONPATH=python .venv/bin/python -m pytest python/tests -q

fmt:
	gofmt -w $$(find cmd internal -name '*.go')

demo:
	go run ./cmd/trainmesh run --config examples/cpu-ddp.yaml
