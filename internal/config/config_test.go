package config

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const validYAML = `version: 1
execution:
  strategy: data_parallel
  workers: 2
  processes_per_worker: 1
  backend: {device: cpu, communication: gloo}
training:
  dataset: {type: synthetic, samples: 1024}
  batch: {per_rank: 16, gradient_accumulation_steps: 2}
  epochs: 1
  precision: fp32
`

func mustParse(t *testing.T, yaml string) Config {
	t.Helper()
	c, err := Parse([]byte(yaml))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return c
}

func TestGlobalBatchSize(t *testing.T) {
	if got := mustParse(t, validYAML).GlobalBatchSize(); got != 64 {
		t.Fatalf("got %d, want 64", got)
	}
}

func TestAccumulationStepsDefaultsToOne(t *testing.T) {
	c := mustParse(t, strings.Replace(validYAML, ", gradient_accumulation_steps: 2", "", 1))
	if got := c.AccumulationSteps(); got != 1 {
		t.Fatalf("AccumulationSteps() = %d, want 1", got)
	}
}

func TestParseRejects(t *testing.T) {
	tests := []struct {
		name    string
		replace [2]string
		want    string
	}{
		{"unknown device", [2]string{"device: cpu", "device: tpu"}, "execution.backend.device must be cpu, mps, cuda or auto"},
		{"nccl without cuda", [2]string{"communication: gloo", "communication: nccl"}, "communication=nccl needs device=cuda"},
		{"unknown precision", [2]string{"precision: fp32", "precision: fp8"}, "training.precision must be fp32, bf16, fp16 or auto"},
		{"unknown balance", [2]string{"per_rank: 16", "per_rank: 16, balance: random"}, "training.batch.balance must be uniform or throughput"},
		{"zero rank batch", [2]string{"per_rank: 16", "per_rank: 16, rank_batch_sizes: [8, 0]"}, "training.batch.rank_batch_sizes"},
		{"zero batch", [2]string{"per_rank: 16", "per_rank: 0"}, "training.batch.per_rank"},
		{"negative threads", [2]string{"processes_per_worker: 1", "processes_per_worker: 1\n  cpu_threads_per_rank: -1"}, "execution.cpu_threads_per_rank"},
		{"unknown dataset", [2]string{"type: synthetic", "type: imagenet"}, "training.dataset.type must be synthetic or teacher"},
		{"unknown key", [2]string{"epochs: 1", "epochs: 1\n  epoch: 2"}, "unknown field"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse([]byte(strings.Replace(validYAML, tt.replace[0], tt.replace[1], 1)))
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Parse() = %v, want ErrInvalid containing %q", err, tt.want)
			}
		})
	}
}

func TestParseReportsEveryProblem(t *testing.T) {
	broken := strings.NewReplacer("version: 1", "version: 2", "precision: fp32", "precision: fp8").Replace(validYAML)
	_, err := Parse([]byte(broken))
	if err == nil || !strings.Contains(err.Error(), "version must be 1") || !strings.Contains(err.Error(), "training.precision must be") {
		t.Fatalf("Parse() = %v, want both version and precision problems", err)
	}
}

func TestLoadMigratesLegacyOptimizerKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	legacy := validYAML + "optimizer: {enabled: false, results: out/result.json}\n"
	if err := os.WriteFile(path, []byte(legacy), 0o600); err != nil {
		t.Fatal(err)
	}

	c, err := Load(path)

	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if !c.UsesLegacyOptimizerKey() {
		t.Error("expected legacy optimizer key to be detected")
	}
	if got := c.Spec.GetRun().GetResultsPath(); got != "out/result.json" {
		t.Errorf("results_path = %q, want out/result.json", got)
	}
	planner := c.Spec.GetPlanner().AsMap()
	if _, hasResults := planner["results"]; hasResults || planner["enabled"] != false {
		t.Errorf("planner = %v, want {enabled:false} without results", planner)
	}
}

func TestAcceleratorConfigsAreValid(t *testing.T) {
	for _, backend := range []string{"device: cuda, communication: nccl", "device: mps, communication: gloo", "device: auto, communication: auto"} {
		if _, err := Parse([]byte(strings.Replace(validYAML, "device: cpu, communication: gloo", backend, 1))); err != nil {
			t.Errorf("%s rejected: %v", backend, err)
		}
	}
}

func TestExampleConfigIsValid(t *testing.T) {
	if _, err := Load(filepath.Join("..", "..", "examples", "cpu-ddp.yaml")); err != nil {
		t.Fatalf("examples/cpu-ddp.yaml: %v", err)
	}
}
