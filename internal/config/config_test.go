package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func valid() Config {
	return Config{
		Version:   1,
		Execution: Execution{Strategy: "data_parallel", Workers: 2, ProcessesPerWorker: 1, Backend: Backend{Device: "cpu", Communication: "gloo"}},
		Training:  Training{Dataset: Dataset{Type: "synthetic", Samples: 1024}, Batch: Batch{PerRank: 16, GradientAccumulationSteps: 2}, Epochs: 1, Precision: "fp32"},
	}
}

func TestGlobalBatchSize(t *testing.T) {
	if got := valid().GlobalBatchSize(); got != 64 {
		t.Fatalf("got %d, want 64", got)
	}
}

func TestValidateRejects(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
		want   string
	}{
		{"gpu backend", func(c *Config) { c.Execution.Backend.Device = "cuda" }, "device=cpu"},
		{"zero batch", func(c *Config) { c.Training.Batch.PerRank = 0 }, "per-rank batch"},
		{"negative threads", func(c *Config) { c.Execution.CPUThreadsPerRank = -1 }, "cpu_threads_per_rank"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := valid()
			tt.mutate(&c)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, tt.want)
			}
		})
	}
}

func TestValidateReportsEveryProblem(t *testing.T) {
	c := valid()
	c.Version = 2
	c.Training.Precision = "fp16"
	err := c.Validate()
	if err == nil || !strings.Contains(err.Error(), "version") || !strings.Contains(err.Error(), "fp32") {
		t.Fatalf("Validate() = %v, want both version and precision problems", err)
	}
}

func TestLoadMigratesLegacyOptimizerKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	yaml := `version: 1
execution: {strategy: data_parallel, workers: 1, processes_per_worker: 1, backend: {device: cpu, communication: gloo}}
training: {dataset: {type: synthetic, samples: 8}, batch: {per_rank: 2, gradient_accumulation_steps: 1}, epochs: 1, precision: fp32}
optimizer: {enabled: false, results: out/result.json}
`
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	c, err := Load(path)

	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if !c.UsesLegacyOptimizerKey() {
		t.Error("expected legacy optimizer key to be detected")
	}
	if c.Run.ResultsPath != "out/result.json" {
		t.Errorf("ResultsPath = %q, want out/result.json", c.Run.ResultsPath)
	}
	if _, hasResults := c.Planner["results"]; hasResults || c.Planner["enabled"] != false {
		t.Errorf("Planner = %v, want {enabled:false} without results", c.Planner)
	}
}
