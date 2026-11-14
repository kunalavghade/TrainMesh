package config

import "testing"

func valid() Config {
	return Config{Version: 1, Execution: Execution{Strategy: "data_parallel", Workers: 2, ProcessesPerWorker: 1, Backend: Backend{Device: "cpu", Communication: "gloo"}}, Training: Training{Dataset: Dataset{Type: "synthetic", Samples: 1024}, Batch: Batch{PerRank: 16, GradientAccumulationSteps: 2}, Epochs: 1, Precision: "fp32"}}
}
func TestGlobalBatchSize(t *testing.T) {
	c := valid()
	if got := c.GlobalBatchSize(); got != 64 {
		t.Fatalf("got %d, want 64", got)
	}
}
func TestRejectsGPUBackend(t *testing.T) {
	c := valid()
	c.Execution.Backend.Device = "cuda"
	if c.Validate() == nil {
		t.Fatal("expected unsupported device to be rejected")
	}
}
func TestRejectsInvalidBatch(t *testing.T) {
	c := valid()
	c.Training.Batch.PerRank = 0
	if c.Validate() == nil {
		t.Fatal("expected invalid batch to be rejected")
	}
}
