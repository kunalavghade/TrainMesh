package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

const testConfig = `version: 1
execution: {strategy: data_parallel, workers: %d, processes_per_worker: 1, backend: {device: cpu, communication: gloo}}
training: {dataset: {type: synthetic, samples: 8}, batch: {per_rank: 2, gradient_accumulation_steps: 3}, epochs: 1, precision: fp32}
`

func writeConfig(t *testing.T, workers int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	content := []byte(fmt.Sprintf(testConfig, workers))
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPlanPrintsResolvedBatch(t *testing.T) {
	var out bytes.Buffer
	root := NewRoot()
	root.SetOut(&out)
	root.SetArgs([]string{"plan", "--config", writeConfig(t, 1)})

	if err := root.Execute(); err != nil {
		t.Fatalf("plan error = %v", err)
	}

	var plan map[string]any
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("plan output is not JSON: %v\n%s", err, out.String())
	}
	if plan["effective_global_batch"] != float64(6) || plan["world_size"] != float64(1) {
		t.Errorf("plan = %v, want effective_global_batch 6 and world_size 1", plan)
	}
}

func TestRunRejectsMultiMachine(t *testing.T) {
	root := NewRoot()
	root.SetOut(&bytes.Buffer{})
	root.SetErr(&bytes.Buffer{})
	root.SetArgs([]string{"run", "--config", writeConfig(t, 2)})

	if err := root.Execute(); !errors.Is(err, ErrMultiMachineUnsupported) {
		t.Fatalf("run error = %v, want ErrMultiMachineUnsupported", err)
	}
}
