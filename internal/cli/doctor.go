package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"

	"github.com/spf13/cobra"
	"github.com/trainmesh/trainmesh/internal/launcher"
)

const probeModule = "trainmesh_runtime.probe"

func doctorCommand() *cobra.Command {
	return &cobra.Command{Use: "doctor", Short: "Inspect CPU training prerequisites", RunE: func(cmd *cobra.Command, _ []string) error {
		h, err := currentHost()
		if err != nil {
			return err
		}
		report := map[string]any{
			"trainmesh": Version, "os": h.goos, "logical_cpu_count": h.logicalCPUs,
			"python": "", "runtime_dir": "", "torch": "missing", "gpu_support": false,
		}
		rt, discoverErr := launcher.DiscoverRuntime(h.getenv, h.cwd, h.execDir)
		if discoverErr == nil {
			report["python"], report["runtime_dir"] = rt.Python, rt.RuntimeDir
			for key, value := range probeCapabilities(cmd, rt, h) {
				report[key] = value
			}
		}
		if err := writeJSON(cmd.OutOrStdout(), report); err != nil {
			return err
		}
		if discoverErr != nil {
			return discoverErr
		}
		if report["torch"] == "missing" {
			return errors.New("PyTorch is missing; install CPU PyTorch with `uv pip install --python .venv/bin/python -e 'python[test]'` (see docs/CPU_BACKEND.md)")
		}
		return nil
	}}
}

func probeCapabilities(cmd *cobra.Command, rt launcher.Runtime, h host) map[string]any {
	probe := exec.CommandContext(cmd.Context(), rt.Python, "-m", probeModule)
	probe.Env = launcher.Spec{RuntimeDir: rt.RuntimeDir, ThreadsPerRank: 1}.Env(h.environ, h.goos)
	out, err := probe.Output()
	if err != nil {
		fmt.Fprintf(cmd.ErrOrStderr(), "warning: capability probe failed: %v\n", err)
		return nil
	}
	var capabilities map[string]any
	if err := json.Unmarshal(out, &capabilities); err != nil {
		fmt.Fprintf(cmd.ErrOrStderr(), "warning: capability probe returned invalid JSON: %v\n", err)
		return nil
	}
	return capabilities
}
