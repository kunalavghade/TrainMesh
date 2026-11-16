package cli

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/spf13/cobra"
	"github.com/trainmesh/trainmesh/internal/config"
	"github.com/trainmesh/trainmesh/internal/launcher"
)

const defaultConfigPath = "examples/cpu-ddp.yaml"

// ErrMultiMachineUnsupported is returned until the agent-based multi-node launcher exists.
var ErrMultiMachineUnsupported = errors.New("execution.workers > 1 runs across machines: start meshd, join the machines with tm-agent, then pass --coordinator host:port (see docs/modules/mesh.md)")

func planCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "plan", Short: "Validate configuration and show an execution plan", RunE: func(cmd *cobra.Command, _ []string) error {
		path, _ := cmd.Flags().GetString("config")
		c, err := loadConfig(cmd, path)
		if err != nil {
			return err
		}
		h, err := currentHost()
		if err != nil {
			return err
		}
		plan := map[string]any{
			"strategy":                    c.Spec.GetExecution().GetStrategy(),
			"workers_requested":           c.Spec.GetExecution().GetWorkers(),
			"processes_per_worker":        c.Spec.GetExecution().GetProcessesPerWorker(),
			"threads_per_rank":            threadsPerRank(c, h),
			"world_size":                  c.WorldSize(),
			"per_rank_batch":              c.Spec.GetTraining().GetBatch().GetPerRank(),
			"gradient_accumulation_steps": c.AccumulationSteps(),
			"effective_global_batch":      c.GlobalBatchSize(),
			"dataset_samples":             c.Spec.GetTraining().GetDataset().GetSamples(),
			"estimated_epochs":            c.Spec.GetTraining().GetEpochs(),
			"device":                      "cpu",
			"communication":               "gloo",
			"execution":                   "single-host torchrun (standalone rendezvous)",
			"multi_machine_supported":     false,
		}
		if c.Spec.GetExecution().GetWorkers() > 1 {
			plan["execution"] = "not executable: multi-machine agent orchestration is not implemented in this release"
		}
		return writeJSON(cmd.OutOrStdout(), plan)
	}}
	cmd.Flags().String("config", defaultConfigPath, "YAML training configuration")
	return cmd
}

func runCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "run", Short: "Run a real PyTorch CPU/Gloo training job", RunE: func(cmd *cobra.Command, _ []string) error {
		path, _ := cmd.Flags().GetString("config")
		if coordinator, _ := cmd.Flags().GetString("coordinator"); coordinator != "" {
			return runOnMesh(cmd, path)
		}
		resume, _ := cmd.Flags().GetString("resume")
		var extra []string
		if resume != "" {
			extra = []string{"--resume", resume}
		}
		return launchTraining(cmd, path, extra)
	}}
	cmd.Flags().String("config", defaultConfigPath, "YAML training configuration")
	cmd.Flags().String("resume", "", "resume from a verified checkpoint directory")
	addAdminFlags(cmd, "")
	cmd.Flags().Lookup("coordinator").Usage = "run across machines through this meshd (host:port); empty runs on this machine"
	cmd.Flags().String("output", "", "with --coordinator: also write rank 0's result JSON here")
	cmd.Flags().Duration("poll-interval", defaultPollInterval, "with --coordinator: how often to check the job")
	cmd.Flags().Duration("timeout", defaultJobTimeout, "with --coordinator: cancel the job if it runs longer")
	return cmd
}

func benchmarkCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "benchmark", Short: "Run a measured local CPU training experiment", RunE: func(cmd *cobra.Command, _ []string) error {
		path, _ := cmd.Flags().GetString("config")
		output, _ := cmd.Flags().GetString("output")
		var extra []string
		if output != "" {
			extra = []string{"--output", output}
		}
		return launchTraining(cmd, path, extra)
	}}
	cmd.Flags().String("config", defaultConfigPath, "YAML training configuration")
	cmd.Flags().String("output", "benchmark-results/latest.json", "JSON result file")
	return cmd
}

func launchTraining(cmd *cobra.Command, configPath string, extraArgs []string) error {
	c, err := loadConfig(cmd, configPath)
	if err != nil {
		return err
	}
	if c.Spec.GetExecution().GetWorkers() != 1 {
		return ErrMultiMachineUnsupported
	}
	h, err := currentHost()
	if err != nil {
		return err
	}
	localRanks := int(c.Spec.GetExecution().GetProcessesPerWorker())
	if localRanks > h.logicalCPUs {
		return fmt.Errorf("requested %d local processes on %d logical CPUs; reduce processes_per_worker to avoid CPU oversubscription", localRanks, h.logicalCPUs)
	}
	threads := threadsPerRank(c, h)
	if localRanks*threads > h.logicalCPUs {
		return fmt.Errorf("%d ranks x %d threads exceeds %d logical CPUs; lower execution.cpu_threads_per_rank", localRanks, threads, h.logicalCPUs)
	}
	rt, err := launcher.DiscoverRuntime(h.getenv, h.cwd, h.execDir)
	if err != nil {
		return err
	}
	absConfig, err := filepath.Abs(configPath)
	if err != nil {
		return fmt.Errorf("resolving config path %s: %w", configPath, err)
	}

	spec := launcher.Spec{
		Python:         rt.Python,
		RuntimeDir:     rt.RuntimeDir,
		ConfigPath:     absConfig,
		LocalRanks:     localRanks,
		ThreadsPerRank: threads,
		ExtraArgs:      extraArgs,
		Stdout:         cmd.OutOrStdout(),
		Stderr:         cmd.ErrOrStderr(),
	}
	fmt.Fprintf(cmd.OutOrStdout(), "Starting CPU/Gloo training: world_size=%d threads_per_rank=%d effective_global_batch=%d\n",
		c.WorldSize(), spec.ThreadsPerRank, c.GlobalBatchSize())
	return launcher.Run(cmd.Context(), spec, h.environ, h.goos)
}

func loadConfig(cmd *cobra.Command, path string) (config.Config, error) {
	c, err := config.Load(path)
	if err != nil {
		return c, err
	}
	if c.UsesLegacyOptimizerKey() {
		fmt.Fprintln(cmd.ErrOrStderr(), "warning: config key 'optimizer' is deprecated; rename it to 'planner' and move 'results' to run.results_path")
	}
	return c, nil
}

func threadsPerRank(c config.Config, h host) int {
	return launcher.ResolveThreadsPerRank(int(c.Spec.GetExecution().GetCpuThreadsPerRank()), h.getenv("OMP_NUM_THREADS"))
}
