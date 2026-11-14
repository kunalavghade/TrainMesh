package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"github.com/spf13/cobra"
	"github.com/trainmesh/trainmesh/internal/config"
)

const version = "0.1.0"

func main() {
	if err := newRoot().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "trainmesh:", err)
		os.Exit(1)
	}
}

func newRoot() *cobra.Command {
	root := &cobra.Command{Use: "trainmesh", Short: "CPU-first distributed AI training engine", SilenceUsage: true}
	root.AddCommand(&cobra.Command{Use: "version", Short: "Print version", Run: func(cmd *cobra.Command, _ []string) { fmt.Fprintln(cmd.OutOrStdout(), version) }})
	root.AddCommand(doctorCommand(), planCommand(), runCommand(), benchmarkCommand(), clusterCommand())
	return root
}

func doctorCommand() *cobra.Command {
	return &cobra.Command{Use: "doctor", Short: "Inspect CPU training prerequisites", RunE: func(cmd *cobra.Command, _ []string) error {
		py := pythonExecutable()
		report := map[string]any{"trainmesh": version, "os": runtime.GOOS, "architecture": runtime.GOARCH, "logical_cpu_count": runtime.NumCPU(), "python": py, "torch": "missing", "distributed": false, "gloo": false, "torchrun": false, "gpu_support": false}
		if py != "" {
			out, err := exec.Command(py, "-c", "import json, torch, torch.distributed.run; print(json.dumps({'torch':torch.__version__,'distributed':torch.distributed.is_available(),'gloo':torch.distributed.is_gloo_available() if torch.distributed.is_available() else False,'torchrun':True}))").Output()
			if err == nil {
				var capabilities map[string]any
				if json.Unmarshal(out, &capabilities) == nil {
					for key, value := range capabilities {
						report[key] = value
					}
				}
			}
		}
		encoded, _ := json.MarshalIndent(report, "", "  ")
		fmt.Fprintln(cmd.OutOrStdout(), string(encoded))
		if report["torch"] == "missing" {
			return errors.New("PyTorch is missing; install CPU PyTorch with `uv pip install --python .venv/bin/python 'torch>=2.2'` after creating the environment (see docs/CPU_BACKEND.md)")
		}
		return nil
	}}
}

func pythonExecutable() string {
	if p := os.Getenv("TRAINMESH_PYTHON"); p != "" {
		return p
	}
	if _, e := os.Stat(".venv/bin/python"); e == nil {
		return filepath.Join(".venv", "bin", "python")
	}
	p, _ := exec.LookPath("python3")
	return p
}

func planCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "plan", Short: "Validate configuration and show an execution plan", RunE: func(cmd *cobra.Command, _ []string) error {
		path, _ := cmd.Flags().GetString("config")
		c, err := config.Load(path)
		if err != nil {
			return err
		}
		plan := map[string]any{"strategy": c.Execution.Strategy, "workers_requested": c.Execution.Workers, "processes_per_worker": c.Execution.ProcessesPerWorker, "world_size": c.WorldSize(), "per_rank_batch": c.Training.Batch.PerRank, "gradient_accumulation_steps": c.Training.Batch.GradientAccumulationSteps, "effective_global_batch": c.GlobalBatchSize(), "dataset_samples": c.Training.Dataset.Samples, "estimated_epochs": c.Training.Epochs, "device": "cpu", "communication": "gloo", "execution": "single-host torchrun", "multi_machine_supported": false}
		if c.Execution.Workers > 1 {
			plan["execution"] = "not executable: multi-machine agent orchestration is not implemented in this release"
		}
		b, _ := json.MarshalIndent(plan, "", "  ")
		fmt.Fprintln(cmd.OutOrStdout(), string(b))
		return nil
	}}
	cmd.Flags().String("config", "examples/cpu-ddp.yaml", "YAML training configuration")
	return cmd
}

func runCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "run", Short: "Run a real PyTorch CPU/Gloo training job", RunE: func(cmd *cobra.Command, _ []string) error {
		path, _ := cmd.Flags().GetString("config")
		c, err := config.Load(path)
		if err != nil {
			return err
		}
		if c.Execution.Workers != 1 {
			return errors.New("multi-machine execution is not yet implemented; set execution.workers: 1. See docs/MULTI_MACHINE_MANUAL_TEST.md for the physical-machine test status and roadmap")
		}
		n := c.Execution.ProcessesPerWorker
		if n > runtime.NumCPU() {
			return fmt.Errorf("requested %d local processes on %d logical CPUs; reduce processes_per_worker to avoid CPU oversubscription", n, runtime.NumCPU())
		}
		py := pythonExecutable()
		if py == "" {
			return errors.New("python3 not found")
		}
		absConfig, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		modulePath, err := filepath.Abs("python")
		if err != nil {
			return err
		}
		args := []string{"-m", "trainmesh_runtime", "--config", absConfig}
		if n > 1 {
			port, err := freeTCPPort()
			if err != nil {
				return err
			}
			args = []string{"-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0", "--nproc_per_node", fmt.Sprint(n), "--master_addr=127.0.0.1", "--master_port=" + port, "-m", "trainmesh_runtime", "--config", absConfig}
		}
		resume, _ := cmd.Flags().GetString("resume")
		if resume != "" {
			args = append(args, "--resume", resume)
		}
		process := exec.CommandContext(cmd.Context(), py, args...)
		process.Stdout = cmd.OutOrStdout()
		process.Stderr = cmd.ErrOrStderr()
		env := os.Environ()
		env = append(env, "PYTHONPATH="+modulePath+string(os.PathListSeparator)+os.Getenv("PYTHONPATH"), "OMP_NUM_THREADS=1", "USE_LIBUV=0")
		if runtime.GOOS == "darwin" && os.Getenv("GLOO_SOCKET_IFNAME") == "" {
			env = append(env, "GLOO_SOCKET_IFNAME=lo0")
		}
		process.Env = env
		fmt.Fprintf(cmd.OutOrStdout(), "Starting CPU/Gloo training: world_size=%d effective_global_batch=%d\n", c.WorldSize(), c.GlobalBatchSize())
		return process.Run()
	}}
	cmd.Flags().String("config", "examples/cpu-ddp.yaml", "YAML training configuration")
	cmd.Flags().String("resume", "", "resume from a verified checkpoint directory")
	return cmd
}

func benchmarkCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "benchmark", Short: "Run a measured local CPU training experiment", RunE: func(cmd *cobra.Command, _ []string) error {
		path, _ := cmd.Flags().GetString("config")
		c, err := config.Load(path)
		if err != nil {
			return err
		}
		if c.Execution.Workers != 1 {
			return errors.New("benchmark currently runs one host only; physical multi-machine benchmark is not available")
		}
		py := pythonExecutable()
		if py == "" {
			return errors.New("python3 not found")
		}
		absConfig, _ := filepath.Abs(path)
		modulePath, _ := filepath.Abs("python")
		out, _ := cmd.Flags().GetString("output")
		args := []string{"-m", "trainmesh_runtime", "--config", absConfig}
		if c.Execution.ProcessesPerWorker > 1 {
			port, err := freeTCPPort()
			if err != nil {
				return err
			}
			args = []string{"-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0", "--nproc_per_node", fmt.Sprint(c.Execution.ProcessesPerWorker), "--master_addr=127.0.0.1", "--master_port=" + port, "-m", "trainmesh_runtime", "--config", absConfig}
		}
		if out != "" {
			args = append(args, "--output", out)
		}
		p := exec.CommandContext(cmd.Context(), py, args...)
		p.Stdout = cmd.OutOrStdout()
		p.Stderr = cmd.ErrOrStderr()
		p.Env = append(os.Environ(), "PYTHONPATH="+modulePath+string(os.PathListSeparator)+os.Getenv("PYTHONPATH"), "OMP_NUM_THREADS=1", "USE_LIBUV=0")
		if runtime.GOOS == "darwin" && os.Getenv("GLOO_SOCKET_IFNAME") == "" {
			p.Env = append(p.Env, "GLOO_SOCKET_IFNAME=lo0")
		}
		return p.Run()
	}}
	cmd.Flags().String("config", "examples/cpu-ddp.yaml", "YAML training configuration")
	cmd.Flags().String("output", "benchmark-results/latest.json", "JSON result file")
	return cmd
}

type clusterState struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Created string `json:"created"`
}

func clusterCommand() *cobra.Command {
	root := &cobra.Command{Use: "cluster", Short: "Manage local TrainMesh cluster metadata"}
	root.AddCommand(&cobra.Command{Use: "init", Short: "Initialize cluster identity on this machine", RunE: func(cmd *cobra.Command, _ []string) error {
		if err := os.MkdirAll(".trainmesh", 0700); err != nil {
			return err
		}
		statePath := ".trainmesh/cluster.json"
		if _, err := os.Stat(statePath); err == nil {
			return errors.New("cluster already initialized (.trainmesh/cluster.json exists)")
		}
		idBytes := make([]byte, 16)
		if _, err := rand.Read(idBytes); err != nil {
			return err
		}
		state := clusterState{ID: hex.EncodeToString(idBytes), Name: "local-cpu-cluster", Created: time.Now().UTC().Format(time.RFC3339)}
		b, _ := json.MarshalIndent(state, "", "  ")
		if err := os.WriteFile(statePath, append(b, '\n'), 0600); err != nil {
			return err
		}
		fmt.Fprintf(cmd.OutOrStdout(), "Initialized cluster %s (%s)\n", state.Name, state.ID)
		return nil
	}})
	root.AddCommand(&cobra.Command{Use: "status", Short: "Show local cluster identity and host resources", RunE: func(cmd *cobra.Command, _ []string) error {
		b, err := os.ReadFile(".trainmesh/cluster.json")
		if err != nil {
			return errors.New("cluster is not initialized; run `trainmesh cluster init`")
		}
		var state clusterState
		if err = json.Unmarshal(b, &state); err != nil {
			return err
		}
		report := map[string]any{"cluster": state, "local_machine": map[string]any{"hostname": hostname(), "os": runtime.GOOS, "architecture": runtime.GOARCH, "logical_cpu_count": runtime.NumCPU()}, "registered_workers": 0, "training_runs": 0, "note": "worker enrollment and remote agent orchestration are not implemented yet"}
		out, _ := json.MarshalIndent(report, "", "  ")
		fmt.Fprintln(cmd.OutOrStdout(), string(out))
		return nil
	}})
	return root
}

func hostname() string { v, _ := os.Hostname(); return v }

func freeTCPPort() (string, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", fmt.Errorf("select local rendezvous port: %w", err)
	}
	defer listener.Close()
	return fmt.Sprint(listener.Addr().(*net.TCPAddr).Port), nil
}
