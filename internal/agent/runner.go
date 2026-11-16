package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/trainmesh/trainmesh/internal/config"
	"github.com/trainmesh/trainmesh/internal/launcher"
)

// Files the runner writes in each job directory.
const (
	jobConfigFile = "config.yaml"
	jobResultFile = "result.json"
	jobLogFile    = "runtime.log"
)

// Assignment is this machine's part of a job, as sent by the coordinator.
type Assignment struct {
	JobID           string
	NodeRank        int
	Machines        int
	RanksPerMachine int
	MasterAddr      string
	MasterPort      int
	ConfigYAML      []byte
}

// Runner executes an assignment and returns rank 0's result JSON (nil on other nodes).
type Runner interface {
	Run(ctx context.Context, assignment Assignment) ([]byte, error)
}

// TorchRunner runs assignments with torchrun through the launcher, one directory per job.
type TorchRunner struct {
	Runtime   launcher.Runtime
	JobsDir   string
	Interface string // NIC for Gloo; set from route detection
}

// Run writes the config, launches this node's ranks and waits. Output goes to runtime.log.
func (r TorchRunner) Run(ctx context.Context, a Assignment) ([]byte, error) {
	jobDir := filepath.Join(r.JobsDir, a.JobID)
	if err := os.MkdirAll(jobDir, 0o700); err != nil {
		return nil, fmt.Errorf("creating job dir: %w", err)
	}
	configPath := filepath.Join(jobDir, jobConfigFile)
	if err := os.WriteFile(configPath, a.ConfigYAML, 0o600); err != nil {
		return nil, fmt.Errorf("writing job config: %w", err)
	}
	parsed, err := config.Parse(a.ConfigYAML)
	if err != nil {
		return nil, err
	}
	logFile, err := os.Create(filepath.Join(jobDir, jobLogFile)) // #nosec G304 -- inside the agent's jobs dir
	if err != nil {
		return nil, fmt.Errorf("creating runtime log: %w", err)
	}
	defer func() { _ = logFile.Close() }()

	resultPath := filepath.Join(jobDir, jobResultFile)
	spec := launcher.Spec{
		Python:         r.Runtime.Python,
		RuntimeDir:     r.Runtime.RuntimeDir,
		ConfigPath:     configPath,
		LocalRanks:     a.RanksPerMachine,
		ThreadsPerRank: launcher.ResolveThreadsPerRank(int(parsed.Spec.GetExecution().GetCpuThreadsPerRank()), os.Getenv("OMP_NUM_THREADS")),
		ExtraArgs:      []string{"--output", resultPath},
		Nodes:          a.Machines,
		NodeRank:       a.NodeRank,
		MasterAddr:     a.MasterAddr,
		MasterPort:     a.MasterPort,
		Interface:      r.Interface,
		WorkDir:        jobDir,
		Stdout:         logFile,
		Stderr:         logFile,
	}
	if err := launcher.Run(ctx, spec, os.Environ(), runtime.GOOS); err != nil {
		return nil, fmt.Errorf("%w (log: %s)", err, logFile.Name())
	}
	if a.NodeRank != 0 {
		return nil, nil
	}
	result, err := os.ReadFile(resultPath) // #nosec G304 -- inside the agent's jobs dir
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("rank 0 finished without writing %s", resultPath)
	}
	return result, err
}
