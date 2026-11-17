package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/trainmesh/trainmesh/internal/config"
	"github.com/trainmesh/trainmesh/internal/launcher"
)

// Files the runner writes in each job directory.
const (
	jobConfigFile = "config.yaml"
	jobResultFile = "result.json"
	jobLogFile    = "runtime.log"
)

// Assignment is this machine's part of a job, as sent by the coordinator. For a diloco job,
// NodeRank, Machines and Master* describe this node's island.
type Assignment struct {
	JobID           string
	NodeRank        int
	Machines        int
	RanksPerMachine int
	MasterAddr      string
	MasterPort      int
	ConfigYAML      []byte
	Island          int
	IslandPeers     []IslandPeer
	TmcommPort      int
}

// IslandPeer is one island leader's tmcomm endpoint.
type IslandPeer struct {
	Island  int    `json:"island"`
	Address string `json:"address"`
	Name    string `json:"name"`
}

// TLSFiles are the node identity files tmcomm uses for mTLS between island leaders.
type TLSFiles struct {
	Cert, Key, CA string
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
	Device    string // device offered to the mesh; used by jobs that ask for device=auto
	Address   string // advertised IP; an island leader's tmcomm listens on it
	TLS       TLSFiles
}

// Environment the runtime reads (see trainmesh_runtime/devices.py and diloco.py).
const (
	deviceEnv      = "TRAINMESH_DEVICE"
	islandEnv      = "TRAINMESH_ISLAND"
	islandPeersEnv = "TRAINMESH_ISLAND_PEERS"
	tmcommListen   = "TRAINMESH_TMCOMM_LISTEN"
	tlsCertEnv     = "TRAINMESH_TLS_CERT"
	tlsKeyEnv      = "TRAINMESH_TLS_KEY"
	tlsCAEnv       = "TRAINMESH_TLS_CA"
)

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
	env, err := r.environment(a)
	if err != nil {
		return nil, err
	}
	if err := launcher.Run(ctx, spec, env, runtime.GOOS); err != nil {
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

// environment is the agent's environment plus what this assignment tells the runtime.
func (r TorchRunner) environment(a Assignment) ([]string, error) {
	env := os.Environ()
	if r.Device != "" {
		env = append(env, deviceEnv+"="+r.Device)
	}
	if len(a.IslandPeers) == 0 {
		return env, nil
	}
	peers, err := json.Marshal(a.IslandPeers)
	if err != nil {
		return nil, fmt.Errorf("encoding island peers: %w", err)
	}
	env = append(env,
		islandEnv+"="+strconv.Itoa(a.Island),
		islandPeersEnv+"="+string(peers),
		tmcommListen+"="+net.JoinHostPort(r.listenHost(), strconv.Itoa(a.TmcommPort)),
	)
	if r.TLS.Cert != "" {
		env = append(env, tlsCertEnv+"="+r.TLS.Cert, tlsKeyEnv+"="+r.TLS.Key, tlsCAEnv+"="+r.TLS.CA)
	}
	return env, nil
}

func (r TorchRunner) listenHost() string {
	if r.Address != "" {
		return r.Address
	}
	return "0.0.0.0"
}
