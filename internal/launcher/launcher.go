// Package launcher starts the Python training runtime on this host as a supervised process group.
package launcher

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	runtimeModule = "trainmesh_runtime"
	// DefaultGracePeriod is how long ranks get to exit after SIGTERM before SIGKILL.
	DefaultGracePeriod = 10 * time.Second
	// maxPortAttempts bounds relaunches when another process grabs the chosen rendezvous port.
	maxPortAttempts = 3
	loopbackAddr    = "127.0.0.1"
)

// portConflictMarker is how c10d reports a rendezvous port taken between selection and bind.
var portConflictMarker = []byte("address already in use")

// loopbackInterfaceByOS pins Gloo to loopback for single-host jobs where auto-detection misbehaves.
var loopbackInterfaceByOS = map[string]string{"darwin": "lo0"}

// Spec fully describes one training launch on this host. For a multi-machine job (Nodes > 1)
// the coordinator supplies NodeRank, MasterAddr and MasterPort, and every machine runs one Spec.
type Spec struct {
	Python         string   // interpreter path
	RuntimeDir     string   // directory that contains the trainmesh_runtime package
	ConfigPath     string   // absolute path to the job config
	LocalRanks     int      // processes on this host
	ThreadsPerRank int      // intra-op CPU threads per rank
	ExtraArgs      []string // forwarded to the runtime, e.g. --resume, --output
	Nodes          int      // machines in the job; 0 or 1 means this host only
	NodeRank       int      // this machine's rank among Nodes
	MasterAddr     string   // rendezvous host (node rank 0); defaults to loopback
	MasterPort     int      // static rendezvous port; chosen by Run for single-host jobs
	Interface      string   // NIC for Gloo traffic (GLOO_SOCKET_IFNAME); empty = OS default
	WorkDir        string   // working directory of the runtime (relative checkpoint paths)
	Stdout         io.Writer
	Stderr         io.Writer
	GracePeriod    time.Duration
}

// Args builds the interpreter arguments. Static rendezvous is used instead of --standalone
// because the c10d rendezvous store ignores USE_LIBUV=0 and fails on macOS.
func (s Spec) Args() []string {
	runtimeArgs := append([]string{"-m", runtimeModule, "--config", s.ConfigPath}, s.ExtraArgs...)
	if !s.usesTorchrun() {
		return runtimeArgs
	}
	master := s.MasterAddr
	if master == "" {
		master = loopbackAddr
	}
	torchrun := []string{
		"-m", "torch.distributed.run",
		"--nnodes=" + strconv.Itoa(max(1, s.Nodes)),
		"--node_rank=" + strconv.Itoa(s.NodeRank),
		"--nproc_per_node=" + strconv.Itoa(max(1, s.LocalRanks)),
		"--master_addr=" + master,
		"--master_port=" + strconv.Itoa(s.MasterPort),
	}
	return append(torchrun, runtimeArgs...)
}

// Env returns base plus the runtime variables; later entries win in os/exec.
func (s Spec) Env(base []string, goos string) []string {
	env := append([]string{}, base...)
	pythonPath := s.RuntimeDir
	if existing := lookupEnv(base, "PYTHONPATH"); existing != "" {
		pythonPath += string(listSeparator(goos)) + existing
	}
	env = append(env,
		"PYTHONPATH="+pythonPath,
		"OMP_NUM_THREADS="+strconv.Itoa(max(1, s.ThreadsPerRank)),
		"USE_LIBUV=0",
	)
	switch iface, hasLoopbackDefault := loopbackInterfaceByOS[goos]; {
	case s.Interface != "":
		env = append(env, "GLOO_SOCKET_IFNAME="+s.Interface)
	case hasLoopbackDefault && !s.isMultiNode() && lookupEnv(base, "GLOO_SOCKET_IFNAME") == "":
		env = append(env, "GLOO_SOCKET_IFNAME="+iface)
	}
	return env
}

func (s Spec) isMultiNode() bool  { return s.Nodes > 1 }
func (s Spec) usesTorchrun() bool { return s.LocalRanks > 1 || s.isMultiNode() }

// DefaultThreadsPerRank matches torchrun's default. On the reference 12-core host more intra-op
// threads were 5-10% slower for small models and at most ~7% faster at hidden=4096.
const DefaultThreadsPerRank = 1

// ResolveThreadsPerRank picks threads: explicit config, then OMP_NUM_THREADS, then the default.
func ResolveThreadsPerRank(configured int, ompEnv string) int {
	if configured > 0 {
		return configured
	}
	if n, err := strconv.Atoi(strings.TrimSpace(ompEnv)); err == nil && n > 0 {
		return n
	}
	return DefaultThreadsPerRank
}

// Run starts the runtime and supervises it until it exits or ctx is cancelled. A single-host
// multi-rank launch whose rendezvous port was taken before torchrun bound it is retried on a new
// port; multi-machine jobs use the coordinator's port as given.
func Run(ctx context.Context, s Spec, base []string, goos string) error {
	grace := s.GracePeriod
	if grace <= 0 {
		grace = DefaultGracePeriod
	}
	if !s.usesTorchrun() || s.isMultiNode() {
		return supervise(ctx, s.command(base, goos, s.Stderr), grace)
	}

	var err error
	for attempt := 1; attempt <= maxPortAttempts; attempt++ {
		if s.MasterPort, err = freeLoopbackPort(); err != nil {
			return err
		}
		detector := &conflictDetector{target: s.Stderr, marker: portConflictMarker}
		err = supervise(ctx, s.command(base, goos, detector), grace)
		if err == nil || ctx.Err() != nil || !detector.seen {
			return err
		}
		fmt.Fprintf(s.Stderr, "trainmesh: rendezvous port %d was taken (attempt %d/%d), retrying\n", s.MasterPort, attempt, maxPortAttempts)
	}
	return fmt.Errorf("%w after %d attempts: %w", ErrRendezvousPortUnavailable, maxPortAttempts, err)
}

// ErrRendezvousPortUnavailable means every chosen rendezvous port was taken before bind.
var ErrRendezvousPortUnavailable = errors.New("no free rendezvous port")

func (s Spec) command(base []string, goos string, stderr io.Writer) *exec.Cmd {
	cmd := exec.Command(s.Python, s.Args()...)
	cmd.Stdout = s.Stdout
	cmd.Stderr = stderr
	cmd.Env = s.Env(base, goos)
	cmd.Dir = s.WorkDir
	return cmd
}

func freeLoopbackPort() (int, error) {
	listener, err := net.Listen("tcp", loopbackAddr+":0")
	if err != nil {
		return 0, fmt.Errorf("selecting rendezvous port: %w", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		return 0, fmt.Errorf("releasing probe listener: %w", err)
	}
	return port, nil
}

// conflictDetector forwards output and records whether marker appeared in it.
type conflictDetector struct {
	target io.Writer
	marker []byte
	tail   []byte
	seen   bool
}

func (d *conflictDetector) Write(p []byte) (int, error) {
	if !d.seen {
		// Keep a short tail so a marker split across writes is still found.
		window := append(d.tail, bytes.ToLower(p)...)
		d.seen = bytes.Contains(window, d.marker)
		keep := min(len(window), len(d.marker))
		d.tail = append(d.tail[:0], window[len(window)-keep:]...)
	}
	if d.target == nil {
		return len(p), nil
	}
	return d.target.Write(p)
}

// supervise runs cmd in its own process group so cancellation reaches every rank, not only torchrun.
func supervise(ctx context.Context, cmd *exec.Cmd, grace time.Duration) error {
	configureProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting %s: %w", cmd.Path, err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	// Reap ranks that outlive a crashed torchrun parent.
	defer killProcessGroup(cmd)

	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("training runtime exited: %w", err)
		}
		return nil
	case <-ctx.Done():
		terminateProcessGroup(cmd)
		select {
		case <-done:
		case <-time.After(grace):
			killProcessGroup(cmd)
			<-done
		}
		return fmt.Errorf("training interrupted: %w", context.Cause(ctx))
	}
}

func lookupEnv(env []string, key string) string {
	value := ""
	for _, entry := range env {
		if k, v, ok := strings.Cut(entry, "="); ok && k == key {
			value = v
		}
	}
	return value
}

func listSeparator(goos string) rune {
	if goos == "windows" {
		return ';'
	}
	return ':'
}
