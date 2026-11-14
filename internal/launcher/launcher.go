// Package launcher starts the Python training runtime on this host as a supervised process group.
package launcher

import (
	"context"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	runtimeModule = "trainmesh_runtime"
	// DefaultGracePeriod is how long ranks get to exit after SIGTERM before SIGKILL.
	DefaultGracePeriod = 10 * time.Second
)

// loopbackInterfaceByOS pins Gloo to loopback for single-host jobs where auto-detection misbehaves.
var loopbackInterfaceByOS = map[string]string{"darwin": "lo0"}

// Spec fully describes one local training launch.
type Spec struct {
	Python         string   // interpreter path
	RuntimeDir     string   // directory that contains the trainmesh_runtime package
	ConfigPath     string   // absolute path to the job config
	LocalRanks     int      // processes on this host
	ThreadsPerRank int      // intra-op CPU threads per rank
	ExtraArgs      []string // forwarded to the runtime, e.g. --resume, --output
	Stdout         io.Writer
	Stderr         io.Writer
	GracePeriod    time.Duration
}

// Args builds the interpreter arguments; torchrun --standalone binds its own free port.
func (s Spec) Args() []string {
	runtimeArgs := append([]string{"-m", runtimeModule, "--config", s.ConfigPath}, s.ExtraArgs...)
	if s.LocalRanks <= 1 {
		return runtimeArgs
	}
	torchrun := []string{
		"-m", "torch.distributed.run",
		"--standalone",
		"--nnodes=1",
		"--nproc-per-node=" + strconv.Itoa(s.LocalRanks),
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
	if iface, ok := loopbackInterfaceByOS[goos]; ok && lookupEnv(base, "GLOO_SOCKET_IFNAME") == "" {
		env = append(env, "GLOO_SOCKET_IFNAME="+iface)
	}
	return env
}

// ResolveThreadsPerRank picks threads: explicit config, then OMP_NUM_THREADS, then an even split.
func ResolveThreadsPerRank(configured int, ompEnv string, logicalCPUs, localRanks int) int {
	if configured > 0 {
		return configured
	}
	if n, err := strconv.Atoi(strings.TrimSpace(ompEnv)); err == nil && n > 0 {
		return n
	}
	return max(1, logicalCPUs/max(1, localRanks))
}

// Run starts the runtime and supervises it until it exits or ctx is cancelled.
func Run(ctx context.Context, s Spec, base []string, goos string) error {
	cmd := exec.Command(s.Python, s.Args()...)
	cmd.Stdout = s.Stdout
	cmd.Stderr = s.Stderr
	cmd.Env = s.Env(base, goos)
	grace := s.GracePeriod
	if grace <= 0 {
		grace = DefaultGracePeriod
	}
	return supervise(ctx, cmd, grace)
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
