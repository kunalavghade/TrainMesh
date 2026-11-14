//go:build !windows

package launcher

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestSuperviseCancelStopsGrandchildren(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	// The shell stands in for torchrun; its background sleep stands in for a rank.
	cmd := exec.Command("sh", "-c", `sleep 60 & echo $! > "$1"; wait`, "sh", pidFile)
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- supervise(ctx, cmd, 2*time.Second) }()

	childPID := waitForPID(t, pidFile)
	cancel()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("supervise() = %v, want context.Canceled", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("supervise() did not return after cancellation")
	}
	if processAlive(childPID) {
		t.Fatalf("grandchild %d survived cancellation", childPID)
	}
}

func TestSuperviseReportsNonZeroExit(t *testing.T) {
	err := supervise(context.Background(), exec.Command("sh", "-c", "exit 3"), time.Second)
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 3 {
		t.Fatalf("supervise() = %v, want exit status 3", err)
	}
}

func waitForPID(t *testing.T, path string) int {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
				return pid
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("child pid file was not written")
	return 0
}

func processAlive(pid int) bool {
	// Allow the kernel a moment to deliver the signal before probing.
	for range 50 {
		if syscall.Kill(pid, 0) != nil {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
	return true
}
