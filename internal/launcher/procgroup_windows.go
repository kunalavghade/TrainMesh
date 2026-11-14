//go:build windows

package launcher

import "os/exec"

// Windows has no POSIX process groups; killing the parent is the best available fallback.
func configureProcessGroup(*exec.Cmd) {}

func terminateProcessGroup(cmd *exec.Cmd) { killProcessGroup(cmd) }

func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
