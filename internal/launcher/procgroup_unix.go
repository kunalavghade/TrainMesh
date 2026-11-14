//go:build !windows

package launcher

import (
	"os/exec"
	"syscall"
)

func configureProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func terminateProcessGroup(cmd *exec.Cmd) { signalProcessGroup(cmd, syscall.SIGTERM) }

func killProcessGroup(cmd *exec.Cmd) { signalProcessGroup(cmd, syscall.SIGKILL) }

func signalProcessGroup(cmd *exec.Cmd, sig syscall.Signal) {
	if cmd.Process == nil {
		return
	}
	// ESRCH means the group is already gone, which is the goal.
	_ = syscall.Kill(-cmd.Process.Pid, sig)
}
