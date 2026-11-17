// Package cli implements the trainmesh command tree; cmd/trainmesh only wires it to the process.
package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"
)

// Version is the CLI release version.
const Version = "0.5.0"

// host captures the process environment so commands stay testable.
type host struct {
	getenv      func(string) string
	environ     []string
	cwd         string
	execDir     string
	goos        string
	logicalCPUs int
}

func currentHost() (host, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return host{}, fmt.Errorf("reading working directory: %w", err)
	}
	execDir := ""
	if executable, err := os.Executable(); err == nil {
		execDir = filepath.Dir(executable)
	}
	return host{
		getenv:      os.Getenv,
		environ:     os.Environ(),
		cwd:         cwd,
		execDir:     execDir,
		goos:        runtime.GOOS,
		logicalCPUs: runtime.NumCPU(),
	}, nil
}

// NewRoot builds the trainmesh root command.
func NewRoot() *cobra.Command {
	root := &cobra.Command{Use: "trainmesh", Short: "CPU-first distributed AI training engine", SilenceUsage: true}
	root.AddCommand(
		&cobra.Command{Use: "version", Short: "Print version", Run: func(cmd *cobra.Command, _ []string) {
			fmt.Fprintln(cmd.OutOrStdout(), Version)
		}},
		doctorCommand(), planCommand(), runCommand(), benchmarkCommand(), clusterCommand(),
		tokenCommand(), nodesCommand(), jobCommand(),
	)
	return root
}

func writeJSON(w io.Writer, value any) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return fmt.Errorf("encoding JSON output: %w", err)
	}
	return nil
}
