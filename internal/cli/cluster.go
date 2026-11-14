package cli

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
)

const (
	stateDir           = ".trainmesh"
	clusterStateFile   = "cluster.json"
	defaultClusterName = "local-cpu-cluster"
	clusterIDBytes     = 16
)

type clusterState struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Created string `json:"created"`
}

func clusterCommand() *cobra.Command {
	root := &cobra.Command{Use: "cluster", Short: "Manage local TrainMesh cluster metadata"}
	root.AddCommand(clusterInitCommand(), clusterStatusCommand())
	return root
}

func clusterInitCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "init", Short: "Initialize cluster identity on this machine", RunE: func(cmd *cobra.Command, _ []string) error {
		name, _ := cmd.Flags().GetString("name")
		if name == "" {
			return errors.New("--name must not be empty")
		}
		statePath := filepath.Join(stateDir, clusterStateFile)
		if _, err := os.Stat(statePath); err == nil {
			return fmt.Errorf("cluster already initialized (%s exists)", statePath)
		}
		idBytes := make([]byte, clusterIDBytes)
		if _, err := rand.Read(idBytes); err != nil {
			return fmt.Errorf("generating cluster id: %w", err)
		}
		state := clusterState{ID: hex.EncodeToString(idBytes), Name: name, Created: time.Now().UTC().Format(time.RFC3339)}
		if err := writeClusterState(statePath, state); err != nil {
			return err
		}
		fmt.Fprintf(cmd.OutOrStdout(), "Initialized cluster %s (%s)\n", state.Name, state.ID)
		return nil
	}}
	cmd.Flags().String("name", defaultClusterName, "cluster name")
	return cmd
}

func clusterStatusCommand() *cobra.Command {
	return &cobra.Command{Use: "status", Short: "Show local cluster identity and host resources", RunE: func(cmd *cobra.Command, _ []string) error {
		b, err := os.ReadFile(filepath.Join(stateDir, clusterStateFile))
		if errors.Is(err, fs.ErrNotExist) {
			return errors.New("cluster is not initialized; run `trainmesh cluster init`")
		}
		if err != nil {
			return fmt.Errorf("reading cluster state: %w", err)
		}
		var state clusterState
		if err := json.Unmarshal(b, &state); err != nil {
			return fmt.Errorf("parsing cluster state: %w", err)
		}
		h, err := currentHost()
		if err != nil {
			return err
		}
		hostname, err := os.Hostname()
		if err != nil {
			hostname = "unknown"
		}
		return writeJSON(cmd.OutOrStdout(), map[string]any{
			"cluster":            state,
			"local_machine":      map[string]any{"hostname": hostname, "os": h.goos, "logical_cpu_count": h.logicalCPUs},
			"registered_workers": 0,
			"training_runs":      0,
			"note":               "worker enrollment and remote agent orchestration are not implemented yet",
		})
	}}
}

func writeClusterState(path string, state clusterState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("creating %s: %w", filepath.Dir(path), err)
	}
	b, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding cluster state: %w", err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}
