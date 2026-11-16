package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/trainmesh/trainmesh/internal/config"
	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/security"
)

// Environment variables that set mesh defaults.
const (
	EnvCoordinator = "TRAINMESH_COORDINATOR"
	EnvAdminDir    = "TRAINMESH_ADMIN_DIR"
)

const (
	defaultCoordinator  = "127.0.0.1:7443"
	defaultPollInterval = 2 * time.Second
	defaultJobTimeout   = 2 * time.Hour
	rpcTimeout          = 15 * time.Second
	cancelTimeout       = 10 * time.Second
	bytesPerGiB         = 1 << 30
)

// ErrJobFailed is returned when a mesh job ends in any state other than SUCCEEDED.
var ErrJobFailed = errors.New("job did not succeed")

func addAdminFlags(cmd *cobra.Command, defaultAddress string) {
	home, _ := os.UserHomeDir()
	cmd.Flags().String("coordinator", envOr(EnvCoordinator, defaultAddress), "meshd host:port")
	cmd.Flags().String("admin-dir", envOr(EnvAdminDir, filepath.Join(home, ".trainmesh", "meshd")), "directory with ca.pem, admin.pem and admin-key.pem (meshd's state dir)")
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

// dialAdmin connects to meshd with the admin certificate.
func dialAdmin(cmd *cobra.Command) (trainmeshv1.CoordinatorServiceClient, func(), error) {
	address, _ := cmd.Flags().GetString("coordinator")
	dir, _ := cmd.Flags().GetString("admin-dir")
	ca, err := security.ReadCert(filepath.Join(dir, security.CAFile))
	if err != nil {
		return nil, nil, fmt.Errorf("admin credentials: %w (is --admin-dir meshd's state dir?)", err)
	}
	cert, key, err := security.ReadCertAndKey(filepath.Join(dir, security.AdminFile), filepath.Join(dir, security.AdminKeyFile))
	if err != nil {
		return nil, nil, fmt.Errorf("admin credentials: %w", err)
	}
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(credentials.NewTLS(security.ClientTLS(ca, cert.Raw, key))))
	if err != nil {
		return nil, nil, fmt.Errorf("connecting to %s: %w", address, err)
	}
	return trainmeshv1.NewCoordinatorServiceClient(conn), func() { _ = conn.Close() }, nil
}

func rpcContext(cmd *cobra.Command) (context.Context, context.CancelFunc) {
	return context.WithTimeout(cmd.Context(), rpcTimeout)
}

func tokenCommand() *cobra.Command {
	root := &cobra.Command{Use: "token", Short: "Manage join tokens"}
	create := &cobra.Command{Use: "create", Short: "Create a join token and print the join string for a new machine", RunE: func(cmd *cobra.Command, _ []string) error {
		ttl, _ := cmd.Flags().GetDuration("ttl")
		uses, _ := cmd.Flags().GetInt32("uses")
		client, closeConn, err := dialAdmin(cmd)
		if err != nil {
			return err
		}
		defer closeConn()
		ctx, cancel := rpcContext(cmd)
		defer cancel()
		response, err := client.CreateToken(ctx, &trainmeshv1.CreateTokenRequest{TtlSeconds: int64(ttl.Seconds()), Uses: uses})
		if err != nil {
			return fmt.Errorf("creating token: %w", err)
		}
		fmt.Fprintf(cmd.ErrOrStderr(), "join string (expires %s, %d use(s)); run on the new machine: tm-agent --join '<join string>'\n",
			response.GetExpiresAt().AsTime().Format(time.RFC3339), uses)
		fmt.Fprintln(cmd.OutOrStdout(), response.GetJoinString())
		return nil
	}}
	create.Flags().Duration("ttl", 15*time.Minute, "how long the token stays valid")
	create.Flags().Int32("uses", 1, "how many machines may join with it")
	addAdminFlags(create, defaultCoordinator)
	root.AddCommand(create)
	return root
}

func nodesCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "nodes", Short: "List machines in the cluster with their state and capabilities", RunE: func(cmd *cobra.Command, _ []string) error {
		client, closeConn, err := dialAdmin(cmd)
		if err != nil {
			return err
		}
		defer closeConn()
		ctx, cancel := rpcContext(cmd)
		defer cancel()
		response, err := client.ListNodes(ctx, &trainmeshv1.ListNodesRequest{})
		if err != nil {
			return fmt.Errorf("listing nodes: %w", err)
		}
		nodes := make([]map[string]any, 0, len(response.GetNodes()))
		for _, n := range response.GetNodes() {
			caps := n.GetCapabilities()
			nodes = append(nodes, map[string]any{
				"id": n.GetId(), "state": n.GetState(), "address": n.GetAddress(), "interface": n.GetInterface(),
				"hostname": caps.GetHostname(), "cpus": caps.GetLogicalCpus(),
				"memory_gib": float64(caps.GetMemoryBytes()) / bytesPerGiB,
				"torch":      caps.GetTorchVersion(), "runtime": caps.GetRuntimeVersion(),
				"last_heartbeat": n.GetLastHeartbeat().AsTime().Format(time.RFC3339), "current_job": n.GetCurrentJobId(),
			})
		}
		return writeJSON(cmd.OutOrStdout(), nodes)
	}}
	addAdminFlags(cmd, defaultCoordinator)
	return cmd
}

func jobCommand() *cobra.Command {
	root := &cobra.Command{Use: "job", Short: "Inspect or cancel mesh jobs"}
	get := &cobra.Command{Use: "get JOB_ID", Short: "Show a job's state, assignments and result", Args: cobra.ExactArgs(1), RunE: func(cmd *cobra.Command, args []string) error {
		client, closeConn, err := dialAdmin(cmd)
		if err != nil {
			return err
		}
		defer closeConn()
		job, err := fetchJob(cmd.Context(), client, args[0])
		if err != nil {
			return err
		}
		return writeJSON(cmd.OutOrStdout(), jobSummary(job))
	}}
	cancelJob := &cobra.Command{Use: "cancel JOB_ID", Short: "Stop a running job on every machine", Args: cobra.ExactArgs(1), RunE: func(cmd *cobra.Command, args []string) error {
		client, closeConn, err := dialAdmin(cmd)
		if err != nil {
			return err
		}
		defer closeConn()
		ctx, cancel := rpcContext(cmd)
		defer cancel()
		if _, err := client.CancelJob(ctx, &trainmeshv1.CancelJobRequest{JobId: args[0]}); err != nil {
			return fmt.Errorf("cancelling %s: %w", args[0], err)
		}
		fmt.Fprintf(cmd.OutOrStdout(), "job %s cancelled\n", args[0])
		return nil
	}}
	addAdminFlags(get, defaultCoordinator)
	addAdminFlags(cancelJob, defaultCoordinator)
	root.AddCommand(get, cancelJob)
	return root
}

// runOnMesh submits the config to meshd and follows the job until it ends. Ctrl-C or the
// timeout cancels the job on every machine.
func runOnMesh(cmd *cobra.Command, configPath string) error {
	raw, err := os.ReadFile(configPath) // #nosec G304 -- operator-supplied config path
	if err != nil {
		return fmt.Errorf("reading config %s: %w", configPath, err)
	}
	if _, err := config.Parse(raw); err != nil {
		return err
	}
	poll, _ := cmd.Flags().GetDuration("poll-interval")
	timeout, _ := cmd.Flags().GetDuration("timeout")
	output, _ := cmd.Flags().GetString("output")
	client, closeConn, err := dialAdmin(cmd)
	if err != nil {
		return err
	}
	defer closeConn()

	ctx, cancel := rpcContext(cmd)
	submitted, err := client.SubmitJob(ctx, &trainmeshv1.SubmitJobRequest{ConfigYaml: raw})
	cancel()
	if err != nil {
		return fmt.Errorf("submitting job: %w", err)
	}
	jobID := submitted.GetJobId()
	progress := cmd.ErrOrStderr()
	fmt.Fprintf(progress, "[%s] submitted\n", jobID)

	job, err := followJob(cmd.Context(), client, jobID, poll, timeout, progress)
	if err != nil {
		cancelCtx, stop := context.WithTimeout(context.Background(), cancelTimeout)
		defer stop()
		if _, cancelErr := client.CancelJob(cancelCtx, &trainmeshv1.CancelJobRequest{JobId: jobID}); cancelErr != nil {
			fmt.Fprintf(progress, "[%s] could not cancel: %v\n", jobID, cancelErr)
		}
		return err
	}
	if job.GetState() != "SUCCEEDED" {
		return fmt.Errorf("%w: %s %s: %s", ErrJobFailed, jobID, job.GetState(), job.GetError())
	}
	return writeResult(cmd.OutOrStdout(), output, job.GetResultJson())
}

func followJob(ctx context.Context, client trainmeshv1.CoordinatorServiceClient, jobID string, poll, timeout time.Duration, progress io.Writer) (*trainmeshv1.Job, error) {
	deadline := time.Now().Add(timeout)
	lastSeen := map[string]string{}
	for {
		job, err := fetchJob(ctx, client, jobID)
		if err != nil {
			return nil, err
		}
		reportChanges(progress, job, lastSeen)
		if isTerminal(job.GetState()) {
			return job, nil
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("job %s still %s after %s", jobID, job.GetState(), timeout)
		}
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("interrupted while following %s: %w", jobID, ctx.Err())
		case <-time.After(poll):
		}
	}
}

func reportChanges(w io.Writer, job *trainmeshv1.Job, lastSeen map[string]string) {
	if lastSeen["job"] != job.GetState() {
		fmt.Fprintf(w, "[%s] %s on %d machine(s), master %s:%d\n", job.GetId(), job.GetState(), job.GetMachines(), job.GetMasterAddr(), job.GetMasterPort())
		lastSeen["job"] = job.GetState()
	}
	for _, a := range job.GetAssignments() {
		if lastSeen[a.GetNodeId()] != a.GetState() {
			fmt.Fprintf(w, "[%s]   %s (node rank %d): %s %s\n", job.GetId(), a.GetNodeId(), a.GetNodeRank(), a.GetState(), a.GetMessage())
			lastSeen[a.GetNodeId()] = a.GetState()
		}
	}
}

func fetchJob(ctx context.Context, client trainmeshv1.CoordinatorServiceClient, jobID string) (*trainmeshv1.Job, error) {
	rpcCtx, cancel := context.WithTimeout(ctx, rpcTimeout)
	defer cancel()
	response, err := client.GetJob(rpcCtx, &trainmeshv1.GetJobRequest{JobId: jobID})
	if err != nil {
		return nil, fmt.Errorf("reading job %s: %w", jobID, err)
	}
	return response.GetJob(), nil
}

func writeResult(stdout io.Writer, path string, result []byte) error {
	var pretty bytes.Buffer
	if err := json.Indent(&pretty, result, "", "  "); err != nil {
		return fmt.Errorf("job result is not JSON: %w", err)
	}
	pretty.WriteByte('\n')
	if path != "" {
		if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
			return err
		}
		if err := os.WriteFile(path, pretty.Bytes(), 0o600); err != nil {
			return fmt.Errorf("writing %s: %w", path, err)
		}
	}
	_, err := stdout.Write(pretty.Bytes())
	return err
}

func jobSummary(job *trainmeshv1.Job) map[string]any {
	assignments := make([]map[string]any, 0, len(job.GetAssignments()))
	for _, a := range job.GetAssignments() {
		assignments = append(assignments, map[string]any{"node": a.GetNodeId(), "node_rank": a.GetNodeRank(), "state": a.GetState(), "message": a.GetMessage()})
	}
	summary := map[string]any{
		"id": job.GetId(), "state": job.GetState(), "machines": job.GetMachines(), "ranks_per_machine": job.GetRanksPerMachine(),
		"master": fmt.Sprintf("%s:%d", job.GetMasterAddr(), job.GetMasterPort()), "assignments": assignments, "error": job.GetError(),
	}
	if len(job.GetResultJson()) > 0 {
		summary["result"] = json.RawMessage(job.GetResultJson())
	}
	return summary
}

func isTerminal(state string) bool {
	return state == "SUCCEEDED" || state == "FAILED" || state == "CANCELLED"
}
