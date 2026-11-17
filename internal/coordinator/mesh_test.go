package coordinator_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/trainmesh/trainmesh/internal/agent"
	"github.com/trainmesh/trainmesh/internal/coordinator"
	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/security"
	"github.com/trainmesh/trainmesh/internal/store"
)

const (
	waitTimeout = 10 * time.Second
	jobYAML     = `version: 1
execution:
  strategy: data_parallel
  workers: %d
  processes_per_worker: 1
  backend: {device: cpu, communication: gloo}
training:
  dataset: {type: teacher, samples: 64}
  batch: {per_rank: 8}
  epochs: 1
`
)

var quietLogger = testLogger()

func testLogger() *slog.Logger {
	if os.Getenv("TRAINMESH_TEST_LOGS") != "" {
		return slog.New(slog.NewTextHandler(os.Stderr, nil))
	}
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// fakeRunner records assignments and runs a per-rank behaviour instead of torchrun.
type fakeRunner struct {
	mu       sync.Mutex
	seen     []agent.Assignment
	behavior func(ctx context.Context, a agent.Assignment) ([]byte, error)
}

func (r *fakeRunner) Run(ctx context.Context, a agent.Assignment) ([]byte, error) {
	r.mu.Lock()
	r.seen = append(r.seen, a)
	r.mu.Unlock()
	return r.behavior(ctx, a)
}

func (r *fakeRunner) assignments() []agent.Assignment {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]agent.Assignment(nil), r.seen...)
}

type mesh struct {
	address string
	dir     string
	admin   trainmeshv1.CoordinatorServiceClient
}

func startCoordinator(t *testing.T) *mesh {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	dir := t.TempDir()
	ca, serverTLS, err := coordinator.PrepareStateDir(dir, address, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	st, err := store.OpenBolt(filepath.Join(dir, coordinator.StateFile))
	if err != nil {
		t.Fatal(err)
	}
	server, err := coordinator.NewServer(st, ca, coordinator.Config{
		AdvertiseAddress:  address,
		HeartbeatInterval: 50 * time.Millisecond,
		Liveness:          membership.LivenessPolicy{SuspectAfter: 250 * time.Millisecond, LostAfter: 600 * time.Millisecond},
		SweepInterval:     50 * time.Millisecond,
	}, quietLogger)
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := coordinator.NewGRPCServer(server, credentials.NewTLS(serverTLS))
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = grpcServer.Serve(listener) }()
	go server.RunSweeper(ctx)
	t.Cleanup(func() {
		cancel()
		grpcServer.Stop()
		_ = st.Close()
	})
	return &mesh{address: address, dir: dir, admin: dialAs(t, address, dir, security.AdminFile, security.AdminKeyFile)}
}

func dialAs(t *testing.T, address, dir, certFile, keyFile string) trainmeshv1.CoordinatorServiceClient {
	t.Helper()
	ca, err := security.ReadCert(filepath.Join(dir, security.CAFile))
	if err != nil {
		t.Fatal(err)
	}
	cert, key, err := security.ReadCertAndKey(filepath.Join(dir, certFile), filepath.Join(dir, keyFile))
	if err != nil {
		t.Fatal(err)
	}
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(credentials.NewTLS(security.ClientTLS(ca, cert.Raw, key))))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return trainmeshv1.NewCoordinatorServiceClient(conn)
}

func (m *mesh) joinString(t *testing.T) string {
	t.Helper()
	response, err := m.admin.CreateToken(context.Background(), &trainmeshv1.CreateTokenRequest{TtlSeconds: 60, Uses: 1})
	if err != nil {
		t.Fatalf("CreateToken: %v", err)
	}
	return response.GetJoinString()
}

type runningAgent struct {
	agent    *agent.Agent
	stateDir string
	stop     context.CancelFunc
}

func startAgent(t *testing.T, join string, runner agent.Runner) (*runningAgent, error) {
	t.Helper()
	stateDir := t.TempDir()
	a, err := agent.New(agent.Config{
		StateDir:   stateDir,
		JoinString: join,
		Address:    "127.0.0.1",
		Interface:  "lo",
		Capabilities: func(context.Context) (*trainmeshv1.Capabilities, error) {
			return &trainmeshv1.Capabilities{Hostname: "test", LogicalCpus: 4, TorchVersion: "2.15.0", RuntimeVersion: "0.3.0"}, nil
		},
		Runner:        runner,
		ReportBackoff: 20 * time.Millisecond,
		ShutdownGrace: 100 * time.Millisecond,
		Logger:        quietLogger,
	})
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	if err := a.Start(ctx); err != nil {
		cancel()
		return nil, err
	}
	done := make(chan struct{})
	go func() { _ = a.Run(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done })
	return &runningAgent{agent: a, stateDir: stateDir, stop: cancel}, nil
}

func waitFor(t *testing.T, what string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(waitTimeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func (m *mesh) nodeStates(t *testing.T) map[string]string {
	t.Helper()
	response, err := m.admin.ListNodes(context.Background(), &trainmeshv1.ListNodesRequest{})
	if err != nil {
		t.Fatalf("ListNodes: %v", err)
	}
	states := map[string]string{}
	for _, node := range response.GetNodes() {
		states[node.GetId()] = node.GetState()
	}
	return states
}

func (m *mesh) allNodes(t *testing.T, want string, count int) func() bool {
	return func() bool {
		states := m.nodeStates(t)
		matching := 0
		for _, state := range states {
			if state == want {
				matching++
			}
		}
		return len(states) == count && matching == count
	}
}

func (m *mesh) job(t *testing.T, id string) *trainmeshv1.Job {
	t.Helper()
	response, err := m.admin.GetJob(context.Background(), &trainmeshv1.GetJobRequest{JobId: id})
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	return response.GetJob()
}

func (m *mesh) submit(t *testing.T, machines int) (string, error) {
	t.Helper()
	response, err := m.admin.SubmitJob(context.Background(), &trainmeshv1.SubmitJobRequest{ConfigYaml: []byte(fmt.Sprintf(jobYAML, machines))})
	return response.GetJobId(), err
}

func succeed(_ context.Context, a agent.Assignment) ([]byte, error) {
	if a.NodeRank == 0 {
		return []byte(`{"world_size": 2, "validation": {"final_accuracy": 0.9}}`), nil
	}
	return nil, nil
}

func blockUntilCancelled(ctx context.Context, _ agent.Assignment) ([]byte, error) {
	<-ctx.Done()
	return nil, context.Cause(ctx)
}

func TestTwoNodesJoinRunAJobAndReturnTheResult(t *testing.T) {
	m := startCoordinator(t)
	runners := []*fakeRunner{{behavior: succeed}, {behavior: succeed}}
	for _, runner := range runners {
		if _, err := startAgent(t, m.joinString(t), runner); err != nil {
			t.Fatalf("agent start: %v", err)
		}
	}
	waitFor(t, "both nodes READY", m.allNodes(t, "READY", 2))

	jobID, err := m.submit(t, 2)
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitFor(t, "job to finish", func() bool { return m.job(t, jobID).GetState() == "SUCCEEDED" })

	var result struct {
		Validation struct {
			FinalAccuracy float64 `json:"final_accuracy"`
		} `json:"validation"`
	}
	if err := json.Unmarshal(m.job(t, jobID).GetResultJson(), &result); err != nil || result.Validation.FinalAccuracy != 0.9 {
		t.Errorf("result = %s (%v), want rank 0's result", m.job(t, jobID).GetResultJson(), err)
	}
	var ranks []int
	for _, runner := range runners {
		for _, a := range runner.assignments() {
			ranks = append(ranks, a.NodeRank)
			if a.Machines != 2 || a.MasterAddr != "127.0.0.1" || a.MasterPort != coordinator.DefaultRendezvousPort {
				t.Errorf("assignment = %+v", a)
			}
		}
	}
	sort.Ints(ranks)
	if fmt.Sprint(ranks) != "[0 1]" {
		t.Errorf("node ranks = %v, want [0 1] once each", ranks)
	}
	waitFor(t, "nodes back to READY", m.allNodes(t, "READY", 2))
}

func TestJoinTokenIsSingleUse(t *testing.T) {
	m := startCoordinator(t)
	join := m.joinString(t)
	if _, err := startAgent(t, join, &fakeRunner{behavior: succeed}); err != nil {
		t.Fatalf("first join: %v", err)
	}

	_, err := startAgent(t, join, &fakeRunner{behavior: succeed})

	if status.Code(errors.Unwrap(err)) != codes.Unauthenticated && !strings.Contains(fmt.Sprint(err), "already been used") {
		t.Fatalf("second join with the same token = %v, want token already used", err)
	}
}

func TestJoinRefusesACoordinatorWithTheWrongCA(t *testing.T) {
	m := startCoordinator(t)
	parsed, err := security.ParseJoinString(m.joinString(t))
	if err != nil {
		t.Fatal(err)
	}
	parsed.CAHash = security.CertificateHash([]byte("some other CA"))

	_, err = startAgent(t, parsed.String(), &fakeRunner{behavior: succeed})

	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("join with a wrong CA hash = %v, want CA mismatch", err)
	}
}

func TestNodeCertificateCannotCallAdminMethods(t *testing.T) {
	m := startCoordinator(t)
	node, err := startAgent(t, m.joinString(t), &fakeRunner{behavior: succeed})
	if err != nil {
		t.Fatal(err)
	}
	nodeClient := dialAs(t, m.address, node.stateDir, "node.pem", "node-key.pem")

	_, err = nodeClient.ListNodes(context.Background(), &trainmeshv1.ListNodesRequest{})

	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("ListNodes with a node certificate = %v, want PermissionDenied", err)
	}
}

func TestOneNodeFailingCancelsTheOthers(t *testing.T) {
	m := startCoordinator(t)
	failRankOne := func(ctx context.Context, a agent.Assignment) ([]byte, error) {
		if a.NodeRank == 1 {
			return nil, errors.New("rank 1 crashed")
		}
		return blockUntilCancelled(ctx, a)
	}
	for range 2 {
		if _, err := startAgent(t, m.joinString(t), &fakeRunner{behavior: failRankOne}); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, "both nodes READY", m.allNodes(t, "READY", 2))

	jobID, err := m.submit(t, 2)
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "job FAILED", func() bool { return m.job(t, jobID).GetState() == "FAILED" })

	job := m.job(t, jobID)
	if !strings.Contains(job.GetError(), "rank 1") || !strings.Contains(job.GetError(), "crashed") {
		t.Errorf("job error = %q, want rank 1's failure", job.GetError())
	}
	waitFor(t, "rank 0 cancelled and every node READY again", m.allNodes(t, "READY", 2))
	for _, a := range m.job(t, jobID).GetAssignments() {
		if a.GetNodeRank() == 0 && a.GetState() != "CANCELLED" {
			t.Errorf("rank 0 assignment = %s, want CANCELLED", a.GetState())
		}
	}
}

func TestSilentNodeIsLostAndFailsItsJob(t *testing.T) {
	m := startCoordinator(t)
	hang := make(chan struct{})
	t.Cleanup(func() { close(hang) })
	hangRankOne := func(ctx context.Context, a agent.Assignment) ([]byte, error) {
		if a.NodeRank == 1 {
			<-hang // ignores cancellation, like a machine that froze
			return nil, nil
		}
		return blockUntilCancelled(ctx, a)
	}
	var agents []*runningAgent
	for range 2 {
		ra, err := startAgent(t, m.joinString(t), &fakeRunner{behavior: hangRankOne})
		if err != nil {
			t.Fatal(err)
		}
		agents = append(agents, ra)
	}
	waitFor(t, "both nodes READY", m.allNodes(t, "READY", 2))
	jobID, err := m.submit(t, 2)
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, "job RUNNING on both", func() bool {
		job := m.job(t, jobID)
		running := 0
		for _, a := range job.GetAssignments() {
			if a.GetState() == "RUNNING" {
				running++
			}
		}
		return running == 2
	})
	var silent string
	for _, a := range m.job(t, jobID).GetAssignments() {
		if a.GetNodeRank() == 1 {
			silent = a.GetNodeId()
		}
	}
	for _, ra := range agents {
		if ra.agent.NodeID() == silent {
			ra.stop() // heartbeats stop; the hung runner never reports
		}
	}

	waitFor(t, "silent node LOST", func() bool { return m.nodeStates(t)[silent] == "LOST" })
	waitFor(t, "job FAILED", func() bool { return m.job(t, jobID).GetState() == "FAILED" })
	if !strings.Contains(m.job(t, jobID).GetError(), "node lost") {
		t.Errorf("job error = %q, want node lost", m.job(t, jobID).GetError())
	}
	waitFor(t, "surviving node READY", func() bool {
		for id, state := range m.nodeStates(t) {
			if id != silent && state == "READY" {
				return true
			}
		}
		return false
	})
}

func TestSubmitNeedsEnoughReadyNodes(t *testing.T) {
	m := startCoordinator(t)
	if _, err := startAgent(t, m.joinString(t), &fakeRunner{behavior: succeed}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "node READY", m.allNodes(t, "READY", 1))

	_, err := m.submit(t, 2)

	if status.Code(err) != codes.FailedPrecondition || !strings.Contains(err.Error(), "needs 2, 1 available") {
		t.Fatalf("SubmitJob = %v, want FailedPrecondition", err)
	}
}

const dilocoMeshYAML = `version: 1
execution:
  strategy: diloco
  workers: 3
  processes_per_worker: 1
  backend: {device: cpu, communication: gloo}
  diloco: {inner_steps: 10}
training:
  dataset: {type: teacher, samples: 64}
  batch: {per_rank: 8}
  epochs: 1
`

func TestDilocoJobSurvivesALostIsland(t *testing.T) {
	m := startCoordinator(t)
	islandTwoFails := func(_ context.Context, a agent.Assignment) ([]byte, error) {
		if a.Island == 2 {
			return nil, errors.New("island machine crashed")
		}
		return []byte(fmt.Sprintf(`{"island": %d}`, a.Island)), nil
	}
	runners := []*fakeRunner{{behavior: islandTwoFails}, {behavior: islandTwoFails}, {behavior: islandTwoFails}}
	for _, runner := range runners {
		if _, err := startAgent(t, m.joinString(t), runner); err != nil {
			t.Fatalf("agent start: %v", err)
		}
	}
	waitFor(t, "three nodes READY", m.allNodes(t, "READY", 3))

	response, err := m.admin.SubmitJob(context.Background(), &trainmeshv1.SubmitJobRequest{ConfigYaml: []byte(dilocoMeshYAML)})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	jobID := response.GetJobId()
	waitFor(t, "job to finish", func() bool { return m.job(t, jobID).GetState() != "RUNNING" })

	job := m.job(t, jobID)
	var result struct{ Island int }
	if job.GetState() != "SUCCEEDED" || json.Unmarshal(job.GetResultJson(), &result) != nil || result.Island != 0 {
		t.Fatalf("job %s (%s), result %s; want SUCCEEDED with island 0's result", job.GetState(), job.GetError(), job.GetResultJson())
	}
	var states []string
	for _, island := range job.GetIslands() {
		states = append(states, island.GetState())
	}
	if fmt.Sprint(states) != "[SUCCEEDED SUCCEEDED FAILED]" || job.GetMinIslands() != 2 {
		t.Errorf("island states %v, min %d", states, job.GetMinIslands())
	}
	for _, runner := range runners {
		for _, a := range runner.assignments() {
			if a.Machines != 1 || a.NodeRank != 0 || len(a.IslandPeers) != 3 || a.TmcommPort != coordinator.DefaultTmcommPort+a.Island {
				t.Errorf("island %d assignment = %+v", a.Island, a)
			}
		}
	}
	waitFor(t, "nodes back to READY", m.allNodes(t, "READY", 3))
}
