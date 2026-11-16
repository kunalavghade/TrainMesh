package coordinator

import (
	"context"
	"io"
	"log/slog"
	"path/filepath"
	"testing"
	"time"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/security"
	"github.com/trainmesh/trainmesh/internal/store"
)

const twoMachineJob = `version: 1
execution:
  strategy: data_parallel
  workers: 2
  processes_per_worker: 1
  backend: {device: cpu, communication: gloo}
training:
  dataset: {type: teacher, samples: 64}
  batch: {per_rank: 8}
  epochs: 1
`

// newTestServer returns a server over a temp store with two READY nodes, a and b.
func newTestServer(t *testing.T) (*Server, store.Store) {
	t.Helper()
	st, err := store.OpenBolt(filepath.Join(t.TempDir(), StateFile))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = st.Close() })
	ca, err := security.NewCA(time.Now(), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewServer(st, ca, Config{AdvertiseAddress: "127.0.0.1:7443"}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	err = st.Update(func(tx store.Tx) error {
		for _, id := range []string{"node-a", "node-b"} {
			node := store.Node{
				ID: id, State: membership.StateReady, Address: "10.0.0." + id[len(id)-1:], LastHeartbeat: time.Now(),
				Capabilities: store.Capabilities{LogicalCPUs: 4, TorchVersion: "2.15.0", RuntimeVersion: "0.3.0"},
			}
			if err := tx.PutNode(node); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return server, st
}

func asNode(id string) context.Context {
	return context.WithValue(context.Background(), identityKey{}, security.Identity{Name: id, Role: security.RoleNode})
}

func TestFailureBeforeAPeerStartedReleasesThatPeer(t *testing.T) {
	server, st := newTestServer(t)
	submitted, err := server.SubmitJob(context.Background(), &trainmeshv1.SubmitJobRequest{ConfigYaml: []byte(twoMachineJob)})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}

	// node-b fails before node-a ever heartbeats and receives its start directive.
	_, err = server.ReportAssignment(asNode("node-b"), &trainmeshv1.ReportAssignmentRequest{JobId: submitted.GetJobId(), State: "FAILED", Message: "boom"})
	if err != nil {
		t.Fatalf("ReportAssignment: %v", err)
	}

	_ = st.View(func(tx store.Tx) error {
		job, _ := tx.Job(submitted.GetJobId())
		pending, _ := job.AssignmentFor("node-a")
		if job.State != store.JobFailed || pending.State != store.JobCancelled {
			t.Errorf("job %s, node-a assignment %s; want FAILED and CANCELLED", job.State, pending.State)
		}
		for _, id := range []string{"node-a", "node-b"} {
			node, _ := tx.Node(id)
			if node.State != membership.StateReady || node.CurrentJobID != "" {
				t.Errorf("%s = %s (job %q), want READY and free", id, node.State, node.CurrentJobID)
			}
		}
		return nil
	})

	heartbeat, err := server.Heartbeat(asNode("node-a"), &trainmeshv1.HeartbeatRequest{})
	if err != nil || len(heartbeat.GetDirectives()) != 0 {
		t.Fatalf("node-a heartbeat directives = %v, %v; want none", heartbeat.GetDirectives(), err)
	}
}

func TestStartDirectiveRepeatsUntilRunningIsReported(t *testing.T) {
	server, _ := newTestServer(t)
	submitted, err := server.SubmitJob(context.Background(), &trainmeshv1.SubmitJobRequest{ConfigYaml: []byte(twoMachineJob)})
	if err != nil {
		t.Fatal(err)
	}

	for range 2 {
		heartbeat, err := server.Heartbeat(asNode("node-a"), &trainmeshv1.HeartbeatRequest{})
		if err != nil || heartbeat.GetDirectives()[0].GetStart().GetNodeRank() != 0 {
			t.Fatalf("heartbeat = %v, %v; want a start directive for rank 0", heartbeat, err)
		}
	}
	if _, err := server.ReportAssignment(asNode("node-a"), &trainmeshv1.ReportAssignmentRequest{JobId: submitted.GetJobId(), State: "RUNNING"}); err != nil {
		t.Fatal(err)
	}
	heartbeat, _ := server.Heartbeat(asNode("node-a"), &trainmeshv1.HeartbeatRequest{})
	if len(heartbeat.GetDirectives()) != 0 {
		t.Fatalf("directives after RUNNING = %v, want none", heartbeat.GetDirectives())
	}
}

func TestPickNodesIsDeterministicAndChecksVersions(t *testing.T) {
	nodes := []store.Node{
		{ID: "b", State: membership.StateReady, Address: "10.0.0.2", Capabilities: store.Capabilities{TorchVersion: "2.15.0"}},
		{ID: "a", State: membership.StateReady, Address: "10.0.0.1", Capabilities: store.Capabilities{TorchVersion: "2.15.0"}},
		{ID: "c", State: membership.StateBusy, Address: "10.0.0.3"},
	}
	chosen, err := pickNodes(nodes, 2)
	if err != nil || chosen[0].ID != "a" || chosen[1].ID != "b" {
		t.Fatalf("pickNodes = %+v, %v; want a then b", chosen, err)
	}
	nodes[0].Capabilities.TorchVersion = "2.14.0"
	if _, err := pickNodes(nodes, 2); err == nil {
		t.Fatal("mixed PyTorch versions accepted")
	}
}
