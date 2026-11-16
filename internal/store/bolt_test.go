package store

import (
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/security"
)

func openTestStore(t *testing.T) (*BoltStore, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	s, err := OpenBolt(path)
	if err != nil {
		t.Fatalf("OpenBolt() error = %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, path
}

func TestRecordsRoundTripAcrossReopen(t *testing.T) {
	s, path := openTestStore(t)
	now := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	node := Node{ID: "node-a", State: membership.StateReady, Address: "10.0.0.2", JoinedAt: now}
	job := Job{ID: "job-1", State: JobRunning, Machines: 2, Assignments: []Assignment{{NodeID: "node-a", NodeRank: 0}}}
	token := security.TokenRecord{Hash: "abc", ExpiresAt: now.Add(time.Minute), UsesLeft: 1}

	err := s.Update(func(tx Tx) error {
		return errors.Join(tx.PutNode(node), tx.PutJob(job), tx.PutToken(token))
	})
	if err != nil {
		t.Fatalf("Update() error = %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenBolt(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()

	err = reopened.View(func(tx Tx) error {
		gotNode, err := tx.Node("node-a")
		if err != nil || gotNode.State != membership.StateReady || !gotNode.JoinedAt.Equal(now) {
			t.Errorf("Node() = %+v, %v", gotNode, err)
		}
		gotJob, err := tx.Job("job-1")
		if err != nil || len(gotJob.Assignments) != 1 || gotJob.Machines != 2 {
			t.Errorf("Job() = %+v, %v", gotJob, err)
		}
		gotToken, err := tx.Token("abc")
		if err != nil || gotToken.UsesLeft != 1 {
			t.Errorf("Token() = %+v, %v", gotToken, err)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestMissingRecordsReturnErrNotFound(t *testing.T) {
	s, _ := openTestStore(t)
	_ = s.View(func(tx Tx) error {
		if _, err := tx.Node("nope"); !errors.Is(err, ErrNotFound) {
			t.Errorf("Node() error = %v", err)
		}
		if _, err := tx.Job("nope"); !errors.Is(err, ErrNotFound) {
			t.Errorf("Job() error = %v", err)
		}
		if _, err := tx.Token("nope"); !errors.Is(err, ErrNotFound) {
			t.Errorf("Token() error = %v", err)
		}
		return nil
	})
}

func TestFailedUpdateRollsBack(t *testing.T) {
	s, _ := openTestStore(t)
	sentinel := errors.New("abort")

	err := s.Update(func(tx Tx) error {
		if err := tx.PutNode(Node{ID: "node-a"}); err != nil {
			return err
		}
		return sentinel
	})

	if !errors.Is(err, sentinel) {
		t.Fatalf("Update() = %v, want sentinel", err)
	}
	_ = s.View(func(tx Tx) error {
		if _, err := tx.Node("node-a"); !errors.Is(err, ErrNotFound) {
			t.Errorf("write survived a failed transaction: %v", err)
		}
		return nil
	})
}

func TestListsAreSortedByID(t *testing.T) {
	s, _ := openTestStore(t)
	_ = s.Update(func(tx Tx) error {
		for _, id := range []string{"node-c", "node-a", "node-b"} {
			if err := tx.PutNode(Node{ID: id}); err != nil {
				return err
			}
		}
		return nil
	})
	_ = s.View(func(tx Tx) error {
		nodes, err := tx.Nodes()
		if err != nil || len(nodes) != 3 || nodes[0].ID != "node-a" || nodes[2].ID != "node-c" {
			t.Errorf("Nodes() = %+v, %v", nodes, err)
		}
		return nil
	})
}

func TestSecondOpenFailsFastWhileLocked(t *testing.T) {
	_, path := openTestStore(t)
	start := time.Now()

	_, err := OpenBolt(path)

	if err == nil {
		t.Fatal("second OpenBolt on a locked file succeeded")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("second open took %s, want a fast failure", time.Since(start))
	}
}

func TestJobHelpers(t *testing.T) {
	job := Job{Assignments: []Assignment{{NodeID: "a", NodeRank: 0}, {NodeID: "b", NodeRank: 1}}}
	if a, ok := job.AssignmentFor("b"); !ok || a.NodeRank != 1 {
		t.Errorf("AssignmentFor(b) = %+v, %v", a, ok)
	}
	if _, ok := job.AssignmentFor("z"); ok {
		t.Error("AssignmentFor(z) found an assignment")
	}
	for state, terminal := range map[JobState]bool{JobPending: false, JobRunning: false, JobSucceeded: true, JobFailed: true, JobCancelled: true} {
		if state.Terminal() != terminal {
			t.Errorf("%s.Terminal() = %v", state, state.Terminal())
		}
	}
}
