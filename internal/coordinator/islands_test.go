package coordinator

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/trainmesh/trainmesh/internal/config"
	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/store"
)

const dilocoJobYAML = `version: 1
execution:
  strategy: %s
  workers: %d
  processes_per_worker: 1
  backend: {device: cpu, communication: gloo}
  diloco: {inner_steps: 10, machines_per_island: %d, min_islands: %d}
training:
  dataset: {type: teacher, samples: 4096}
  batch: {per_rank: 64}
  epochs: 1
`

var testPorts = islandPorts{rendezvous: 29500, tmcomm: 29700}

func islandSpec(t *testing.T, strategy string, workers, perIsland, minIslands int) *trainmeshv1.JobSpec {
	t.Helper()
	parsed, err := config.Parse(fmt.Appendf(nil, dilocoJobYAML, strategy, workers, perIsland, minIslands))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return parsed.Spec
}

func islandNodes(count int) []store.Node {
	nodes := make([]store.Node, count)
	for i := range nodes {
		nodes[i] = store.Node{ID: fmt.Sprintf("node-%d", i), Address: fmt.Sprintf("10.0.0.%d", i+1), State: membership.StateReady}
	}
	return nodes
}

func TestDataParallelIsOneIslandThatMustFinish(t *testing.T) {
	islands, required := formIslands(islandSpec(t, "data_parallel", 3, 0, 0), islandNodes(3), testPorts)
	if len(islands) != 1 || required != 1 || len(islands[0].NodeIDs) != 3 {
		t.Fatalf("islands = %+v, required %d", islands, required)
	}
	if islands[0].MasterAddr != "10.0.0.1" || islands[0].MasterPort != 29500 || islands[0].TmcommAddress != "" {
		t.Errorf("island = %+v", islands[0])
	}
}

func TestDilocoIslandsGetTheirOwnMastersAndTmcommPorts(t *testing.T) {
	islands, required := formIslands(islandSpec(t, "diloco", 4, 2, 0), islandNodes(4), testPorts)
	if len(islands) != 2 || required != 2 {
		t.Fatalf("got %d islands, required %d; want 2 and a majority of 2 = 2", len(islands), required)
	}
	second := islands[1]
	if fmt.Sprint(second.NodeIDs) != "[node-2 node-3]" || second.MasterAddr != "10.0.0.3" || second.MasterPort != 29501 {
		t.Errorf("island 1 = %+v", second)
	}
	if second.TmcommAddress != "10.0.0.3:29701" || second.TmcommPort != 29701 {
		t.Errorf("island 1 tmcomm = %q/%d", second.TmcommAddress, second.TmcommPort)
	}

	if _, required := formIslands(islandSpec(t, "diloco", 3, 1, 0), islandNodes(3), testPorts); required != 2 {
		t.Errorf("3 islands: required %d, want a majority (2)", required)
	}
	if _, required := formIslands(islandSpec(t, "diloco", 3, 1, 1), islandNodes(3), testPorts); required != 1 {
		t.Errorf("min_islands 1: required %d", required)
	}
}

func TestIslandRulesAreValidated(t *testing.T) {
	if _, err := config.Parse(fmt.Appendf(nil, dilocoJobYAML, "diloco", 3, 2, 0)); err == nil || !strings.Contains(err.Error(), "multiple of") {
		t.Errorf("3 workers in islands of 2: err = %v", err)
	}
	if _, err := config.Parse(fmt.Appendf(nil, dilocoJobYAML, "diloco", 2, 1, 3)); err == nil || !strings.Contains(err.Error(), "min_islands") {
		t.Errorf("min_islands 3 of 2: err = %v", err)
	}
}

// dilocoJob is a running 3-island job, one node per island, two of which must finish.
func dilocoJob(t *testing.T, perIsland int) store.Job {
	t.Helper()
	nodes := islandNodes(3 * perIsland)
	islands, required := formIslands(islandSpec(t, "diloco", 3*perIsland, perIsland, 0), nodes, testPorts)
	job := store.Job{ID: "job-1", State: store.JobRunning, Strategy: strategyDiloco, Islands: islands, MinIslands: required}
	for position, node := range nodes {
		job.Assignments = append(job.Assignments, store.Assignment{
			NodeID: node.ID, Island: position / perIsland, NodeRank: position % perIsland, State: store.JobRunning,
		})
	}
	return job
}

func report(job *store.Job, nodeID string, state store.JobState, result string) {
	assignment, _ := job.AssignmentFor(nodeID)
	assignment.State, assignment.Message = state, "test"
	if state == store.JobSucceeded && keepsResult(job, *assignment) {
		job.Result, job.ResultIsland = []byte(result), assignment.Island
	}
	settleJob(job, *assignment, time.Unix(0, 0))
}

func TestALostIslandDoesNotStopAQuorum(t *testing.T) {
	job := dilocoJob(t, 1)
	report(&job, "node-2", store.JobFailed, "")
	if job.State != store.JobRunning {
		t.Fatalf("one of three islands failed: job %s, want RUNNING", job.State)
	}
	report(&job, "node-1", store.JobSucceeded, `{"island": 1}`)
	report(&job, "node-0", store.JobSucceeded, `{"island": 0}`)
	if job.State != store.JobSucceeded {
		t.Fatalf("two islands finished: job %s (%s), want SUCCEEDED", job.State, job.Error)
	}
	if string(job.Result) != `{"island": 0}` {
		t.Errorf("result = %s, want the lowest island's whatever the report order", job.Result)
	}
	states := islandStates(&job)
	if states[0] != islandSucceeded || states[2] != islandFailed {
		t.Errorf("island states = %v", states)
	}
}

func TestLosingTheQuorumFailsTheJob(t *testing.T) {
	job := dilocoJob(t, 1)
	report(&job, "node-2", store.JobFailed, "")
	report(&job, "node-0", store.JobCancelled, "")
	if job.State != store.JobFailed || !strings.Contains(job.Error, "2 of 3 islands failed, 2 must finish") {
		t.Errorf("job %s: %q", job.State, job.Error)
	}
}

func TestAFailedIslandCancelsItsOtherMachines(t *testing.T) {
	job := dilocoJob(t, 2) // islands of two machines
	job.Assignments[1].State = store.JobPending
	report(&job, "node-0", store.JobFailed, "") // island 0's first machine
	directives := directivesFor(job, "node-1")
	if len(directives) != 1 || !strings.Contains(directives[0].GetCancel().GetReason(), "island 0 failed") {
		t.Errorf("directives for island 0's other machine = %v, want a cancel", directives)
	}
	if got := directivesFor(job, "node-2"); len(got) != 0 {
		t.Errorf("island 1 must keep running, got %v", got)
	}
}

func TestStartTellsAnIslandWhereEveryLeaderIs(t *testing.T) {
	job := dilocoJob(t, 2)
	job.Machines, job.RanksPerMachine = 6, 1
	job.Assignments[3].State = store.JobPending // island 1, node rank 1
	start := directivesFor(job, "node-3")[0].GetStart()
	if start.GetIsland() != 1 || start.GetNodeRank() != 1 || start.GetMachines() != 2 {
		t.Errorf("start = island %d rank %d machines %d, want 1/1/2", start.GetIsland(), start.GetNodeRank(), start.GetMachines())
	}
	if start.GetMasterAddr() != "10.0.0.3" || start.GetMasterPort() != 29501 || start.GetTmcommPort() != 29701 {
		t.Errorf("start master %s:%d tmcomm %d", start.GetMasterAddr(), start.GetMasterPort(), start.GetTmcommPort())
	}
	peers := start.GetIslandPeers()
	if len(peers) != 3 || peers[2].GetAddress() != "10.0.0.5:29702" || peers[2].GetName() != "node-4" {
		t.Errorf("peers = %v", peers)
	}
}
