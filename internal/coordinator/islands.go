package coordinator

import (
	"fmt"
	"net"
	"strconv"
	"time"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/store"
)

const strategyDiloco = "diloco"

// islandPorts are the base ports of island 0; island k uses base + k, so several islands can
// share one host (as in the local e2e tests) without colliding.
type islandPorts struct {
	rendezvous int
	tmcomm     int
}

// formIslands splits the chosen nodes (already in rank order) into islands. data_parallel jobs
// are one island that must finish; diloco jobs get workers / machines_per_island islands and
// need diloco.min_islands of them (default: a majority) to finish.
func formIslands(spec *trainmeshv1.JobSpec, nodes []store.Node, ports islandPorts) ([]store.Island, int) {
	perIsland := len(nodes)
	diloco := spec.GetExecution().GetStrategy() == strategyDiloco
	if diloco {
		perIsland = max(1, int(spec.GetExecution().GetDiloco().GetMachinesPerIsland()))
	}
	islands := make([]store.Island, 0, len(nodes)/perIsland)
	for index := 0; index*perIsland < len(nodes); index++ {
		members := nodes[index*perIsland : (index+1)*perIsland]
		island := store.Island{
			Index:      index,
			MasterAddr: members[0].Address,
			MasterPort: ports.rendezvous + index,
		}
		for _, node := range members {
			island.NodeIDs = append(island.NodeIDs, node.ID)
		}
		if diloco {
			island.TmcommPort = ports.tmcomm + index
			island.TmcommAddress = net.JoinHostPort(members[0].Address, strconv.Itoa(island.TmcommPort))
		}
		islands = append(islands, island)
	}
	required := len(islands)
	if diloco {
		required = int(spec.GetExecution().GetDiloco().GetMinIslands())
		if required == 0 {
			required = len(islands)/2 + 1
		}
	}
	return islands, required
}

type islandState int

const (
	islandRunning islandState = iota
	islandSucceeded
	islandFailed
)

func (s islandState) String() string {
	return [...]string{"RUNNING", "SUCCEEDED", "FAILED"}[s]
}

// islandStates: an island fails as soon as one of its nodes fails or is cancelled (its torchrun
// world cannot continue without it) and succeeds when all of its nodes succeed.
func islandStates(job *store.Job) []islandState {
	states := make([]islandState, job.IslandCount())
	done := make([]bool, len(states))
	for i := range done {
		done[i] = true
	}
	for _, a := range job.Assignments {
		switch a.State {
		case store.JobFailed, store.JobCancelled:
			states[a.Island] = islandFailed
		case store.JobSucceeded:
		default:
			done[a.Island] = false
		}
	}
	for i := range states {
		if states[i] != islandFailed && done[i] {
			states[i] = islandSucceeded
		}
	}
	return states
}

// settleJob moves the job forward after one assignment changed: it fails once fewer islands
// than required can still finish, and succeeds once every island has finished.
func settleJob(job *store.Job, changed store.Assignment, now time.Time) {
	if job.State.Terminal() {
		return
	}
	failed, running := 0, 0
	for _, state := range islandStates(job) {
		switch state {
		case islandFailed:
			failed++
		case islandRunning:
			running++
		}
	}
	total, required := job.IslandCount(), job.RequiredIslands()
	if total-failed < required {
		job.State, job.FinishedAt = store.JobFailed, now
		job.Error = fmt.Sprintf("node %s (rank %d) %s: %s", changed.NodeID, changed.NodeRank, changed.State, changed.Message)
		if total > 1 {
			job.Error += fmt.Sprintf(" (%d of %d islands failed, %d must finish)", failed, total, required)
		}
		return
	}
	if running == 0 {
		job.State, job.FinishedAt = store.JobSucceeded, now
	}
}

// islandFailed reports whether the assignment's island has already failed.
func islandHasFailed(job *store.Job, island int) bool {
	return islandStates(job)[island] == islandFailed
}

// keepsResult reports whether a succeeding island leader's result should become the job's: the
// lowest island that succeeds wins, so the result does not depend on report order.
func keepsResult(job *store.Job, a store.Assignment) bool {
	return a.NodeRank == 0 && (job.Result == nil || a.Island < job.ResultIsland)
}

// islandPeers lists every island leader's tmcomm endpoint (empty for data_parallel).
func islandPeers(job *store.Job) []*trainmeshv1.IslandPeer {
	if job.Strategy != strategyDiloco {
		return nil
	}
	peers := make([]*trainmeshv1.IslandPeer, 0, len(job.Islands))
	for _, island := range job.Islands {
		peers = append(peers, &trainmeshv1.IslandPeer{
			Island:  toInt32(island.Index),
			Address: island.TmcommAddress,
			Name:    island.NodeIDs[0],
		})
	}
	return peers
}
