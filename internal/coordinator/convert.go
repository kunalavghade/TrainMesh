package coordinator

import (
	"math"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/store"
)

func capabilitiesFromProto(c *trainmeshv1.Capabilities) store.Capabilities {
	return store.Capabilities{
		Hostname:       c.GetHostname(),
		OS:             c.GetOs(),
		Arch:           c.GetArch(),
		LogicalCPUs:    int(c.GetLogicalCpus()),
		MemoryBytes:    c.GetMemoryBytes(),
		RuntimeVersion: c.GetRuntimeVersion(),
		TorchVersion:   c.GetTorchVersion(),
	}
}

func capabilitiesToProto(c store.Capabilities) *trainmeshv1.Capabilities {
	return &trainmeshv1.Capabilities{
		Hostname:       c.Hostname,
		Os:             c.OS,
		Arch:           c.Arch,
		LogicalCpus:    toInt32(c.LogicalCPUs),
		MemoryBytes:    c.MemoryBytes,
		RuntimeVersion: c.RuntimeVersion,
		TorchVersion:   c.TorchVersion,
	}
}

func nodeToProto(n store.Node) *trainmeshv1.Node {
	return &trainmeshv1.Node{
		Id:            n.ID,
		State:         string(n.State),
		Address:       n.Address,
		Interface:     n.Interface,
		Capabilities:  capabilitiesToProto(n.Capabilities),
		JoinedAt:      timestamp(n.JoinedAt),
		LastHeartbeat: timestamp(n.LastHeartbeat),
		CurrentJobId:  n.CurrentJobID,
	}
}

func jobToProto(j store.Job) *trainmeshv1.Job {
	job := &trainmeshv1.Job{
		Id:              j.ID,
		State:           string(j.State),
		Machines:        toInt32(j.Machines),
		RanksPerMachine: toInt32(j.RanksPerMachine),
		MasterAddr:      j.MasterAddr,
		MasterPort:      toInt32(j.MasterPort),
		ResultJson:      j.Result,
		Error:           j.Error,
		CreatedAt:       timestamp(j.CreatedAt),
		FinishedAt:      timestamp(j.FinishedAt),
	}
	for _, a := range j.Assignments {
		job.Assignments = append(job.Assignments, &trainmeshv1.Assignment{
			NodeId: a.NodeID, NodeRank: toInt32(a.NodeRank), State: string(a.State), Message: a.Message,
		})
	}
	return job
}

// toInt32 converts a bounded count (machines, ranks, CPUs, ports) for the wire, saturating
// instead of wrapping should a value ever exceed int32.
func toInt32[T int | int64](v T) int32 {
	switch {
	case v > math.MaxInt32:
		return math.MaxInt32
	case v < math.MinInt32:
		return math.MinInt32
	default:
		return int32(v) // #nosec G115 -- range checked above
	}
}

func millis(d time.Duration) int32 { return toInt32(d.Milliseconds()) }

func timestamp(t time.Time) *timestamppb.Timestamp {
	if t.IsZero() {
		return nil
	}
	return timestamppb.New(t)
}
