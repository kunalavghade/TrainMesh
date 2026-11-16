package coordinator

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/store"
)

// Collective islands: nodes in one island can share a process group backend.
const (
	islandHost = "gloo-host" // cpu and mps ranks reduce through host memory with Gloo
	islandCUDA = "nccl-cuda" // all-CUDA groups can use NCCL
)

// Values of execution.communication and training.batch.balance the planner acts on.
const (
	commAuto       = "auto"
	commGloo       = "gloo"
	commNCCL       = "nccl"
	balanceThrough = "throughput"
	deviceAuto     = "auto"
	deviceCUDA     = "cuda"
	minRankBatch   = 1
)

// ErrNoThroughput means balance=throughput was asked for but a chosen node has no profile.
var ErrNoThroughput = errors.New("throughput balancing needs a measured profile on every node")

// plan is the coordinator's decision for one job.
type plan struct {
	nodes          []store.Node
	communication  string
	rankBatchSizes []int // empty: every rank uses per_rank
	configYAML     []byte
}

// island is the collective family of a node's own device.
func island(caps store.Capabilities) string {
	if caps.Device == deviceCUDA {
		return islandCUDA
	}
	return islandHost
}

// planJob chooses nodes that offer the requested device, one collective backend for the whole
// group, and (for balance=throughput) per-rank micro-batches proportional to measured speed.
// The returned config is the submitted one with those decisions written in, so every rank reads
// the same plan.
func planJob(spec *trainmeshv1.JobSpec, nodes []store.Node, machines, ranksPerMachine int) (plan, error) {
	device := spec.GetExecution().GetBackend().GetDevice()
	offering := make([]store.Node, 0, len(nodes))
	for _, node := range nodes {
		if node.Capabilities.Offers(device) {
			offering = append(offering, node)
		}
	}
	chosen, err := pickNodes(offering, machines)
	if err != nil {
		if device != deviceAuto {
			return plan{}, fmt.Errorf("%w (only nodes offering device %q count)", err, device)
		}
		return plan{}, err
	}

	planned := plan{nodes: chosen, communication: chooseCommunication(spec, chosen)}
	if spec.GetTraining().GetBatch().GetBalance() == balanceThrough && len(spec.GetTraining().GetBatch().GetRankBatchSizes()) == 0 {
		weights, err := rankWeights(chosen, device, ranksPerMachine)
		if err != nil {
			return plan{}, err
		}
		total := int(spec.GetTraining().GetBatch().GetPerRank()) * machines * ranksPerMachine
		planned.rankBatchSizes = apportion(total, weights)
	}

	planned.configYAML, err = writePlan(spec, planned)
	return planned, err
}

// chooseCommunication resolves "auto" once for the group: ranks resolving it on their own would
// disagree in a cuda+cpu job (nccl vs gloo) and never rendezvous.
func chooseCommunication(spec *trainmeshv1.JobSpec, chosen []store.Node) string {
	requested := spec.GetExecution().GetBackend().GetCommunication()
	if requested != commAuto && requested != "" {
		return requested
	}
	device := spec.GetExecution().GetBackend().GetDevice()
	for _, node := range chosen {
		effective := device
		if effective == deviceAuto {
			effective = node.Capabilities.Device
		}
		if effective != deviceCUDA {
			return commGloo
		}
	}
	return commNCCL
}

// rankWeights gives each rank its node's measured throughput on the job's device. Ranks are
// assumed to run like the single benchmark process (true for one-thread CPU ranks; ranks sharing
// one accelerator are over-weighted, so keep processes_per_worker at 1 on GPU nodes).
func rankWeights(chosen []store.Node, device string, ranksPerMachine int) ([]float64, error) {
	weights := make([]float64, 0, len(chosen)*ranksPerMachine)
	for _, node := range chosen {
		throughput, ok := node.Capabilities.Throughput(device)
		if !ok || throughput <= 0 {
			return nil, fmt.Errorf("%w: node %s has no measured throughput for %q", ErrNoThroughput, node.ID, device)
		}
		for range ranksPerMachine {
			weights = append(weights, throughput)
		}
	}
	return weights, nil
}

// apportion splits total into len(weights) parts proportional to weights with the
// largest-remainder (Hamilton) method, so the parts always sum to total; ties go to the lower
// rank. A rank rounded below minRankBatch is raised to it, taken from the largest part. total
// must be at least minRankBatch*len(weights) (per_rank >= 1 guarantees it).
func apportion(total int, weights []float64) []int {
	var sum float64
	for _, w := range weights {
		sum += w
	}
	sizes := make([]int, len(weights))
	fractions := make([]float64, len(weights))
	assigned := 0
	for rank, w := range weights {
		share := float64(total) * w / sum
		sizes[rank] = int(math.Floor(share))
		fractions[rank] = share - math.Floor(share)
		assigned += sizes[rank]
	}
	order := make([]int, len(weights))
	for rank := range order {
		order[rank] = rank
	}
	sort.SliceStable(order, func(i, j int) bool { return fractions[order[i]] > fractions[order[j]] })
	for i := 0; i < total-assigned; i++ {
		sizes[order[i]]++
	}
	for rank := range sizes {
		for sizes[rank] < minRankBatch {
			sizes[slices.Index(sizes, slices.Max(sizes))]--
			sizes[rank]++
		}
	}
	return sizes
}

// writePlan returns the spec with the planner's choices as config YAML (JSON is valid YAML).
func writePlan(spec *trainmeshv1.JobSpec, p plan) ([]byte, error) {
	planned, _ := proto.Clone(spec).(*trainmeshv1.JobSpec)
	planned.Execution.Backend.Communication = p.communication // backend is required by validation
	if len(p.rankBatchSizes) > 0 {
		planned.Training.Batch.RankBatchSizes = make([]int32, len(p.rankBatchSizes))
		for i, size := range p.rankBatchSizes {
			planned.Training.Batch.RankBatchSizes[i] = toInt32(size)
		}
	}
	encoded, err := protojson.MarshalOptions{UseProtoNames: true}.Marshal(planned)
	if err != nil {
		return nil, fmt.Errorf("encoding planned config: %w", err)
	}
	return encoded, nil
}
