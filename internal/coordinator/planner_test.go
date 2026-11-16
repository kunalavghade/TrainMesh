package coordinator

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"testing"

	"github.com/trainmesh/trainmesh/internal/config"
	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/store"
)

const plannerJobYAML = `version: 1
execution:
  strategy: data_parallel
  workers: %d
  processes_per_worker: %d
  backend: {device: %s, communication: %s}
training:
  dataset: {type: teacher, samples: 4096}
  batch: {per_rank: 64, balance: %s}
  epochs: 1
`

func plannerSpec(t *testing.T, machines, ranks int, device, communication, balance string) config.Config {
	t.Helper()
	parsed, err := config.Parse(fmt.Appendf(nil, plannerJobYAML, machines, ranks, device, communication, balance))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return parsed
}

func profiledNode(id, device string, profiles map[string]float64) store.Node {
	caps := store.Capabilities{TorchVersion: "2.15", RuntimeVersion: "0.3.0", Device: device}
	for _, kind := range slices.Sorted(maps.Keys(profiles)) {
		caps.Devices = append(caps.Devices, store.DeviceProfile{Kind: kind, SamplesPerSecond: profiles[kind]})
	}
	return store.Node{ID: id, State: membership.StateReady, Address: "10.0.0." + id, Capabilities: caps}
}

func TestApportionKeepsTheTotalAndFollowsTheWeights(t *testing.T) {
	cases := []struct {
		total   int
		weights []float64
		want    []int
	}{
		{128, []float64{1, 1}, []int{64, 64}},
		{128, []float64{3, 1}, []int{96, 32}},
		{128, []float64{512000, 270000}, []int{84, 44}}, // 83.8 / 44.2
		{10, []float64{1, 1, 1}, []int{4, 3, 3}},        // tie goes to the lower rank
		{3, []float64{1000, 1, 1}, []int{1, 1, 1}},      // every rank keeps at least one sample
		{100, []float64{1e6, 1, 1}, []int{98, 1, 1}},    // slow ranks still get the minimum
		{7, []float64{2, 1}, []int{5, 2}},               // 7 * 2/3 = 4.67 rounds up by remainder
	}
	for _, tc := range cases {
		got := apportion(tc.total, tc.weights)
		sum := 0
		for _, size := range got {
			sum += size
		}
		if !slices.Equal(got, tc.want) || sum != tc.total {
			t.Errorf("apportion(%d, %v) = %v (sum %d), want %v", tc.total, tc.weights, got, sum, tc.want)
		}
	}
}

func TestPlanSplitsBatchesByMeasuredThroughput(t *testing.T) {
	nodes := []store.Node{
		profiledNode("1", "mps", map[string]float64{"mps": 512000, "cpu": 270000}),
		profiledNode("2", "cpu", map[string]float64{"cpu": 270000}),
	}
	spec := plannerSpec(t, 2, 1, "auto", "auto", "throughput")
	planned, err := planJob(spec.Spec, nodes, 2, 1)
	if err != nil {
		t.Fatalf("planJob: %v", err)
	}
	if !slices.Equal(planned.rankBatchSizes, []int{84, 44}) {
		t.Errorf("rank batches = %v, want [84 44] (512k vs 270k samples/s)", planned.rankBatchSizes)
	}
	if planned.communication != commGloo {
		t.Errorf("communication = %q, want gloo for an mps+cpu group", planned.communication)
	}

	written, err := config.Parse(planned.configYAML)
	if err != nil {
		t.Fatalf("planned config does not parse: %v\n%s", err, planned.configYAML)
	}
	if got := written.Spec.GetTraining().GetBatch().GetRankBatchSizes(); !slices.Equal(got, []int32{84, 44}) {
		t.Errorf("planned config rank_batch_sizes = %v", got)
	}
	if got := written.Spec.GetExecution().GetBackend().GetCommunication(); got != commGloo {
		t.Errorf("planned config communication = %q", got)
	}
}

func TestPlanUsesTheThroughputOfTheRequestedDevice(t *testing.T) {
	nodes := []store.Node{
		profiledNode("1", "mps", map[string]float64{"mps": 500000, "cpu": 300000}),
		profiledNode("2", "cpu", map[string]float64{"cpu": 100000}),
	}
	planned, err := planJob(plannerSpec(t, 2, 1, "cpu", "gloo", "throughput").Spec, nodes, 2, 1)
	if err != nil {
		t.Fatalf("planJob: %v", err)
	}
	if !slices.Equal(planned.rankBatchSizes, []int{96, 32}) {
		t.Errorf("rank batches = %v, want [96 32] from the cpu profiles", planned.rankBatchSizes)
	}
}

func TestUniformBalanceLeavesBatchesAlone(t *testing.T) {
	nodes := []store.Node{
		profiledNode("1", "mps", map[string]float64{"mps": 500000}),
		profiledNode("2", "cpu", map[string]float64{"cpu": 100000}),
	}
	planned, err := planJob(plannerSpec(t, 2, 1, "auto", "auto", "uniform").Spec, nodes, 2, 1)
	if err != nil {
		t.Fatalf("planJob: %v", err)
	}
	if len(planned.rankBatchSizes) != 0 {
		t.Errorf("rank batches = %v, want none for balance=uniform", planned.rankBatchSizes)
	}
}

func TestPlanOnlyPicksNodesOfferingTheDevice(t *testing.T) {
	nodes := []store.Node{
		profiledNode("1", "cpu", map[string]float64{"cpu": 1}),
		profiledNode("2", "mps", map[string]float64{"mps": 1}),
		profiledNode("3", "mps", map[string]float64{"mps": 1, "cpu": 1}),
	}
	planned, err := planJob(plannerSpec(t, 2, 1, "mps", "gloo", "uniform").Spec, nodes, 2, 1)
	if err != nil {
		t.Fatalf("planJob: %v", err)
	}
	ids := []string{planned.nodes[0].ID, planned.nodes[1].ID}
	if !slices.Equal(ids, []string{"2", "3"}) {
		t.Errorf("chosen nodes = %v, want the two mps nodes", ids)
	}

	if _, err := planJob(plannerSpec(t, 3, 1, "mps", "gloo", "uniform").Spec, nodes, 3, 1); !errors.Is(err, ErrNotEnoughNodes) {
		t.Errorf("3 mps machines from 2 offering nodes: err = %v, want ErrNotEnoughNodes", err)
	}
}

func TestCommunicationAutoIsDecidedForTheWholeGroup(t *testing.T) {
	cuda := profiledNode("1", "cuda", map[string]float64{"cuda": 1})
	otherCUDA := profiledNode("2", "cuda", map[string]float64{"cuda": 1})
	cpu := profiledNode("3", "cpu", map[string]float64{"cpu": 1})
	auto := plannerSpec(t, 2, 1, "auto", "auto", "uniform").Spec

	if got := chooseCommunication(auto, []store.Node{cuda, otherCUDA}); got != commNCCL {
		t.Errorf("all-cuda group: %q, want nccl", got)
	}
	if got := chooseCommunication(auto, []store.Node{cuda, cpu}); got != commGloo {
		t.Errorf("cuda+cpu group: %q, want gloo (nccl cannot carry cpu ranks)", got)
	}
	cpuJob := plannerSpec(t, 2, 1, "cpu", "auto", "uniform").Spec
	if got := chooseCommunication(cpuJob, []store.Node{cuda, otherCUDA}); got != commGloo {
		t.Errorf("device=cpu on cuda nodes: %q, want gloo", got)
	}
	explicit := plannerSpec(t, 2, 1, "auto", "gloo", "uniform").Spec
	if got := chooseCommunication(explicit, []store.Node{cuda, otherCUDA}); got != commGloo {
		t.Errorf("explicit gloo was overridden: %q", got)
	}
}

func TestThroughputBalanceNeedsEveryProfile(t *testing.T) {
	legacy := store.Node{ID: "2", State: membership.StateReady, Address: "10.0.0.2",
		Capabilities: store.Capabilities{TorchVersion: "2.15", RuntimeVersion: "0.3.0"}}
	nodes := []store.Node{profiledNode("1", "cpu", map[string]float64{"cpu": 1}), legacy}
	_, err := planJob(plannerSpec(t, 2, 1, "cpu", "gloo", "throughput").Spec, nodes, 2, 1)
	if !errors.Is(err, ErrNoThroughput) {
		t.Errorf("err = %v, want ErrNoThroughput for a node without a profile", err)
	}
}

func TestIslandFollowsTheNodeDevice(t *testing.T) {
	for device, want := range map[string]string{"cuda": islandCUDA, "mps": islandHost, "cpu": islandHost, "": islandHost} {
		if got := island(store.Capabilities{Device: device}); got != want {
			t.Errorf("island(%q) = %q, want %q", device, got, want)
		}
	}
}
