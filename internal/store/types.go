// Package store persists coordinator state (join tokens, nodes, jobs) behind a small
// transactional interface. The bbolt implementation keeps everything in one local file.
package store

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/security"
)

// ErrNotFound is returned when a record does not exist.
var ErrNotFound = errors.New("record not found")

// Capabilities is what an agent reports about its machine.
type Capabilities struct {
	Hostname       string          `json:"hostname"`
	OS             string          `json:"os"`
	Arch           string          `json:"arch"`
	LogicalCPUs    int             `json:"logical_cpus"`
	MemoryBytes    uint64          `json:"memory_bytes"`
	RuntimeVersion string          `json:"runtime_version"`
	TorchVersion   string          `json:"torch_version"`
	Device         string          `json:"device"`
	Devices        []DeviceProfile `json:"devices,omitempty"`
}

// DeviceProfile is a measured throughput of one device kind on a node.
type DeviceProfile struct {
	Kind             string  `json:"kind"`
	SamplesPerSecond float64 `json:"samples_per_second"`
}

// Throughput returns the measured samples/s of kind ("auto" means the node's own device).
func (c Capabilities) Throughput(kind string) (float64, bool) {
	if kind == "auto" {
		kind = c.Device
	}
	for _, profile := range c.Devices {
		if profile.Kind == kind {
			return profile.SamplesPerSecond, true
		}
	}
	return 0, false
}

// Offers reports whether the node can train on kind. "auto" is always offered; a node that
// reported no device (an agent older than device profiling) is assumed to offer only cpu.
func (c Capabilities) Offers(kind string) bool {
	switch {
	case kind == "auto" || kind == c.Device:
		return true
	case c.Device == "":
		return kind == "cpu"
	}
	_, ok := c.Throughput(kind)
	return ok
}

// Node is one machine that joined the cluster.
type Node struct {
	ID            string           `json:"id"`
	State         membership.State `json:"state"`
	Address       string           `json:"address"`   // IP other nodes use to reach it
	Interface     string           `json:"interface"` // NIC name for Gloo
	Capabilities  Capabilities     `json:"capabilities"`
	JoinedAt      time.Time        `json:"joined_at"`
	LastHeartbeat time.Time        `json:"last_heartbeat"`
	CurrentJobID  string           `json:"current_job_id,omitempty"`
}

// JobState is the lifecycle of a training job.
type JobState string

// Job states. A job is terminal once it reaches SUCCEEDED, FAILED or CANCELLED.
const (
	JobPending   JobState = "PENDING"
	JobRunning   JobState = "RUNNING"
	JobSucceeded JobState = "SUCCEEDED"
	JobFailed    JobState = "FAILED"
	JobCancelled JobState = "CANCELLED"
)

// Terminal reports whether no further transitions happen.
func (s JobState) Terminal() bool {
	return s == JobSucceeded || s == JobFailed || s == JobCancelled
}

// Assignment is one node's part of a job.
type Assignment struct {
	NodeID   string   `json:"node_id"`
	NodeRank int      `json:"node_rank"`
	State    JobState `json:"state"`
	Message  string   `json:"message,omitempty"`
}

// Job is a training run spread over one or more nodes.
type Job struct {
	ID              string          `json:"id"`
	State           JobState        `json:"state"`
	ConfigYAML      []byte          `json:"config_yaml"`
	Machines        int             `json:"machines"`
	RanksPerMachine int             `json:"ranks_per_machine"`
	MasterAddr      string          `json:"master_addr,omitempty"`
	MasterPort      int             `json:"master_port,omitempty"`
	Communication   string          `json:"communication,omitempty"`
	RankBatchSizes  []int           `json:"rank_batch_sizes,omitempty"`
	Assignments     []Assignment    `json:"assignments,omitempty"`
	Result          json.RawMessage `json:"result,omitempty"`
	Error           string          `json:"error,omitempty"`
	CreatedAt       time.Time       `json:"created_at"`
	FinishedAt      time.Time       `json:"finished_at,omitempty"`
}

// AssignmentFor returns the node's assignment in this job.
func (j *Job) AssignmentFor(nodeID string) (*Assignment, bool) {
	for i := range j.Assignments {
		if j.Assignments[i].NodeID == nodeID {
			return &j.Assignments[i], true
		}
	}
	return nil, false
}

// Tx reads and writes records inside one transaction.
type Tx interface {
	Token(hash string) (security.TokenRecord, error)
	PutToken(record security.TokenRecord) error
	Node(id string) (Node, error)
	PutNode(node Node) error
	Nodes() ([]Node, error)
	Job(id string) (Job, error)
	PutJob(job Job) error
	Jobs() ([]Job, error)
}

// Store runs functions in read-only or read-write transactions. A non-nil error from fn rolls
// the transaction back.
type Store interface {
	View(fn func(Tx) error) error
	Update(fn func(Tx) error) error
	Close() error
}
