package coordinator

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/trainmesh/trainmesh/internal/config"
	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/membership"
	"github.com/trainmesh/trainmesh/internal/security"
	"github.com/trainmesh/trainmesh/internal/store"
)

const idRandomBytes = 4

// Server implements CoordinatorService on top of a Store and the cluster CA.
type Server struct {
	trainmeshv1.UnimplementedCoordinatorServiceServer
	store  store.Store
	ca     *security.CA
	cfg    Config
	logger *slog.Logger
}

// NewServer validates cfg (after defaults) and returns a coordinator.
func NewServer(st store.Store, ca *security.CA, cfg Config, logger *slog.Logger) (*Server, error) {
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("coordinator config: %w", err)
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{store: st, ca: ca, cfg: cfg, logger: logger.With("component", "coordinator")}, nil
}

// NewGRPCServer builds a gRPC server serving s with TLS and per-method authorization.
func NewGRPCServer(s *Server, creds credentials.TransportCredentials) *grpc.Server {
	server := grpc.NewServer(grpc.Creds(creds), grpc.UnaryInterceptor(authorize))
	trainmeshv1.RegisterCoordinatorServiceServer(server, s)
	return server
}

// Join redeems a token and issues a node certificate for the caller's CSR.
func (s *Server) Join(_ context.Context, req *trainmeshv1.JoinRequest) (*trainmeshv1.JoinResponse, error) {
	if err := security.ValidateTokenFormat(req.GetToken()); err != nil {
		return nil, status.Error(codes.Unauthenticated, err.Error())
	}
	now := s.cfg.Now()
	nodeID := newID("node")
	certDER, err := s.ca.SignCSR(req.GetCsrDer(), nodeID, security.RoleNode, now, s.cfg.NodeCertValidity)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}

	err = s.store.Update(func(tx store.Tx) error {
		record, err := tx.Token(security.HashToken(req.GetToken()))
		if errors.Is(err, store.ErrNotFound) {
			return security.ErrTokenUnknown
		}
		if err != nil {
			return err
		}
		if err := security.CheckRedeemable(record, now); err != nil {
			return err
		}
		record.UsesLeft--
		if err := tx.PutToken(record); err != nil {
			return err
		}
		state, err := membership.Transition(membership.StateJoining, membership.StateProfiling)
		if err != nil {
			return err
		}
		return tx.PutNode(store.Node{ID: nodeID, State: state, JoinedAt: now, LastHeartbeat: now})
	})
	if err != nil {
		s.logger.Warn("join rejected", "error", err)
		return nil, toStatus(err)
	}
	s.logger.Info("node joined", "node", nodeID)
	return &trainmeshv1.JoinResponse{
		NodeId:              nodeID,
		CertificateDer:      certDER,
		CaCertificateDer:    s.ca.Cert.Raw,
		HeartbeatIntervalMs: millis(s.cfg.HeartbeatInterval),
	}, nil
}

// Heartbeat records liveness and capabilities and returns directives for this node.
func (s *Server) Heartbeat(ctx context.Context, req *trainmeshv1.HeartbeatRequest) (*trainmeshv1.HeartbeatResponse, error) {
	nodeID := callerID(ctx)
	now := s.cfg.Now()
	response := &trainmeshv1.HeartbeatResponse{HeartbeatIntervalMs: millis(s.cfg.HeartbeatInterval)}

	err := s.store.Update(func(tx store.Tx) error {
		node, err := tx.Node(nodeID)
		if err != nil {
			return err
		}
		previous := node.State
		node.LastHeartbeat = now
		applyHeartbeat(&node, req)
		job, hasJob := currentJob(tx, node)
		node.State = stateAfterHeartbeat(node, hasJob && !job.State.Terminal())
		if node.State != previous {
			s.logger.Info("node state changed", "node", node.ID, "from", previous, "to", node.State)
		}
		if hasJob {
			response.Directives = directivesFor(job, node.ID)
		}
		response.State = string(node.State)
		return tx.PutNode(node)
	})
	if err != nil {
		return nil, toStatus(err)
	}
	return response, nil
}

// ReportAssignment applies a node's progress report to its job.
func (s *Server) ReportAssignment(ctx context.Context, req *trainmeshv1.ReportAssignmentRequest) (*trainmeshv1.ReportAssignmentResponse, error) {
	nodeID := callerID(ctx)
	reported := store.JobState(req.GetState())
	if reported != store.JobRunning && !reported.Terminal() {
		return nil, status.Errorf(codes.InvalidArgument, "state %q is not RUNNING, SUCCEEDED, FAILED or CANCELLED", req.GetState())
	}
	now := s.cfg.Now()

	err := s.store.Update(func(tx store.Tx) error {
		job, err := tx.Job(req.GetJobId())
		if err != nil {
			return err
		}
		assignment, ok := job.AssignmentFor(nodeID)
		if !ok {
			return status.Errorf(codes.PermissionDenied, "node %s has no assignment in job %s", nodeID, job.ID)
		}
		if assignment.State.Terminal() {
			return nil // duplicate or late report; already settled
		}
		assignment.State, assignment.Message = reported, req.GetMessage()
		if reported == store.JobSucceeded && assignment.NodeRank == 0 {
			job.Result = req.GetResultJson()
		}
		settleJob(&job, *assignment, now)
		if err := finishPending(tx, &job); err != nil {
			return err
		}
		if err := tx.PutJob(job); err != nil {
			return err
		}
		if reported.Terminal() {
			return releaseNode(tx, nodeID, job.ID)
		}
		return nil
	})
	if err != nil {
		return nil, toStatus(err)
	}
	s.logger.Info("assignment report", "node", nodeID, "job", req.GetJobId(), "state", reported, "message", req.GetMessage())
	return &trainmeshv1.ReportAssignmentResponse{}, nil
}

// CreateToken stores a new join token and returns the join string for it.
func (s *Server) CreateToken(_ context.Context, req *trainmeshv1.CreateTokenRequest) (*trainmeshv1.CreateTokenResponse, error) {
	ttl := time.Duration(req.GetTtlSeconds()) * time.Second
	if ttl == 0 {
		ttl = s.cfg.TokenTTL
	}
	uses := int(req.GetUses())
	if uses == 0 {
		uses = 1
	}
	secret, record, err := security.NewToken(s.cfg.Now(), ttl, uses)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	if err := s.store.Update(func(tx store.Tx) error { return tx.PutToken(record) }); err != nil {
		return nil, toStatus(err)
	}
	join := security.JoinString{Address: s.cfg.AdvertiseAddress, Token: secret, CAHash: s.ca.Hash()}
	s.logger.Info("join token created", "expires_at", record.ExpiresAt, "uses", uses)
	return &trainmeshv1.CreateTokenResponse{JoinString: join.String(), ExpiresAt: timestamp(record.ExpiresAt)}, nil
}

// ListNodes returns every node.
func (s *Server) ListNodes(context.Context, *trainmeshv1.ListNodesRequest) (*trainmeshv1.ListNodesResponse, error) {
	var nodes []store.Node
	if err := s.store.View(func(tx store.Tx) (err error) { nodes, err = tx.Nodes(); return err }); err != nil {
		return nil, toStatus(err)
	}
	response := &trainmeshv1.ListNodesResponse{}
	for _, node := range nodes {
		response.Nodes = append(response.Nodes, nodeToProto(node))
	}
	return response, nil
}

// SubmitJob validates the config and assigns it to READY nodes.
func (s *Server) SubmitJob(_ context.Context, req *trainmeshv1.SubmitJobRequest) (*trainmeshv1.SubmitJobResponse, error) {
	parsed, err := config.Parse(req.GetConfigYaml())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	machines := parsed.WorldSize() / int(parsed.Spec.GetExecution().GetProcessesPerWorker())
	job := store.Job{
		ID:              newID("job"),
		State:           store.JobRunning,
		ConfigYAML:      req.GetConfigYaml(),
		Machines:        machines,
		RanksPerMachine: int(parsed.Spec.GetExecution().GetProcessesPerWorker()),
		MasterPort:      s.cfg.RendezvousPort,
		CreatedAt:       s.cfg.Now(),
	}

	err = s.store.Update(func(tx store.Tx) error {
		nodes, err := tx.Nodes()
		if err != nil {
			return err
		}
		chosen, err := pickNodes(nodes, machines)
		if err != nil {
			return status.Error(codes.FailedPrecondition, err.Error())
		}
		job.MasterAddr = chosen[0].Address
		for rank, node := range chosen {
			job.Assignments = append(job.Assignments, store.Assignment{NodeID: node.ID, NodeRank: rank, State: store.JobPending})
			if node.State, err = membership.Transition(node.State, membership.StateBusy); err != nil {
				return err
			}
			node.CurrentJobID = job.ID
			if err := tx.PutNode(node); err != nil {
				return err
			}
		}
		return tx.PutJob(job)
	})
	if err != nil {
		return nil, toStatus(err)
	}
	s.logger.Info("job submitted", "job", job.ID, "machines", machines, "master", job.MasterAddr)
	return &trainmeshv1.SubmitJobResponse{JobId: job.ID}, nil
}

// GetJob returns a job with its assignments and result.
func (s *Server) GetJob(_ context.Context, req *trainmeshv1.GetJobRequest) (*trainmeshv1.GetJobResponse, error) {
	var job store.Job
	if err := s.store.View(func(tx store.Tx) (err error) { job, err = tx.Job(req.GetJobId()); return err }); err != nil {
		return nil, toStatus(err)
	}
	return &trainmeshv1.GetJobResponse{Job: jobToProto(job)}, nil
}

// CancelJob marks a running job cancelled; its nodes are told to stop on their next heartbeat.
func (s *Server) CancelJob(_ context.Context, req *trainmeshv1.CancelJobRequest) (*trainmeshv1.CancelJobResponse, error) {
	now := s.cfg.Now()
	err := s.store.Update(func(tx store.Tx) error {
		job, err := tx.Job(req.GetJobId())
		if err != nil {
			return err
		}
		if job.State.Terminal() {
			return nil
		}
		job.State, job.Error, job.FinishedAt = store.JobCancelled, "cancelled by admin", now
		if err := finishPending(tx, &job); err != nil {
			return err
		}
		return tx.PutJob(job)
	})
	if err != nil {
		return nil, toStatus(err)
	}
	s.logger.Info("job cancelled", "job", req.GetJobId())
	return &trainmeshv1.CancelJobResponse{}, nil
}

// Sweep applies the liveness policy to every node. A node that becomes LOST fails its job.
func (s *Server) Sweep() error {
	now := s.cfg.Now()
	return s.store.Update(func(tx store.Tx) error {
		nodes, err := tx.Nodes()
		if err != nil {
			return err
		}
		for _, node := range nodes {
			next, changed := s.cfg.Liveness.Next(node.State, node.LastHeartbeat, now)
			if !changed {
				continue
			}
			s.logger.Warn("node liveness changed", "node", node.ID, "from", node.State, "to", next, "last_heartbeat", node.LastHeartbeat)
			if next == membership.StateLost && node.CurrentJobID != "" {
				if err := failJobForLostNode(tx, node, now); err != nil {
					return err
				}
				node.CurrentJobID = ""
			}
			node.State = next
			if err := tx.PutNode(node); err != nil {
				return err
			}
		}
		return nil
	})
}

// RunSweeper calls Sweep every SweepInterval until ctx ends.
func (s *Server) RunSweeper(ctx context.Context) {
	ticker := time.NewTicker(s.cfg.SweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := s.Sweep(); err != nil {
				s.logger.Error("liveness sweep failed", "error", err)
			}
		}
	}
}

func applyHeartbeat(node *store.Node, req *trainmeshv1.HeartbeatRequest) {
	if caps := req.GetCapabilities(); caps != nil {
		node.Capabilities = capabilitiesFromProto(caps)
	}
	if req.GetAddress() != "" {
		node.Address = req.GetAddress()
	}
	if req.GetInterface() != "" {
		node.Interface = req.GetInterface()
	}
}

// stateAfterHeartbeat: a heartbeat proves the node is alive; profiling ends once capabilities
// are known; a node that recovers returns to BUSY if its job is still running.
func stateAfterHeartbeat(node store.Node, runningJob bool) membership.State {
	profiled := node.Capabilities.LogicalCPUs > 0
	switch node.State {
	case membership.StateProfiling:
		if profiled {
			return membership.StateReady
		}
	case membership.StateDegraded:
		if runningJob {
			return membership.StateBusy
		}
		return membership.StateReady
	case membership.StateLost:
		return membership.StateProfiling
	}
	return node.State
}

func currentJob(tx store.Tx, node store.Node) (store.Job, bool) {
	if node.CurrentJobID == "" {
		return store.Job{}, false
	}
	job, err := tx.Job(node.CurrentJobID)
	return job, err == nil
}

// directivesFor returns what this node must do for its current job, if anything. Start is
// repeated until the agent reports RUNNING, so a lost heartbeat response is harmless.
func directivesFor(job store.Job, nodeID string) []*trainmeshv1.Directive {
	assignment, ok := job.AssignmentFor(nodeID)
	if !ok || assignment.State.Terminal() {
		return nil
	}
	if job.State.Terminal() {
		reason := job.Error
		if reason == "" {
			reason = "job " + string(job.State)
		}
		return []*trainmeshv1.Directive{{Kind: &trainmeshv1.Directive_Cancel{Cancel: &trainmeshv1.CancelAssignment{JobId: job.ID, Reason: reason}}}}
	}
	if assignment.State != store.JobPending {
		return nil
	}
	return []*trainmeshv1.Directive{{Kind: &trainmeshv1.Directive_Start{Start: &trainmeshv1.StartAssignment{
		JobId:           job.ID,
		NodeRank:        toInt32(assignment.NodeRank),
		Machines:        toInt32(job.Machines),
		RanksPerMachine: toInt32(job.RanksPerMachine),
		MasterAddr:      job.MasterAddr,
		MasterPort:      toInt32(job.MasterPort),
		ConfigYaml:      job.ConfigYAML,
	}}}}
}

// settleJob moves the job forward after one assignment changed.
func settleJob(job *store.Job, changed store.Assignment, now time.Time) {
	if job.State.Terminal() {
		return
	}
	switch changed.State {
	case store.JobFailed, store.JobCancelled:
		job.State = store.JobFailed
		job.Error = fmt.Sprintf("node %s (rank %d) %s: %s", changed.NodeID, changed.NodeRank, changed.State, changed.Message)
		job.FinishedAt = now
	case store.JobSucceeded:
		for _, assignment := range job.Assignments {
			if assignment.State != store.JobSucceeded {
				return
			}
		}
		job.State, job.FinishedAt = store.JobSucceeded, now
	}
}

func releaseNode(tx store.Tx, nodeID, jobID string) error {
	node, err := tx.Node(nodeID)
	if err != nil {
		return err
	}
	if node.CurrentJobID != jobID {
		return nil
	}
	node.CurrentJobID = ""
	if node.State == membership.StateBusy {
		node.State = membership.StateReady
	}
	return tx.PutNode(node)
}

func failJobForLostNode(tx store.Tx, node store.Node, now time.Time) error {
	job, err := tx.Job(node.CurrentJobID)
	if err != nil {
		return err
	}
	if assignment, ok := job.AssignmentFor(node.ID); ok && !assignment.State.Terminal() {
		assignment.State, assignment.Message = store.JobFailed, "node lost (no heartbeats)"
		settleJob(&job, *assignment, now)
	}
	if err := finishPending(tx, &job); err != nil {
		return err
	}
	return tx.PutJob(job)
}

// finishPending settles assignments that never started once the job has ended and frees their
// nodes. Without it a node whose start directive lost the race with a failure stays BUSY forever:
// it would only ever be told to cancel a job it never started.
func finishPending(tx store.Tx, job *store.Job) error {
	if !job.State.Terminal() {
		return nil
	}
	for i := range job.Assignments {
		assignment := &job.Assignments[i]
		if assignment.State != store.JobPending {
			continue
		}
		assignment.State, assignment.Message = store.JobCancelled, "job ended before this node started"
		if err := releaseNode(tx, assignment.NodeID, job.ID); err != nil {
			return err
		}
	}
	return nil
}

func newID(prefix string) string {
	raw := make([]byte, idRandomBytes)
	_, _ = rand.Read(raw) // crypto/rand never fails on supported platforms
	return prefix + "-" + hex.EncodeToString(raw)
}

// toStatus maps domain errors to gRPC codes; errors that already carry a status pass through.
func toStatus(err error) error {
	if _, ok := status.FromError(err); ok {
		return err
	}
	switch {
	case errors.Is(err, store.ErrNotFound):
		return status.Error(codes.NotFound, err.Error())
	case errors.Is(err, security.ErrTokenUnknown), errors.Is(err, security.ErrTokenExpired),
		errors.Is(err, security.ErrTokenExhausted), errors.Is(err, security.ErrTokenRevoked):
		return status.Error(codes.Unauthenticated, err.Error())
	case errors.Is(err, membership.ErrIllegalTransition):
		return status.Error(codes.FailedPrecondition, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}
