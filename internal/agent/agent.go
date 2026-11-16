package agent

import (
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/security"
)

// Defaults for Config fields left zero.
const (
	DefaultRPCTimeout     = 10 * time.Second
	DefaultReportAttempts = 5
	DefaultReportBackoff  = 500 * time.Millisecond
	DefaultShutdownGrace  = 15 * time.Second
	fallbackHeartbeat     = 2 * time.Second
)

// Job states reported to the coordinator.
const (
	reportRunning   = "RUNNING"
	reportSucceeded = "SUCCEEDED"
	reportFailed    = "FAILED"
	reportCancelled = "CANCELLED"
)

// ErrRemoved means the coordinator no longer knows this node; it must join again.
var ErrRemoved = errors.New("coordinator does not know this node; remove the agent state dir and join again")

var errCancelledByCoordinator = errors.New("cancelled by coordinator")

// Config configures an agent. Capabilities and Runner are required.
type Config struct {
	StateDir       string
	JoinString     string // used only when StateDir has no identity yet
	Address        string // advertised IP; detected from the route to the coordinator if empty
	Interface      string // NIC owning Address; detected if empty
	Capabilities   func(context.Context) (*trainmeshv1.Capabilities, error)
	Runner         Runner
	RPCTimeout     time.Duration
	ReportAttempts int
	ReportBackoff  time.Duration
	ShutdownGrace  time.Duration
	Logger         *slog.Logger
}

// Agent is one machine's member of the cluster.
type Agent struct {
	cfg      Config
	identity Identity
	client   trainmeshv1.CoordinatorServiceClient
	conn     *grpc.ClientConn
	caps     *trainmeshv1.Capabilities
	logger   *slog.Logger

	mu       sync.Mutex
	running  map[string]context.CancelCauseFunc
	finished map[string]bool
	jobs     sync.WaitGroup
}

// New applies defaults; it does not touch the network.
func New(cfg Config) (*Agent, error) {
	if cfg.Capabilities == nil || cfg.Runner == nil || cfg.StateDir == "" {
		return nil, errors.New("agent config needs StateDir, Capabilities and Runner")
	}
	if cfg.RPCTimeout == 0 {
		cfg.RPCTimeout = DefaultRPCTimeout
	}
	if cfg.ReportAttempts == 0 {
		cfg.ReportAttempts = DefaultReportAttempts
	}
	if cfg.ReportBackoff == 0 {
		cfg.ReportBackoff = DefaultReportBackoff
	}
	if cfg.ShutdownGrace == 0 {
		cfg.ShutdownGrace = DefaultShutdownGrace
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &Agent{
		cfg:      cfg,
		logger:   cfg.Logger.With("component", "agent"),
		running:  map[string]context.CancelCauseFunc{},
		finished: map[string]bool{},
	}, nil
}

// NodeID is known after Start.
func (a *Agent) NodeID() string { return a.identity.NodeID }

// Interface is the network interface used for training traffic, known after Start.
func (a *Agent) Interface() string { return a.cfg.Interface }

// Device is the device this node trains on for device=auto jobs, known after Start.
func (a *Agent) Device() string { return a.caps.GetDevice() }

// Start joins if needed, connects with the node certificate and probes capabilities.
func (a *Agent) Start(ctx context.Context) error {
	identity, err := LoadIdentity(a.cfg.StateDir)
	if errors.Is(err, ErrNotJoined) && a.cfg.JoinString != "" {
		identity, err = a.join(ctx)
	}
	if err != nil {
		return err
	}
	a.identity = identity
	a.logger = a.logger.With("node", identity.NodeID)

	if a.cfg.Address == "" || a.cfg.Interface == "" {
		address, iface, err := DetectRoute(identity.Coordinator)
		if err != nil {
			return err
		}
		a.cfg.Address, a.cfg.Interface = firstNonEmpty(a.cfg.Address, address), firstNonEmpty(a.cfg.Interface, iface)
	}
	if a.caps, err = a.cfg.Capabilities(ctx); err != nil {
		return fmt.Errorf("probing capabilities: %w", err)
	}
	tlsConfig := security.ClientTLS(identity.CA, identity.CertDER, identity.Key)
	if a.conn, err = grpc.NewClient(identity.Coordinator, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig))); err != nil {
		return fmt.Errorf("connecting to %s: %w", identity.Coordinator, err)
	}
	a.client = trainmeshv1.NewCoordinatorServiceClient(a.conn)
	a.logger.Info("agent started", "coordinator", identity.Coordinator, "address", a.cfg.Address,
		"interface", a.cfg.Interface, "device", a.caps.GetDevice(), "profiles", deviceProfiles(a.caps))
	return nil
}

// Run heartbeats until ctx ends or the coordinator rejects this node, then stops running jobs.
func (a *Agent) Run(ctx context.Context) error {
	defer a.shutdown()
	interval := fallbackHeartbeat
	for {
		next, err := a.heartbeat(ctx)
		switch {
		case err == nil:
			interval = next
		case status.Code(err) == codes.NotFound:
			return fmt.Errorf("%w: %w", ErrRemoved, err)
		case isFatal(err):
			return err
		case ctx.Err() == nil:
			a.logger.Warn("heartbeat failed; retrying", "error", err)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(interval):
		}
	}
}

func (a *Agent) heartbeat(ctx context.Context) (time.Duration, error) {
	rpcCtx, cancel := context.WithTimeout(ctx, a.cfg.RPCTimeout)
	defer cancel()
	response, err := a.client.Heartbeat(rpcCtx, &trainmeshv1.HeartbeatRequest{
		Capabilities: a.caps, Address: a.cfg.Address, Interface: a.cfg.Interface,
	})
	if err != nil {
		return 0, err
	}
	for _, directive := range response.GetDirectives() {
		a.apply(ctx, directive)
	}
	return max(time.Duration(response.GetHeartbeatIntervalMs())*time.Millisecond, 100*time.Millisecond), nil
}

func (a *Agent) apply(ctx context.Context, directive *trainmeshv1.Directive) {
	if start := directive.GetStart(); start != nil {
		a.startAssignment(ctx, start)
	}
	if cancel := directive.GetCancel(); cancel != nil {
		a.mu.Lock()
		stop, running := a.running[cancel.GetJobId()]
		a.mu.Unlock()
		if running {
			a.logger.Info("cancelling assignment", "job", cancel.GetJobId(), "reason", cancel.GetReason())
			stop(fmt.Errorf("%w: %s", errCancelledByCoordinator, cancel.GetReason()))
		}
	}
}

// startAssignment is idempotent: the coordinator repeats Start until it sees RUNNING.
func (a *Agent) startAssignment(ctx context.Context, start *trainmeshv1.StartAssignment) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, running := a.running[start.GetJobId()]; running || a.finished[start.GetJobId()] {
		return
	}
	jobCtx, cancel := context.WithCancelCause(ctx)
	a.running[start.GetJobId()] = cancel
	a.jobs.Add(1)
	go a.execute(ctx, jobCtx, Assignment{
		JobID:           start.GetJobId(),
		NodeRank:        int(start.GetNodeRank()),
		Machines:        int(start.GetMachines()),
		RanksPerMachine: int(start.GetRanksPerMachine()),
		MasterAddr:      start.GetMasterAddr(),
		MasterPort:      int(start.GetMasterPort()),
		ConfigYAML:      start.GetConfigYaml(),
	})
}

func (a *Agent) execute(agentCtx, jobCtx context.Context, assignment Assignment) {
	defer a.jobs.Done()
	logger := a.logger.With("job", assignment.JobID, "node_rank", assignment.NodeRank)
	logger.Info("assignment started", "machines", assignment.Machines, "master", fmt.Sprintf("%s:%d", assignment.MasterAddr, assignment.MasterPort))
	a.report(assignment.JobID, reportRunning, "", nil)

	result, err := a.cfg.Runner.Run(jobCtx, assignment)

	state, message := reportSucceeded, ""
	switch {
	case err == nil:
	case errors.Is(context.Cause(jobCtx), errCancelledByCoordinator):
		state, message = reportCancelled, context.Cause(jobCtx).Error()
	case agentCtx.Err() != nil:
		state, message = reportCancelled, "agent stopped"
	default:
		state, message = reportFailed, err.Error()
	}
	logger.Info("assignment finished", "state", state, "message", message)
	a.report(assignment.JobID, state, message, result)

	a.mu.Lock()
	if cancel, ok := a.running[assignment.JobID]; ok {
		cancel(nil)
	}
	delete(a.running, assignment.JobID)
	a.finished[assignment.JobID] = true
	a.mu.Unlock()
}

// report retries with linear backoff; it uses its own context so a final report still goes out
// while the agent is shutting down.
func (a *Agent) report(jobID, state, message string, result []byte) {
	request := &trainmeshv1.ReportAssignmentRequest{JobId: jobID, State: state, Message: message, ResultJson: result}
	var err error
	for attempt := 1; attempt <= a.cfg.ReportAttempts; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), a.cfg.RPCTimeout)
		_, err = a.client.ReportAssignment(ctx, request)
		cancel()
		if err == nil || isFatal(err) {
			break
		}
		time.Sleep(time.Duration(attempt) * a.cfg.ReportBackoff)
	}
	if err != nil {
		a.logger.Error("could not report assignment state", "job", jobID, "state", state, "error", err)
	}
}

func (a *Agent) shutdown() {
	a.mu.Lock()
	for _, cancel := range a.running {
		cancel(errors.New("agent stopped"))
	}
	a.mu.Unlock()
	done := make(chan struct{})
	go func() { a.jobs.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(a.cfg.ShutdownGrace):
		a.logger.Warn("jobs did not stop within the shutdown grace period")
	}
	if a.conn != nil {
		_ = a.conn.Close()
	}
}

func (a *Agent) join(ctx context.Context) (Identity, error) {
	join, err := security.ParseJoinString(a.cfg.JoinString)
	if err != nil {
		return Identity{}, err
	}
	key, err := security.GenerateKey()
	if err != nil {
		return Identity{}, err
	}
	csr, err := security.CreateCSR(key)
	if err != nil {
		return Identity{}, err
	}
	conn, err := grpc.NewClient(join.Address, grpc.WithTransportCredentials(credentials.NewTLS(security.PinnedTLS(join.CAHash))))
	if err != nil {
		return Identity{}, fmt.Errorf("connecting to %s: %w", join.Address, err)
	}
	defer func() { _ = conn.Close() }()

	rpcCtx, cancel := context.WithTimeout(ctx, a.cfg.RPCTimeout)
	defer cancel()
	response, err := trainmeshv1.NewCoordinatorServiceClient(conn).Join(rpcCtx, &trainmeshv1.JoinRequest{Token: join.Token, CsrDer: csr})
	if err != nil {
		return Identity{}, fmt.Errorf("joining %s: %w", join.Address, err)
	}
	if security.CertificateHash(response.GetCaCertificateDer()) != join.CAHash {
		return Identity{}, errors.New("coordinator returned a CA that does not match the join string")
	}
	identity := Identity{NodeID: response.GetNodeId(), Coordinator: join.Address, CertDER: response.GetCertificateDer(), Key: key}
	if identity.CA, err = x509.ParseCertificate(response.GetCaCertificateDer()); err != nil {
		return Identity{}, err
	}
	if err := identity.Save(a.cfg.StateDir); err != nil {
		return Identity{}, err
	}
	a.logger.Info("joined cluster", "node", identity.NodeID, "coordinator", join.Address)
	return identity, nil
}

// isFatal: errors that retrying cannot fix.
func isFatal(err error) bool {
	switch status.Code(err) {
	case codes.NotFound, codes.Unauthenticated, codes.PermissionDenied:
		return true
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// deviceProfiles formats measured throughputs for logs, e.g. "mps=512000/s cpu=270000/s".
func deviceProfiles(caps *trainmeshv1.Capabilities) string {
	parts := make([]string, 0, len(caps.GetDevices()))
	for _, d := range caps.GetDevices() {
		parts = append(parts, fmt.Sprintf("%s=%.0f/s", d.GetKind(), d.GetSamplesPerSecond()))
	}
	return strings.Join(parts, " ")
}
