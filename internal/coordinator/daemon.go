package coordinator

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/trainmesh/trainmesh/internal/store"
)

// DefaultShutdownGrace bounds how long meshd waits for in-flight RPCs on shutdown.
const DefaultShutdownGrace = 10 * time.Second

// DaemonOptions configures RunDaemon.
type DaemonOptions struct {
	StateDir      string
	ListenAddress string
	Config        Config
	Logger        *slog.Logger
	// Ready, if set, receives the bound address once the server accepts connections.
	Ready chan<- string
}

// RunDaemon prepares the state dir, serves CoordinatorService and sweeps liveness until ctx ends.
func RunDaemon(ctx context.Context, opts DaemonOptions) error {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}
	cfg := opts.Config.WithDefaults()
	ca, serverTLS, err := PrepareStateDir(opts.StateDir, cfg.AdvertiseAddress, cfg.Now())
	if err != nil {
		return err
	}
	st, err := store.OpenBolt(filepath.Join(opts.StateDir, StateFile))
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()
	server, err := NewServer(st, ca, cfg, logger)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", opts.ListenAddress)
	if err != nil {
		return fmt.Errorf("listening on %s: %w", opts.ListenAddress, err)
	}
	grpcServer := NewGRPCServer(server, credentials.NewTLS(serverTLS))

	sweepCtx, stopSweeping := context.WithCancel(ctx)
	defer stopSweeping()
	go server.RunSweeper(sweepCtx)

	served := make(chan error, 1)
	go func() { served <- grpcServer.Serve(listener) }()
	logger.Info("meshd listening", "listen", listener.Addr().String(), "advertise", cfg.AdvertiseAddress,
		"ca_hash", ca.Hash(), "state_dir", opts.StateDir)
	if opts.Ready != nil {
		opts.Ready <- listener.Addr().String()
	}

	select {
	case err := <-served:
		if errors.Is(err, grpc.ErrServerStopped) {
			return nil
		}
		return fmt.Errorf("serving: %w", err)
	case <-ctx.Done():
		logger.Info("meshd shutting down")
		stopGracefully(grpcServer, DefaultShutdownGrace)
		return nil
	}
}

func stopGracefully(server *grpc.Server, grace time.Duration) {
	stopped := make(chan struct{})
	go func() { server.GracefulStop(); close(stopped) }()
	select {
	case <-stopped:
	case <-time.After(grace):
		server.Stop()
	}
}
