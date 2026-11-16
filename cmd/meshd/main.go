// Command meshd is the TrainMesh coordinator: it admits machines with join tokens, tracks their
// health and runs training jobs across them.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/trainmesh/trainmesh/internal/coordinator"
	"github.com/trainmesh/trainmesh/internal/membership"
)

func main() {
	home, _ := os.UserHomeDir()
	stateDir := flag.String("state-dir", filepath.Join(home, ".trainmesh", "meshd"), "CA, admin certificate and state database")
	listen := flag.String("listen", "0.0.0.0:7443", "address to listen on")
	advertise := flag.String("advertise", "127.0.0.1:7443", "host:port agents and the CLI use to reach meshd (goes into join strings)")
	heartbeat := flag.Duration("heartbeat-interval", coordinator.DefaultHeartbeatInterval, "how often agents heartbeat")
	suspect := flag.Duration("suspect-after", coordinator.DefaultSuspectAfter, "missing heartbeats for this long marks a node DEGRADED")
	lost := flag.Duration("lost-after", coordinator.DefaultLostAfter, "missing heartbeats for this long marks a node LOST and fails its job")
	rendezvousPort := flag.Int("rendezvous-port", coordinator.DefaultRendezvousPort, "TCP port node rank 0 uses for torchrun rendezvous")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	err := coordinator.RunDaemon(ctx, coordinator.DaemonOptions{
		StateDir:      *stateDir,
		ListenAddress: *listen,
		Logger:        logger,
		Config: coordinator.Config{
			AdvertiseAddress:  *advertise,
			HeartbeatInterval: *heartbeat,
			Liveness:          membership.LivenessPolicy{SuspectAfter: *suspect, LostAfter: *lost},
			RendezvousPort:    *rendezvousPort,
		},
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "meshd:", err)
		os.Exit(1)
	}
}
