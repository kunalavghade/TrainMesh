// Command tm-agent joins this machine to a TrainMesh cluster and runs the training work the
// coordinator assigns to it.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/trainmesh/trainmesh/internal/agent"
	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/launcher"
)

func main() {
	home, _ := os.UserHomeDir()
	stateDir := flag.String("state-dir", filepath.Join(home, ".trainmesh", "agent"), "node identity and job directories")
	join := flag.String("join", "", "join string from `trainmesh token create` (first start only)")
	address := flag.String("address", "", "IP other machines use to reach this one (default: route to the coordinator)")
	iface := flag.String("interface", "", "network interface for training traffic (default: the one owning --address)")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(*stateDir, *join, *address, *iface, logger); err != nil {
		fmt.Fprintln(os.Stderr, "tm-agent:", err)
		os.Exit(1)
	}
}

func run(stateDir, join, address, iface string, logger *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	executable, _ := os.Executable()
	rt, err := launcher.DiscoverRuntime(os.Getenv, cwd, filepath.Dir(executable))
	if err != nil {
		return err
	}
	runner := &agent.TorchRunner{Runtime: rt, JobsDir: filepath.Join(stateDir, "jobs")}
	a, err := agent.New(agent.Config{
		StateDir:   stateDir,
		JoinString: join,
		Address:    address,
		Interface:  iface,
		Capabilities: func(ctx context.Context) (*trainmeshv1.Capabilities, error) {
			return agent.ProbeCapabilities(ctx, rt)
		},
		Runner: runner,
		Logger: logger,
	})
	if err != nil {
		return err
	}
	if err := a.Start(ctx); err != nil {
		return err
	}
	runner.Interface = a.Interface()
	err = a.Run(ctx)
	if errors.Is(err, context.Canceled) {
		return nil
	}
	return err
}
