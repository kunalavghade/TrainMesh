package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/trainmesh/trainmesh/internal/cli"
)

func main() {
	// Ranks run in their own process group, so Ctrl-C must be forwarded via context cancellation.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err := cli.NewRoot().ExecuteContext(ctx)
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, "trainmesh:", err)
		os.Exit(1)
	}
}
