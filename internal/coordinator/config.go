// Package coordinator implements meshd: the gRPC control plane that admits nodes with join
// tokens, tracks their liveness, schedules jobs onto READY nodes and relays results.
package coordinator

import (
	"errors"
	"fmt"
	"time"

	"github.com/trainmesh/trainmesh/internal/membership"
)

// Config holds every coordinator tunable; zero values are replaced by defaults.
type Config struct {
	// Address the coordinator is reachable at, put into join strings (host:port).
	AdvertiseAddress string
	// Heartbeat interval agents are told to use.
	HeartbeatInterval time.Duration
	// Liveness thresholds; must be several heartbeat intervals.
	Liveness membership.LivenessPolicy
	// How often the coordinator re-evaluates liveness.
	SweepInterval time.Duration
	// TCP port node rank 0 uses for torchrun's static rendezvous (island k: + k).
	RendezvousPort int
	// TCP port a diloco island leader's tmcomm listens on (island k: + k).
	TmcommPort int
	// Default join token lifetime and certificate lifetime for nodes.
	TokenTTL         time.Duration
	NodeCertValidity time.Duration
	// Now is the clock; tests inject a fake one.
	Now func() time.Time
}

// Defaults applied by WithDefaults.
const (
	DefaultHeartbeatInterval = 2 * time.Second
	DefaultSuspectAfter      = 6 * time.Second
	DefaultLostAfter         = 20 * time.Second
	DefaultSweepInterval     = time.Second
	DefaultRendezvousPort    = 29500
	DefaultTmcommPort        = 29700
	DefaultTokenTTL          = 15 * time.Minute
	DefaultNodeCertValidity  = 30 * 24 * time.Hour
)

// WithDefaults fills every zero field.
func (c Config) WithDefaults() Config {
	if c.HeartbeatInterval == 0 {
		c.HeartbeatInterval = DefaultHeartbeatInterval
	}
	if c.Liveness.SuspectAfter == 0 {
		c.Liveness.SuspectAfter = DefaultSuspectAfter
	}
	if c.Liveness.LostAfter == 0 {
		c.Liveness.LostAfter = DefaultLostAfter
	}
	if c.SweepInterval == 0 {
		c.SweepInterval = DefaultSweepInterval
	}
	if c.RendezvousPort == 0 {
		c.RendezvousPort = DefaultRendezvousPort
	}
	if c.TmcommPort == 0 {
		c.TmcommPort = DefaultTmcommPort
	}
	if c.TokenTTL == 0 {
		c.TokenTTL = DefaultTokenTTL
	}
	if c.NodeCertValidity == 0 {
		c.NodeCertValidity = DefaultNodeCertValidity
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

// Validate reports every problem at once.
func (c Config) Validate() error {
	var problems []error
	if c.AdvertiseAddress == "" {
		problems = append(problems, errors.New("advertise address is required (host:port agents can reach)"))
	}
	if err := c.Liveness.Validate(); err != nil {
		problems = append(problems, err)
	}
	if c.Liveness.SuspectAfter < 2*c.HeartbeatInterval {
		problems = append(problems, fmt.Errorf("suspect_after (%s) must be at least 2 heartbeat intervals (%s)", c.Liveness.SuspectAfter, c.HeartbeatInterval))
	}
	if c.RendezvousPort <= 0 || c.RendezvousPort > 65535 {
		problems = append(problems, fmt.Errorf("rendezvous port %d out of range", c.RendezvousPort))
	}
	if c.TmcommPort <= 0 || c.TmcommPort > 65535 {
		problems = append(problems, fmt.Errorf("tmcomm port %d out of range", c.TmcommPort))
	}
	return errors.Join(problems...)
}
