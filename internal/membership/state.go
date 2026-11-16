// Package membership holds the node lifecycle rules: which states exist, which transitions are
// legal, and how heartbeat age moves a node between healthy, degraded and lost. It is pure: no
// IO, no clocks — callers pass the time in.
package membership

import (
	"errors"
	"fmt"
	"time"
)

// State is a node's lifecycle state.
type State string

// Node lifecycle states.
const (
	StateJoining   State = "JOINING"   // token redeemed, identity being issued
	StateProfiling State = "PROFILING" // waiting for the first capability report
	StateReady     State = "READY"     // healthy and idle; can be assigned work
	StateBusy      State = "BUSY"      // running a job assignment
	StateDegraded  State = "DEGRADED"  // heartbeats are late; not assignable
	StateLost      State = "LOST"      // heartbeats stopped for too long; its jobs are failed
	StateDraining  State = "DRAINING"  // operator removal requested; finishes then leaves
)

// ErrIllegalTransition is returned for a transition the lifecycle does not allow.
var ErrIllegalTransition = errors.New("illegal node state transition")

var allowedTransitions = map[State][]State{
	StateJoining:   {StateProfiling},
	StateProfiling: {StateReady, StateDegraded},
	StateReady:     {StateBusy, StateDegraded, StateDraining, StateProfiling},
	StateBusy:      {StateReady, StateDegraded, StateDraining},
	StateDegraded:  {StateReady, StateBusy, StateLost},
	StateLost:      {StateProfiling},
	StateDraining:  {},
}

// CanTransition reports whether from → to is allowed.
func CanTransition(from, to State) bool {
	for _, allowed := range allowedTransitions[from] {
		if allowed == to {
			return true
		}
	}
	return false
}

// Transition returns to, or ErrIllegalTransition naming both states.
func Transition(from, to State) (State, error) {
	if !CanTransition(from, to) {
		return from, fmt.Errorf("%w: %s → %s", ErrIllegalTransition, from, to)
	}
	return to, nil
}

// Assignable reports whether a node in this state may receive a new job.
func Assignable(s State) bool { return s == StateReady }

// LivenessPolicy decides how heartbeat age changes a node's state.
type LivenessPolicy struct {
	SuspectAfter time.Duration // no heartbeat for this long → DEGRADED
	LostAfter    time.Duration // no heartbeat for this long → LOST
}

// Validate requires 0 < SuspectAfter < LostAfter.
func (p LivenessPolicy) Validate() error {
	if p.SuspectAfter <= 0 || p.LostAfter <= p.SuspectAfter {
		return fmt.Errorf("liveness policy needs 0 < suspect_after (%s) < lost_after (%s)", p.SuspectAfter, p.LostAfter)
	}
	return nil
}

// Next returns the state a node should move to given its last heartbeat, and whether it changes.
func (p LivenessPolicy) Next(current State, lastHeartbeat, now time.Time) (State, bool) {
	age := now.Sub(lastHeartbeat)
	switch current {
	case StateReady, StateBusy, StateProfiling:
		if age >= p.SuspectAfter {
			return StateDegraded, true
		}
	case StateDegraded:
		if age >= p.LostAfter {
			return StateLost, true
		}
		if age < p.SuspectAfter {
			return StateReady, true
		}
	}
	return current, false
}
