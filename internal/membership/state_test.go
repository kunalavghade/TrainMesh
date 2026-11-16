package membership

import (
	"errors"
	"testing"
	"time"
)

var allStates = []State{StateJoining, StateProfiling, StateReady, StateBusy, StateDegraded, StateLost, StateDraining}

func TestTransitionTable(t *testing.T) {
	legal := map[[2]State]bool{
		{StateJoining, StateProfiling}:  true,
		{StateProfiling, StateReady}:    true,
		{StateProfiling, StateDegraded}: true,
		{StateReady, StateBusy}:         true,
		{StateReady, StateDegraded}:     true,
		{StateReady, StateDraining}:     true,
		{StateReady, StateProfiling}:    true,
		{StateBusy, StateReady}:         true,
		{StateBusy, StateDegraded}:      true,
		{StateBusy, StateDraining}:      true,
		{StateDegraded, StateReady}:     true,
		{StateDegraded, StateBusy}:      true,
		{StateDegraded, StateLost}:      true,
		{StateLost, StateProfiling}:     true,
	}
	for _, from := range allStates {
		for _, to := range allStates {
			got, err := Transition(from, to)
			if legal[[2]State{from, to}] {
				if err != nil || got != to {
					t.Errorf("%s → %s: got (%s, %v), want allowed", from, to, got, err)
				}
				continue
			}
			if !errors.Is(err, ErrIllegalTransition) || got != from {
				t.Errorf("%s → %s: got (%s, %v), want ErrIllegalTransition", from, to, got, err)
			}
		}
	}
}

func TestOnlyReadyIsAssignable(t *testing.T) {
	for _, s := range allStates {
		if Assignable(s) != (s == StateReady) {
			t.Errorf("Assignable(%s) = %v", s, Assignable(s))
		}
	}
}

func TestLivenessPolicyValidate(t *testing.T) {
	if err := (LivenessPolicy{SuspectAfter: 6 * time.Second, LostAfter: 20 * time.Second}).Validate(); err != nil {
		t.Errorf("valid policy rejected: %v", err)
	}
	for _, bad := range []LivenessPolicy{{}, {SuspectAfter: time.Second, LostAfter: time.Second}} {
		if bad.Validate() == nil {
			t.Errorf("policy %+v accepted, want error", bad)
		}
	}
}

func TestLivenessNext(t *testing.T) {
	policy := LivenessPolicy{SuspectAfter: 6 * time.Second, LostAfter: 20 * time.Second}
	now := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name    string
		current State
		age     time.Duration
		want    State
		changed bool
	}{
		{"fresh ready stays", StateReady, time.Second, StateReady, false},
		{"late ready degrades", StateReady, 6 * time.Second, StateDegraded, true},
		{"late busy degrades", StateBusy, 7 * time.Second, StateDegraded, true},
		{"late profiling degrades", StateProfiling, 7 * time.Second, StateDegraded, true},
		{"degraded recovers", StateDegraded, time.Second, StateReady, true},
		{"degraded waits", StateDegraded, 10 * time.Second, StateDegraded, false},
		{"degraded is lost", StateDegraded, 20 * time.Second, StateLost, true},
		{"lost stays lost", StateLost, time.Hour, StateLost, false},
		{"draining unaffected", StateDraining, time.Hour, StateDraining, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, changed := policy.Next(tt.current, now.Add(-tt.age), now)
			if got != tt.want || changed != tt.changed {
				t.Fatalf("Next(%s, age %s) = (%s, %v), want (%s, %v)", tt.current, tt.age, got, changed, tt.want, tt.changed)
			}
			if changed {
				if _, err := Transition(tt.current, got); err != nil {
					t.Fatalf("liveness produced an illegal transition: %v", err)
				}
			}
		})
	}
}
