package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

func lookup(env []string, key string) (string, bool) {
	for i := len(env) - 1; i >= 0; i-- {
		if value, ok := strings.CutPrefix(env[i], key+"="); ok {
			return value, true
		}
	}
	return "", false
}

func TestIslandLeaderGetsItsLayoutAndIdentity(t *testing.T) {
	runner := TorchRunner{Device: "mps", Address: "10.0.0.7", TLS: TLSFiles{Cert: "/s/node.pem", Key: "/s/node-key.pem", CA: "/s/ca.pem"}}
	peers := []IslandPeer{{Island: 0, Address: "10.0.0.1:29700", Name: "node-a"}, {Island: 1, Address: "10.0.0.7:29701", Name: "node-b"}}
	env, err := runner.environment(Assignment{Island: 1, IslandPeers: peers, TmcommPort: 29701})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		deviceEnv:    "mps",
		islandEnv:    "1",
		tmcommListen: "10.0.0.7:29701",
		tlsCertEnv:   "/s/node.pem",
		tlsKeyEnv:    "/s/node-key.pem",
		tlsCAEnv:     "/s/ca.pem",
	}
	for key, value := range want {
		if got, _ := lookup(env, key); got != value {
			t.Errorf("%s = %q, want %q", key, got, value)
		}
	}
	raw, _ := lookup(env, islandPeersEnv)
	var decoded []IslandPeer
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil || len(decoded) != 2 || decoded[1] != peers[1] {
		t.Errorf("peers = %s (%v)", raw, err)
	}
}

func TestDataParallelAssignmentHasNoIslandEnvironment(t *testing.T) {
	env, err := TorchRunner{}.environment(Assignment{})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{islandEnv, islandPeersEnv, tmcommListen, tlsCertEnv, deviceEnv} {
		if _, ok := lookup(env, key); ok {
			t.Errorf("%s set for a data_parallel job", key)
		}
	}
}
