package launcher

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestArgs(t *testing.T) {
	single := Spec{ConfigPath: "/c.yaml", LocalRanks: 1, ExtraArgs: []string{"--resume", "/ck"}}
	if got, want := single.Args(), []string{"-m", "trainmesh_runtime", "--config", "/c.yaml", "--resume", "/ck"}; !slices.Equal(got, want) {
		t.Errorf("single-rank Args() = %v, want %v", got, want)
	}

	multi := Spec{ConfigPath: "/c.yaml", LocalRanks: 4, MasterPort: 29500}
	got := strings.Join(multi.Args(), " ")
	for _, want := range []string{"torch.distributed.run", "--nproc_per_node=4", "--master_addr=127.0.0.1", "--master_port=29500", "-m trainmesh_runtime --config /c.yaml"} {
		if !strings.Contains(got, want) {
			t.Errorf("multi-rank Args() = %q, missing %q", got, want)
		}
	}
	if strings.Contains(got, "--standalone") {
		t.Errorf("multi-rank Args() = %q, must not use --standalone (libuv store fails on macOS)", got)
	}
}

func TestEnv(t *testing.T) {
	spec := Spec{RuntimeDir: "/rt", ThreadsPerRank: 6}

	darwin := spec.Env([]string{"PYTHONPATH=/existing"}, "darwin")
	for _, want := range []string{"PYTHONPATH=/rt:/existing", "OMP_NUM_THREADS=6", "USE_LIBUV=0", "GLOO_SOCKET_IFNAME=lo0"} {
		if !slices.Contains(darwin, want) {
			t.Errorf("darwin Env() = %v, missing %q", darwin, want)
		}
	}

	userIface := spec.Env([]string{"GLOO_SOCKET_IFNAME=en0"}, "darwin")
	if slices.Contains(userIface, "GLOO_SOCKET_IFNAME=lo0") {
		t.Errorf("Env() overrode user GLOO_SOCKET_IFNAME: %v", userIface)
	}

	linux := spec.Env(nil, "linux")
	if lookupEnv(linux, "GLOO_SOCKET_IFNAME") != "" {
		t.Errorf("linux Env() should not pin an interface: %v", linux)
	}
	if got := lookupEnv(spec.Env([]string{"PYTHONPATH=a"}, "windows"), "PYTHONPATH"); got != "/rt;a" {
		t.Errorf("windows PYTHONPATH = %q, want /rt;a", got)
	}
}

func TestResolveThreadsPerRank(t *testing.T) {
	tests := []struct {
		name       string
		configured int
		ompEnv     string
		want       int
	}{
		{"explicit config wins", 3, "8", 3},
		{"user OMP env next", 0, "8", 8},
		{"default when unset", 0, "", DefaultThreadsPerRank},
		{"invalid env ignored", 0, "abc", DefaultThreadsPerRank},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ResolveThreadsPerRank(tt.configured, tt.ompEnv); got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestConflictDetectorFindsMarkerAcrossWrites(t *testing.T) {
	var out bytes.Buffer
	detector := &conflictDetector{target: &out, marker: portConflictMarker}

	_, _ = detector.Write([]byte("server socket failed: Address already "))
	_, _ = detector.Write([]byte("in use\n"))

	if !detector.seen {
		t.Error("marker split across writes was not detected")
	}
	if out.String() != "server socket failed: Address already in use\n" {
		t.Errorf("output not forwarded unchanged: %q", out.String())
	}
}

func TestDiscoverRuntime(t *testing.T) {
	root := t.TempDir()
	pkg := filepath.Join(root, "python", runtimeModule)
	mustWrite(t, filepath.Join(pkg, "__init__.py"))
	mustWrite(t, filepath.Join(root, ".venv", "bin", "python"))
	nested := filepath.Join(root, "examples", "deep")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	noEnv := func(string) string { return "" }

	rt, err := DiscoverRuntime(noEnv, nested, "")

	if err != nil {
		t.Fatalf("DiscoverRuntime() error = %v", err)
	}
	if rt.RuntimeDir != filepath.Join(root, "python") || rt.Python != filepath.Join(root, ".venv", "bin", "python") {
		t.Errorf("DiscoverRuntime() = %+v", rt)
	}

	override := func(key string) string {
		return map[string]string{EnvPython: "/custom/python", EnvRuntimeDir: "/custom/rt"}[key]
	}
	if rt, err := DiscoverRuntime(override, t.TempDir(), ""); err != nil || rt != (Runtime{Python: "/custom/python", RuntimeDir: "/custom/rt"}) {
		t.Errorf("env overrides: got %+v, %v", rt, err)
	}
}

func mustWrite(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestMultiNodeArgsAndEnv(t *testing.T) {
	spec := Spec{
		ConfigPath: "/c.yaml", RuntimeDir: "/rt", ThreadsPerRank: 1, LocalRanks: 2,
		Nodes: 3, NodeRank: 1, MasterAddr: "10.0.0.5", MasterPort: 29500, Interface: "en0",
	}

	args := strings.Join(spec.Args(), " ")
	for _, want := range []string{"--nnodes=3", "--node_rank=1", "--nproc_per_node=2", "--master_addr=10.0.0.5", "--master_port=29500"} {
		if !strings.Contains(args, want) {
			t.Errorf("Args() = %q, missing %q", args, want)
		}
	}
	if got := lookupEnv(spec.Env([]string{"GLOO_SOCKET_IFNAME=lo0"}, "darwin"), "GLOO_SOCKET_IFNAME"); got != "en0" {
		t.Errorf("GLOO_SOCKET_IFNAME = %q, want the assigned interface en0", got)
	}
	if got := lookupEnv(Spec{Nodes: 2}.Env(nil, "darwin"), "GLOO_SOCKET_IFNAME"); got != "" {
		t.Errorf("multi-node job pinned to loopback: %q", got)
	}
	if !strings.Contains(strings.Join(Spec{LocalRanks: 1, Nodes: 2}.Args(), " "), "torch.distributed.run") {
		t.Error("one rank per machine must still use torchrun for a multi-machine job")
	}
}
