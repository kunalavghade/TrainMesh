package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
	"github.com/trainmesh/trainmesh/internal/launcher"
)

const probeModule = "trainmesh_runtime.probe"

// memoryReaders returns total physical memory per OS.
var memoryReaders = map[string]func() (uint64, error){
	"darwin": darwinMemory,
	"linux":  linuxMemory,
}

// ProbeCapabilities describes this machine: OS, CPUs, memory, the runtime/PyTorch versions and
// the measured reference-model throughput of each device it offers (device "auto" offers every
// available one, otherwise only that one), all from the Python environment that will train.
func ProbeCapabilities(ctx context.Context, rt launcher.Runtime, device string) (*trainmeshv1.Capabilities, error) {
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}
	caps := &trainmeshv1.Capabilities{
		Hostname:    hostname,
		Os:          runtime.GOOS,
		Arch:        runtime.GOARCH,
		LogicalCpus: int32(min(runtime.NumCPU(), math.MaxInt32)), // #nosec G115 -- clamped
	}
	if read, ok := memoryReaders[runtime.GOOS]; ok {
		if caps.MemoryBytes, err = read(); err != nil {
			return nil, fmt.Errorf("reading memory size: %w", err)
		}
	}

	// #nosec G204 -- discovered interpreter; device is validated by the flag parser
	probe := exec.CommandContext(ctx, rt.Python, "-m", probeModule, "--benchmark", "--device", device)
	probe.Env = launcher.Spec{RuntimeDir: rt.RuntimeDir, ThreadsPerRank: 1}.Env(os.Environ(), runtime.GOOS)
	var stderr strings.Builder
	probe.Stderr = &stderr
	out, err := probe.Output()
	if err != nil {
		return nil, fmt.Errorf("running %s with %s: %w: %s", probeModule, rt.Python, err, strings.TrimSpace(stderr.String()))
	}
	var report struct {
		Torch          string `json:"torch"`
		RuntimeVersion string `json:"runtime_version"`
		Device         string `json:"device"`
		Devices        []struct {
			Kind             string  `json:"kind"`
			SamplesPerSecond float64 `json:"samples_per_second"`
		} `json:"devices"`
	}
	if err := json.Unmarshal(out, &report); err != nil {
		return nil, fmt.Errorf("parsing probe output: %w", err)
	}
	if report.Torch == "" || report.Torch == "missing" {
		return nil, fmt.Errorf("PyTorch is not installed for %s", rt.Python)
	}
	caps.TorchVersion, caps.RuntimeVersion, caps.Device = report.Torch, report.RuntimeVersion, report.Device
	for _, d := range report.Devices {
		caps.Devices = append(caps.Devices, &trainmeshv1.DeviceProfile{Kind: d.Kind, SamplesPerSecond: d.SamplesPerSecond})
	}
	return caps, nil
}

// DetectRoute returns the local IP used to reach the coordinator and the interface that owns it.
// A UDP "connect" sends no packets; it only asks the OS which route it would use.
func DetectRoute(coordinator string) (address, iface string, err error) {
	conn, err := net.Dial("udp", coordinator)
	if err != nil {
		return "", "", fmt.Errorf("finding route to %s: %w", coordinator, err)
	}
	defer func() { _ = conn.Close() }()
	local := conn.LocalAddr().(*net.UDPAddr).IP
	iface, err = interfaceFor(local)
	if err != nil {
		return "", "", err
	}
	return local.String(), iface, nil
}

func interfaceFor(ip net.IP) (string, error) {
	interfaces, err := net.Interfaces()
	if err != nil {
		return "", fmt.Errorf("listing interfaces: %w", err)
	}
	for _, candidate := range interfaces {
		addresses, err := candidate.Addrs()
		if err != nil {
			continue
		}
		for _, address := range addresses {
			if network, ok := address.(*net.IPNet); ok && network.IP.Equal(ip) {
				return candidate.Name, nil
			}
		}
	}
	return "", fmt.Errorf("no network interface owns %s", ip)
}

func darwinMemory() (uint64, error) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
}

func linuxMemory() (uint64, error) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	defer func() { _ = file.Close() }()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) >= 2 && fields[0] == "MemTotal:" {
			kib, err := strconv.ParseUint(fields[1], 10, 64)
			return kib * 1024, err
		}
	}
	return 0, fmt.Errorf("MemTotal not found in /proc/meminfo")
}
