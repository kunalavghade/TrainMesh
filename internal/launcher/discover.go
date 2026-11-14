package launcher

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// Env variables that override runtime discovery.
const (
	EnvPython     = "TRAINMESH_PYTHON"
	EnvRuntimeDir = "TRAINMESH_RUNTIME_DIR"
)

// ErrRuntimeNotFound means no directory containing the trainmesh_runtime package was found.
var ErrRuntimeNotFound = errors.New("trainmesh_runtime package not found")

// Runtime locates the interpreter and the Python package used for training.
type Runtime struct {
	Python     string
	RuntimeDir string
}

// DiscoverRuntime resolves the runtime from env overrides, then from cwd and executable ancestors.
func DiscoverRuntime(getenv func(string) string, cwd, executableDir string) (Runtime, error) {
	root := findProjectRoot(cwd, executableDir)
	runtimeDir := getenv(EnvRuntimeDir)
	if runtimeDir == "" && root != "" {
		runtimeDir = filepath.Join(root, "python")
	}
	if runtimeDir == "" {
		return Runtime{}, fmt.Errorf("%w: run from a TrainMesh checkout or set %s", ErrRuntimeNotFound, EnvRuntimeDir)
	}
	python, err := ResolvePython(getenv, root)
	if err != nil {
		return Runtime{}, err
	}
	return Runtime{Python: python, RuntimeDir: runtimeDir}, nil
}

// ResolvePython returns TRAINMESH_PYTHON, the project virtualenv interpreter, or python3 on PATH.
func ResolvePython(getenv func(string) string, root string) (string, error) {
	if p := getenv(EnvPython); p != "" {
		return p, nil
	}
	if root != "" {
		for _, candidate := range venvInterpreters(root) {
			if isFile(candidate) {
				return candidate, nil
			}
		}
	}
	p, err := exec.LookPath("python3")
	if err != nil {
		return "", fmt.Errorf("python3 not found: create .venv (see docs/CPU_BACKEND.md) or set %s: %w", EnvPython, err)
	}
	return p, nil
}

func findProjectRoot(starts ...string) string {
	for _, start := range starts {
		if start == "" {
			continue
		}
		for dir := filepath.Clean(start); ; dir = filepath.Dir(dir) {
			if isFile(filepath.Join(dir, "python", runtimeModule, "__init__.py")) {
				return dir
			}
			if filepath.Dir(dir) == dir {
				break
			}
		}
	}
	return ""
}

func venvInterpreters(root string) []string {
	return []string{
		filepath.Join(root, ".venv", "bin", "python"),
		filepath.Join(root, ".venv", "Scripts", "python.exe"),
	}
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
