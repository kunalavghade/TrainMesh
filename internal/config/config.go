// Package config loads and validates the v1 TrainMesh job configuration.
package config

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config is the subset of the v1 job configuration the Go CLI needs.
type Config struct {
	Version   int            `yaml:"version" json:"version"`
	Run       Run            `yaml:"run" json:"run"`
	Execution Execution      `yaml:"execution" json:"execution"`
	Training  Training       `yaml:"training" json:"training"`
	Planner   map[string]any `yaml:"planner" json:"planner,omitempty"`
	// LegacyOptimizer is the deprecated name of Planner, still accepted on load.
	LegacyOptimizer map[string]any `yaml:"optimizer" json:"-"`
}

// Run holds run identity and output locations.
type Run struct {
	Name        string `yaml:"name" json:"name"`
	Seed        int    `yaml:"seed" json:"seed"`
	Resume      string `yaml:"resume,omitempty" json:"resume,omitempty"`
	ResultsPath string `yaml:"results_path,omitempty" json:"results_path,omitempty"`
}

// Execution describes how many processes run and on which backend.
type Execution struct {
	Strategy           string `yaml:"strategy" json:"strategy"`
	Workers            int    `yaml:"workers" json:"workers"`
	ProcessesPerWorker int    `yaml:"processes_per_worker" json:"processes_per_worker"`
	// CPUThreadsPerRank pins intra-op threads per rank; 0 means auto.
	CPUThreadsPerRank int     `yaml:"cpu_threads_per_rank" json:"cpu_threads_per_rank"`
	Backend           Backend `yaml:"backend" json:"backend"`
}

// Backend selects the compute device and collective library.
type Backend struct {
	Device        string `yaml:"device" json:"device"`
	Communication string `yaml:"communication" json:"communication"`
}

// Training holds the training parameters validated on the Go side.
type Training struct {
	Dataset    Dataset    `yaml:"dataset" json:"dataset"`
	Batch      Batch      `yaml:"batch" json:"batch"`
	Epochs     int        `yaml:"epochs" json:"epochs"`
	Precision  string     `yaml:"precision" json:"precision"`
	Checkpoint Checkpoint `yaml:"checkpoint" json:"checkpoint"`
}

// Dataset describes the training data source.
type Dataset struct {
	Type    string `yaml:"type" json:"type"`
	Samples int    `yaml:"samples" json:"samples"`
}

// Batch describes per-rank batch size and gradient accumulation.
type Batch struct {
	PerRank                   int `yaml:"per_rank" json:"per_rank"`
	GradientAccumulationSteps int `yaml:"gradient_accumulation_steps" json:"gradient_accumulation_steps"`
}

// Checkpoint controls checkpoint writing.
type Checkpoint struct {
	Enabled   bool   `yaml:"enabled" json:"enabled"`
	Directory string `yaml:"directory" json:"directory"`
}

// Load reads, migrates and validates the configuration at path.
func Load(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c, fmt.Errorf("reading config %s: %w", path, err)
	}
	if err := yaml.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("parsing config %s: %w", path, err)
	}
	c.migrateLegacyKeys()
	return c, c.Validate()
}

// UsesLegacyOptimizerKey reports whether the deprecated top-level `optimizer` key was present.
func (c Config) UsesLegacyOptimizerKey() bool { return c.LegacyOptimizer != nil }

func (c *Config) migrateLegacyKeys() {
	if c.LegacyOptimizer == nil || c.Planner != nil {
		return
	}
	c.Planner = make(map[string]any, len(c.LegacyOptimizer))
	for key, value := range c.LegacyOptimizer {
		if key == "results" {
			if results, ok := value.(string); ok && c.Run.ResultsPath == "" {
				c.Run.ResultsPath = results
			}
			continue
		}
		c.Planner[key] = value
	}
}

// Validate returns every validation problem at once, joined.
func (c Config) Validate() error {
	var problems []error
	add := func(format string, args ...any) { problems = append(problems, fmt.Errorf(format, args...)) }

	if c.Version != 1 {
		add("version must be 1")
	}
	if c.Execution.Strategy != "data_parallel" {
		add("execution.strategy must be data_parallel")
	}
	if c.Execution.Workers < 1 || c.Execution.ProcessesPerWorker < 1 {
		add("execution.workers and processes_per_worker must be positive")
	}
	if c.Execution.CPUThreadsPerRank < 0 {
		add("execution.cpu_threads_per_rank must be 0 (auto) or positive")
	}
	if strings.ToLower(c.Execution.Backend.Device) != "cpu" || strings.ToLower(c.Execution.Backend.Communication) != "gloo" {
		add("this release supports device=cpu and communication=gloo only")
	}
	if c.Training.Dataset.Type != "synthetic" {
		add("only training.dataset.type=synthetic is supported")
	}
	if c.Training.Dataset.Samples < 1 || c.Training.Epochs < 1 {
		add("training dataset samples and epochs must be positive")
	}
	if c.Training.Batch.PerRank < 1 || c.Training.Batch.GradientAccumulationSteps < 1 {
		add("per-rank batch size and accumulation steps must be positive")
	}
	if c.Training.Precision != "fp32" {
		add("only fp32 precision is supported")
	}
	return errors.Join(problems...)
}

// WorldSize is the total number of training ranks.
func (c Config) WorldSize() int { return c.Execution.Workers * c.Execution.ProcessesPerWorker }

// GlobalBatchSize is samples per optimizer step across all ranks.
func (c Config) GlobalBatchSize() int {
	return c.Training.Batch.PerRank * c.WorldSize() * c.Training.Batch.GradientAccumulationSteps
}
