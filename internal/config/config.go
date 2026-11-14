package config

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Version   int            `yaml:"version" json:"version"`
	Run       Run            `yaml:"run" json:"run"`
	Execution Execution      `yaml:"execution" json:"execution"`
	Training  Training       `yaml:"training" json:"training"`
	Optimizer map[string]any `yaml:"optimizer" json:"optimizer"`
}
type Run struct {
	Name   string `yaml:"name" json:"name"`
	Seed   int    `yaml:"seed" json:"seed"`
	Resume string `yaml:"resume,omitempty" json:"resume,omitempty"`
}
type Execution struct {
	Strategy           string  `yaml:"strategy" json:"strategy"`
	Workers            int     `yaml:"workers" json:"workers"`
	ProcessesPerWorker int     `yaml:"processes_per_worker" json:"processes_per_worker"`
	Backend            Backend `yaml:"backend" json:"backend"`
}
type Backend struct {
	Device        string `yaml:"device" json:"device"`
	Communication string `yaml:"communication" json:"communication"`
}
type Training struct {
	Dataset    Dataset    `yaml:"dataset" json:"dataset"`
	Batch      Batch      `yaml:"batch" json:"batch"`
	Epochs     int        `yaml:"epochs" json:"epochs"`
	Precision  string     `yaml:"precision" json:"precision"`
	Checkpoint Checkpoint `yaml:"checkpoint" json:"checkpoint"`
}
type Dataset struct {
	Type    string `yaml:"type" json:"type"`
	Samples int    `yaml:"samples" json:"samples"`
}
type Batch struct {
	PerRank                   int `yaml:"per_rank" json:"per_rank"`
	GradientAccumulationSteps int `yaml:"gradient_accumulation_steps" json:"gradient_accumulation_steps"`
}
type Checkpoint struct {
	Enabled   bool   `yaml:"enabled" json:"enabled"`
	Directory string `yaml:"directory" json:"directory"`
}

func Load(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err := yaml.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("parse config: %w", err)
	}
	return c, c.Validate()
}

func (c Config) Validate() error {
	if c.Version != 1 {
		return errors.New("version must be 1")
	}
	if c.Execution.Strategy != "data_parallel" {
		return errors.New("execution.strategy must be data_parallel")
	}
	if c.Execution.Workers < 1 || c.Execution.ProcessesPerWorker < 1 {
		return errors.New("execution.workers and processes_per_worker must be positive")
	}
	if strings.ToLower(c.Execution.Backend.Device) != "cpu" || strings.ToLower(c.Execution.Backend.Communication) != "gloo" {
		return errors.New("this release supports device=cpu and communication=gloo only")
	}
	if c.Training.Dataset.Type != "synthetic" {
		return errors.New("only training.dataset.type=synthetic is supported")
	}
	if c.Training.Dataset.Samples < 1 || c.Training.Epochs < 1 {
		return errors.New("training dataset samples and epochs must be positive")
	}
	if c.Training.Batch.PerRank < 1 || c.Training.Batch.GradientAccumulationSteps < 1 {
		return errors.New("per-rank batch size and accumulation steps must be positive")
	}
	if c.Training.Precision != "fp32" {
		return errors.New("only fp32 precision is supported")
	}
	return nil
}

func (c Config) WorldSize() int { return c.Execution.Workers * c.Execution.ProcessesPerWorker }
func (c Config) GlobalBatchSize() int {
	return c.Training.Batch.PerRank * c.WorldSize() * c.Training.Batch.GradientAccumulationSteps
}
