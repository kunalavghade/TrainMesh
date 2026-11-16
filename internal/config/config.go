// Package config loads a TrainMesh job config (YAML, version 1) into the generated JobSpec and
// checks it against the rules declared in proto/trainmesh/v1/jobspec.proto.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"buf.build/go/protovalidate"
	"google.golang.org/protobuf/encoding/protojson"
	"gopkg.in/yaml.v3"

	trainmeshv1 "github.com/trainmesh/trainmesh/internal/gen/trainmesh/v1"
)

const (
	legacyPlannerKey  = "optimizer"
	plannerKey        = "planner"
	legacyResultsKey  = "results"
	defaultAccumSteps = 1
)

// ErrInvalid wraps every config problem so callers can branch with errors.Is.
var ErrInvalid = errors.New("invalid job config")

// Config is a validated JobSpec plus facts about how it was loaded.
type Config struct {
	Spec *trainmeshv1.JobSpec
	// usesLegacyOptimizerKey records that the deprecated top-level `optimizer:` key was migrated.
	usesLegacyOptimizerKey bool
}

// Load reads, migrates, parses and validates the config at path.
func Load(path string) (Config, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- operator-supplied config path
	if err != nil {
		return Config{}, fmt.Errorf("reading config %s: %w", path, err)
	}
	c, err := Parse(raw)
	if err != nil {
		return Config{}, fmt.Errorf("config %s: %w", path, err)
	}
	return c, nil
}

// Parse migrates legacy keys, decodes YAML into a JobSpec (unknown keys are errors) and validates it.
func Parse(raw []byte) (Config, error) {
	var document map[string]any
	if err := yaml.Unmarshal(raw, &document); err != nil {
		return Config{}, fmt.Errorf("%w: parsing YAML: %w", ErrInvalid, err)
	}
	if document == nil {
		return Config{}, fmt.Errorf("%w: config is empty", ErrInvalid)
	}
	usedLegacyKey := migrateLegacyKeys(document)

	encoded, err := json.Marshal(document)
	if err != nil {
		return Config{}, fmt.Errorf("%w: re-encoding YAML: %w", ErrInvalid, err)
	}
	spec := &trainmeshv1.JobSpec{}
	if err := protojson.Unmarshal(encoded, spec); err != nil {
		return Config{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	if err := Validate(spec); err != nil {
		return Config{}, err
	}
	return Config{Spec: spec, usesLegacyOptimizerKey: usedLegacyKey}, nil
}

// Validate checks spec against the buf.validate rules in jobspec.proto, reporting every problem.
func Validate(spec *trainmeshv1.JobSpec) error {
	err := protovalidate.Validate(spec)
	var validationErr *protovalidate.ValidationError
	if !errors.As(err, &validationErr) {
		if err != nil {
			return fmt.Errorf("%w: %w", ErrInvalid, err)
		}
		return nil
	}
	problems := make([]string, 0, len(validationErr.Violations))
	for _, violation := range validationErr.Violations {
		path := protovalidate.FieldPathString(violation.Proto.GetField())
		problems = append(problems, fmt.Sprintf("%s: %s", path, violation.Proto.GetMessage()))
	}
	return fmt.Errorf("%w:\n  %s", ErrInvalid, strings.Join(problems, "\n  "))
}

// UsesLegacyOptimizerKey reports whether the deprecated top-level `optimizer` key was present.
func (c Config) UsesLegacyOptimizerKey() bool { return c.usesLegacyOptimizerKey }

// WorldSize is the total number of training ranks.
func (c Config) WorldSize() int {
	execution := c.Spec.GetExecution()
	return int(execution.GetWorkers()) * int(execution.GetProcessesPerWorker())
}

// AccumulationSteps is the gradient accumulation window, defaulting to 1.
func (c Config) AccumulationSteps() int {
	if steps := int(c.Spec.GetTraining().GetBatch().GetGradientAccumulationSteps()); steps > 0 {
		return steps
	}
	return defaultAccumSteps
}

// GlobalBatchSize is samples per optimizer step across all ranks.
func (c Config) GlobalBatchSize() int {
	return int(c.Spec.GetTraining().GetBatch().GetPerRank()) * c.WorldSize() * c.AccumulationSteps()
}

// migrateLegacyKeys moves the deprecated top-level `optimizer:` block to `planner:` and its
// `results` entry to run.results_path. It reports whether the legacy key was present.
func migrateLegacyKeys(document map[string]any) bool {
	legacy, found := document[legacyPlannerKey].(map[string]any)
	if !found {
		return false
	}
	delete(document, legacyPlannerKey)

	if results, ok := legacy[legacyResultsKey].(string); ok && results != "" {
		run, _ := document["run"].(map[string]any)
		if run == nil {
			run = map[string]any{}
			document["run"] = run
		}
		if _, set := run["results_path"]; !set {
			run["results_path"] = results
		}
	}
	delete(legacy, legacyResultsKey)
	if _, set := document[plannerKey]; !set {
		document[plannerKey] = legacy
	}
	return true
}
