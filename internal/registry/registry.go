package registry

import (
	"context"
	"fmt"
	"sort"
)

// RunOptions configures a benchmark run.
type RunOptions struct {
	DataPath string
	Model    string
	Limit    int
	Seed     int64
	FitPath  string             // optional training dataset path
	ModelIn  string             // optional load model from JSON
	ModelOut string             // optional save learned model to JSON
	Params   map[string]float64 // optional hyperparameters (alpha, w_lo, w_vec, vec_dim)
}

// UsageError indicates the caller provided invalid inputs for a runner.
type UsageError struct {
	Msg  string
	Hint string
}

func (e *UsageError) Error() string { return e.Msg }

// Summary is a generic run summary with task-specific metrics.
type Summary struct {
	Task       string             `json:"task"`
	Model      string             `json:"model"`
	Count      int                `json:"count"`
	Metrics    map[string]float64 `json:"metrics"`
	StartedAt  int64              `json:"started_at_unix"`
	FinishedAt int64              `json:"finished_at_unix"`
	// Optional details; structure may vary per task but stays JSON-safe.
	Details any `json:"details,omitempty"`
}

// Runner runs a specific benchmark task.
type Runner interface {
	ID() string
	Description() string
	DefaultMetric() string
	Run(ctx context.Context, opts RunOptions) (*Summary, error)
}

var (
	runners = map[string]Runner{}
)

// Register adds a runner to the global registry.
func Register(r Runner) {
	runners[r.ID()] = r
}

// Lookup returns a runner by id.
func Lookup(id string) Runner { return runners[id] }

// All returns all runners in stable order by id.
func All() []Runner {
	ids := make([]string, 0, len(runners))
	for id := range runners {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]Runner, 0, len(ids))
	for _, id := range ids {
		out = append(out, runners[id])
	}
	return out
}

// Errf is a helper to create UsageErrors with printf-style formatting.
func Errf(hint, format string, args ...any) *UsageError {
	return &UsageError{Msg: fmt.Sprintf(format, args...), Hint: hint}
}
