// internal/lnn/tune.go - LNN-enhanced tuning
package lnn

import (
	"context"
	"fmt"
	"log"

	"ai_benchmarks/internal/registry"
)

// LNNTuner handles LNN-based recursive tuning
type LNNTuner struct {
	lnn     Calibrator
	task    registry.Runner
	history []TuningStep
	config  LNNTuneConfig
}

type TuningStep struct {
	Params  map[string]float64
	Metrics map[string]float64
}

type LNNTuneConfig struct {
	LearningRate  float64
	MaxIterations int
	StateFile     string
	UseRecursive  bool
}

func NewLNNTuner(task registry.Runner, cfg LNNTuneConfig) (*LNNTuner, error) {
	lnn, err := LookupCalibrator(task.ID())
	if err != nil {
		return nil, err
	}

	tuner := &LNNTuner{
		lnn:    lnn,
		task:   task,
		config: cfg,
	}

	// Load previous state if exists
	if cfg.StateFile != "" {
		if err := lnn.Load(cfg.StateFile); err != nil {
			log.Printf("warn: failed to load LNN state %q: %v", cfg.StateFile, err)
		}
	}

	return tuner, nil
}

// Tune performs recursive LNN-based tuning
func (t *LNNTuner) Tune() (map[string]float64, error) {
	var bestParams map[string]float64
	bestScore := -1.0

	for iter := 0; iter < t.config.MaxIterations; iter++ {
		// Generate parameters using LNN with context
		output, err := t.lnn.Generate(t.task.ID(), t.getPreviousPerformance())
		if err != nil {
			return nil, fmt.Errorf("LNN parameter generation failed: %v", err)
		}
		params := output.Params

		// Evaluate with generated parameters
		metrics, err := t.evaluateWithParams(params)
		if err != nil {
			log.Printf("Evaluation failed: %v", err)
			continue
		}

		score := metrics["accuracy"] // or relevant metric

		// Update LNN based on performance feedback
		if t.config.UseRecursive {
			if err := t.lnn.UpdateFromFeedback(params, score); err != nil {
				log.Printf("LNN update failed: %v", err)
			}
		}

		// Record history
		step := TuningStep{
			Params:  params,
			Metrics: metrics,
		}
		t.history = append(t.history, step)

		// Update best
		if score > bestScore {
			bestScore = score
			bestParams = params
		}

		fmt.Printf("Iteration %d: Score=%.4f, Params=%v\n", iter, score, params)
	}

	// Save LNN state for future recursive learning
	if t.config.StateFile != "" {
		if err := t.lnn.Save(t.config.StateFile); err != nil {
			log.Printf("warn: failed to save LNN state %q: %v", t.config.StateFile, err)
		}
	}

	return bestParams, nil
}

func (t *LNNTuner) getPreviousPerformance() map[string]float64 {
	if len(t.history) == 0 {
		return make(map[string]float64)
	}

	lastStep := t.history[len(t.history)-1]
	return lastStep.Metrics
}

func (t *LNNTuner) evaluateWithParams(params map[string]float64) (map[string]float64, error) {
	// Run evaluation using the existing benchmark infrastructure
	opts := registry.RunOptions{
		Model:  "hybrid", // Assuming LNN always uses the hybrid model
		Params: params,
	}
	summary, err := t.task.Run(context.Background(), opts)
	if err != nil {
		return nil, err
	}

	return summary.Metrics, nil
}
