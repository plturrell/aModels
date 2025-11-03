// internal/lnn/nash.go - LNN-enhanced Nash calibration
package lnn

import (
	"context"
	"fmt"
	"log"

	"ai_benchmarks/internal/registry"
)

// LNNNashCalibrator handles LNN-based multi-task calibration
type LNNNashCalibrator struct {
	tasks   []registry.Runner
	history map[string][]TuningStep // Task ID -> History
	config  LNNTuneConfig
}

// NewLNNNashCalibrator creates a new LNN-based Nash calibrator
func NewLNNNashCalibrator(tasks []registry.Runner, cfg LNNTuneConfig) (*LNNNashCalibrator, error) {
	// The calibrator itself doesn't need a central LNN, just the config to create temporary ones.
	calibrator := &LNNNashCalibrator{
		tasks:   tasks,
		history: make(map[string][]TuningStep),
		config:  cfg,
	}
	return calibrator, nil
}

// Calibrate uses a collection of specialist LNNs to find a globally optimal static parameter set.
func (nc *LNNNashCalibrator) Calibrate() (map[string]float64, error) {
	// Step 1: Load each specialist LNN and generate a set of candidate parameters.
	candidateSets := make(map[string]map[string]float64) // Task ID -> Candidate Params
	for _, task := range nc.tasks {
		lnnStateFile := task.ID() + ".lnn.state"
		calibrator, err := LookupCalibrator(task.ID())
		if err != nil {
			return nil, fmt.Errorf("failed to create calibrator for task %s: %v", task.ID(), err)
		}
		if err := calibrator.Load(lnnStateFile); err != nil {
			log.Printf("warn: could not load LNN state for task %s from %s: %v", task.ID(), lnnStateFile, err)
			continue // Skip if a specialist LNN hasn't been trained
		}
		output, err := calibrator.Generate(task.ID(), nil) // Generate params in inference mode
		if err != nil {
			log.Printf("warn: could not generate params for task %s: %v", task.ID(), err)
			continue
		}
		params := output.Params
		candidateSets[task.ID()] = params
		fmt.Printf("Generated candidate params for %s: %v\n", task.ID(), params)
	}

	if len(candidateSets) == 0 {
		return nil, fmt.Errorf("no specialist LNNs were found or could be loaded. Run 'aibench tune --lnn' for each benchmark first.")
	}

	// Step 2: Search for a single static parameter set that maximizes the Nash product.
	// For simplicity, we'll create a composite set by averaging the candidates.
	// A more advanced implementation would search this space.
	finalParams := make(map[string]float64)
	paramCounts := make(map[string]int)
	for _, paramSet := range candidateSets {
		for k, v := range paramSet {
			finalParams[k] += v
			paramCounts[k]++
		}
	}
	for k, total := range finalParams {
		if count := paramCounts[k]; count > 0 {
			finalParams[k] = total / float64(count)
		}
	}

	fmt.Printf("\n--- Nash Calibration Complete ---\n")
	fmt.Printf("Final calibrated parameters (averaged from %d specialists): %v\n", len(candidateSets), finalParams)

	return finalParams, nil
}

func (nc *LNNNashCalibrator) evaluateWithParams(task registry.Runner, params map[string]float64) (map[string]float64, error) {
	opts := registry.RunOptions{
		Model:  "hybrid",
		Params: params,
	}
	summary, err := task.Run(context.Background(), opts)
	if err != nil {
		return nil, err
	}
	return summary.Metrics, nil
}
