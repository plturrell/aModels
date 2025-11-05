package main

import (
	"log"
	"os"
)

// MathematicalProcessor handles mathematical computations for data quality and analysis
type MathematicalProcessor struct {
	logger  *log.Logger
	enabled bool
}

// NewMathematicalProcessor creates a new mathematical processor
func NewMathematicalProcessor(logger *log.Logger) *MathematicalProcessor {
	return &MathematicalProcessor{
		logger:  logger,
		enabled: os.Getenv("USE_MATHS_PROCESSING") == "true",
	}
}

// ComputeStatistics computes statistical measures for numerical data
func (mp *MathematicalProcessor) ComputeStatistics(values []float64) map[string]float64 {
	if !mp.enabled {
		return nil
	}

	stats := make(map[string]float64)
	
	if len(values) == 0 {
		return stats
	}

	// Basic statistics
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	stats["mean"] = mean

	// Variance
	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(values))
	stats["variance"] = variance
	stats["std_dev"] = sqrt(variance)

	// Min/Max
	min := values[0]
	max := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	stats["min"] = min
	stats["max"] = max
	stats["range"] = max - min

	return stats
}

// sqrt computes square root (simplified implementation)
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}
	
	// Newton's method approximation
	guess := x
	for i := 0; i < 10; i++ {
		guess = (guess + x/guess) / 2
	}
	return guess
}

// ComputeCorrelation computes correlation coefficient between two datasets
func (mp *MathematicalProcessor) ComputeCorrelation(x, y []float64) float64 {
	if !mp.enabled || len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	// Calculate means
	meanX := 0.0
	meanY := 0.0
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(y))

	// Calculate correlation
	numerator := 0.0
	varSumX := 0.0
	varSumY := 0.0

	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		varSumX += dx * dx
		varSumY += dy * dy
	}

	if varSumX == 0 || varSumY == 0 {
		return 0.0
	}

	denominator := sqrt(varSumX * varSumY)
	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

