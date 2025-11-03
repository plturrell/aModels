package adaptive

import (
	"sync"
	"time"
)

// OpProfile represents a performance profile for an operation
type OpProfile struct {
	Size        int
	Duration    time.Duration
	Algorithm   string
	Timestamp   time.Time
	Throughput  float64 // Operations per second
	MemoryUsage uint64  // Bytes used
}

// AdaptiveOptimizer automatically selects the best algorithm based on runtime profiling
type AdaptiveOptimizer struct {
	mu          sync.RWMutex
	profiles    map[string][]OpProfile
	thresholds  map[string]int
	baselines   map[string]float64
	learnRate   float64
	maxProfiles int
}

// Algorithm types
const (
	AlgorithmSequential = "sequential"
	AlgorithmSIMD       = "simd"
	AlgorithmParallel   = "parallel"
	AlgorithmBLAS       = "blas"
	AlgorithmFortran    = "fortran"
)

// NewAdaptiveOptimizer creates a new adaptive optimizer
func NewAdaptiveOptimizer() *AdaptiveOptimizer {
	return &AdaptiveOptimizer{
		profiles:    make(map[string][]OpProfile),
		baselines:   make(map[string]float64),
		learnRate:   0.1,
		maxProfiles: 1000,
		thresholds: map[string]int{
			"parallel": 10000,  // Use parallel for >10k elements
			"simd":     1000,   // Use SIMD for >1k elements
			"blas":     100000, // Use BLAS for >100k elements
		},
	}
}

// SelectAlgorithm chooses the best algorithm for an operation
func (ao *AdaptiveOptimizer) SelectAlgorithm(opType string, size int) string {
	ao.mu.RLock()
	defer ao.mu.RUnlock()

	// Check if we have enough data for this operation type
	profiles, exists := ao.profiles[opType]
	if !exists || len(profiles) < 10 {
		// Use default thresholds for new operations
		return ao.selectByThreshold(size)
	}

	// Use learned thresholds
	if size > ao.thresholds["blas"] {
		return AlgorithmBLAS
	} else if size > ao.thresholds["parallel"] {
		return AlgorithmParallel
	} else if size > ao.thresholds["simd"] {
		return AlgorithmSIMD
	}
	return AlgorithmSequential
}

// selectByThreshold selects algorithm based on static thresholds
func (ao *AdaptiveOptimizer) selectByThreshold(size int) string {
	if size > ao.thresholds["blas"] {
		return AlgorithmBLAS
	} else if size > ao.thresholds["parallel"] {
		return AlgorithmParallel
	} else if size > ao.thresholds["simd"] {
		return AlgorithmSIMD
	}
	return AlgorithmSequential
}

// RecordPerformance records the performance of an algorithm
func (ao *AdaptiveOptimizer) RecordPerformance(opType string, size int, duration time.Duration, algorithm string, memoryUsage uint64) {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	throughput := float64(size) / duration.Seconds()

	profile := OpProfile{
		Size:        size,
		Duration:    duration,
		Algorithm:   algorithm,
		Timestamp:   time.Now(),
		Throughput:  throughput,
		MemoryUsage: memoryUsage,
	}

	ao.profiles[opType] = append(ao.profiles[opType], profile)

	// Keep only recent profiles
	if len(ao.profiles[opType]) > ao.maxProfiles {
		ao.profiles[opType] = ao.profiles[opType][len(ao.profiles[opType])-ao.maxProfiles:]
	}

	// Adjust thresholds based on performance
	ao.adjustThresholds(opType)
}

// adjustThresholds adjusts algorithm selection thresholds based on performance data
func (ao *AdaptiveOptimizer) adjustThresholds(opType string) {
	profiles := ao.profiles[opType]
	if len(profiles) < 20 {
		return // Need more data
	}

	// Analyze performance by algorithm and size
	algorithmPerformance := make(map[string][]OpProfile)
	for _, profile := range profiles {
		algorithmPerformance[profile.Algorithm] = append(algorithmPerformance[profile.Algorithm], profile)
	}

	// Find optimal thresholds using statistical analysis
	ao.optimizeThresholds(opType, algorithmPerformance)
}

// optimizeThresholds finds optimal thresholds using performance data
func (ao *AdaptiveOptimizer) optimizeThresholds(opType string, algorithmPerformance map[string][]OpProfile) {
	// Calculate average throughput by algorithm and size ranges
	for _, algorithm := range []string{AlgorithmSequential, AlgorithmSIMD, AlgorithmParallel, AlgorithmBLAS} {
		profiles := algorithmPerformance[algorithm]
		if len(profiles) == 0 {
			continue
		}

		// Calculate performance metrics
		avgThroughput := 0.0
		for _, profile := range profiles {
			avgThroughput += profile.Throughput
		}
		avgThroughput /= float64(len(profiles))

		// Update baseline performance
		ao.baselines[algorithm] = avgThroughput
	}

	// Find crossover points where one algorithm becomes better than another
	ao.findCrossoverPoints()
}

// findCrossoverPoints finds where algorithms become optimal
func (ao *AdaptiveOptimizer) findCrossoverPoints() {
	// Find where SIMD becomes better than sequential
	simdBaseline := ao.baselines[AlgorithmSIMD]
	seqBaseline := ao.baselines[AlgorithmSequential]

	if simdBaseline > seqBaseline && simdBaseline > 0 {
		// SIMD is faster, find optimal threshold
		// This is a simplified heuristic - real implementation would use more sophisticated analysis
		newThreshold := int(float64(ao.thresholds["simd"]) * (seqBaseline / simdBaseline))
		if newThreshold > 0 && newThreshold < ao.thresholds["parallel"] {
			ao.thresholds["simd"] = newThreshold
		}
	}

	// Similar logic for other algorithms...
}

// GetOptimalBatchSize returns the optimal batch size for an operation
func (ao *AdaptiveOptimizer) GetOptimalBatchSize(opType string, totalSize int) int {
	ao.mu.RLock()
	defer ao.mu.RUnlock()

	profiles := ao.profiles[opType]
	if len(profiles) < 10 {
		// Default batch size
		return min(totalSize, 1000)
	}

	// Find batch size with best throughput
	bestBatchSize := 1000
	bestThroughput := 0.0

	for _, profile := range profiles {
		if profile.Throughput > bestThroughput {
			bestThroughput = profile.Throughput
			bestBatchSize = profile.Size
		}
	}

	return min(bestBatchSize, totalSize)
}

// GetPerformanceReport returns a performance report for an operation type
func (ao *AdaptiveOptimizer) GetPerformanceReport(opType string) map[string]interface{} {
	ao.mu.RLock()
	defer ao.mu.RUnlock()

	profiles := ao.profiles[opType]
	if len(profiles) == 0 {
		return map[string]interface{}{
			"operation": opType,
			"profiles":  0,
			"message":   "No performance data available",
		}
	}

	// Calculate statistics by algorithm
	algorithmStats := make(map[string]map[string]float64)
	algorithmCounts := make(map[string]int)

	for _, profile := range profiles {
		algorithm := profile.Algorithm
		algorithmCounts[algorithm]++

		if algorithmStats[algorithm] == nil {
			algorithmStats[algorithm] = make(map[string]float64)
		}

		algorithmStats[algorithm]["total_throughput"] += profile.Throughput
		algorithmStats[algorithm]["total_duration"] += float64(profile.Duration.Nanoseconds())
		algorithmStats[algorithm]["total_memory"] += float64(profile.MemoryUsage)
	}

	// Calculate averages
	for algorithm, stats := range algorithmStats {
		count := float64(algorithmCounts[algorithm])
		stats["avg_throughput"] = stats["total_throughput"] / count
		stats["avg_duration_ns"] = stats["total_duration"] / count
		stats["avg_memory_bytes"] = stats["total_memory"] / count
		stats["count"] = count
	}

	return map[string]interface{}{
		"operation":       opType,
		"total_profiles":  len(profiles),
		"thresholds":      ao.thresholds,
		"baselines":       ao.baselines,
		"algorithm_stats": algorithmStats,
		"recommendations": ao.generateRecommendations(opType, algorithmStats),
	}
}

// generateRecommendations generates optimization recommendations
func (ao *AdaptiveOptimizer) generateRecommendations(opType string, algorithmStats map[string]map[string]float64) []string {
	recommendations := make([]string, 0)

	// Find best performing algorithm
	bestAlgorithm := ""
	bestThroughput := 0.0

	for algorithm, stats := range algorithmStats {
		throughput := stats["avg_throughput"]
		if throughput > bestThroughput {
			bestThroughput = throughput
			bestAlgorithm = algorithm
		}
	}

	if bestAlgorithm != "" {
		recommendations = append(recommendations,
			"Consider using "+bestAlgorithm+" algorithm for better performance")
	}

	// Check for memory usage issues
	for algorithm, stats := range algorithmStats {
		memoryUsage := stats["avg_memory_bytes"]
		if memoryUsage > 100*1024*1024 { // 100MB
			recommendations = append(recommendations,
				algorithm+" algorithm uses high memory ("+formatBytes(uint64(memoryUsage))+")")
		}
	}

	return recommendations
}

// GetThresholds returns current algorithm selection thresholds
func (ao *AdaptiveOptimizer) GetThresholds() map[string]int {
	ao.mu.RLock()
	defer ao.mu.RUnlock()

	// Return copy to prevent external modification
	thresholds := make(map[string]int)
	for k, v := range ao.thresholds {
		thresholds[k] = v
	}
	return thresholds
}

// SetThresholds sets algorithm selection thresholds
func (ao *AdaptiveOptimizer) SetThresholds(thresholds map[string]int) {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	for k, v := range thresholds {
		ao.thresholds[k] = v
	}
}

// Reset clears all performance data
func (ao *AdaptiveOptimizer) Reset() {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	ao.profiles = make(map[string][]OpProfile)
	ao.baselines = make(map[string]float64)
}

// Helper functions

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return string(rune(bytes)) + " B"
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return string(rune(bytes/uint64(div))) + " " + string(rune(exp)) + "B"
}

// Global adaptive optimizer instance
var globalOptimizer *AdaptiveOptimizer
var optimizerOnce sync.Once

// GetGlobalOptimizer returns the global adaptive optimizer
func GetGlobalOptimizer() *AdaptiveOptimizer {
	optimizerOnce.Do(func() {
		globalOptimizer = NewAdaptiveOptimizer()
	})
	return globalOptimizer
}
