package tuning

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// AutoTuner provides automatic parameter adjustment based on performance data
type AutoTuner struct {
	mu             sync.RWMutex
	parameters     map[string]int
	baseline       map[string]float64
	testVariants   map[string]int
	measurements   map[string][]Measurement
	adjustmentHistory map[string][]Adjustment
	learningRate   float64
	explorationRate float64
}

// Measurement represents a performance measurement
type Measurement struct {
	Parameter   string
	Value       int
	Throughput  float64
	Latency     time.Duration
	MemoryUsage uint64
	Timestamp   time.Time
}

// Adjustment represents a parameter adjustment
type Adjustment struct {
	Parameter string
	OldValue  int
	NewValue  int
	Reason    string
	Timestamp time.Time
	Result    string // "improved", "degraded", "neutral"
}

// NewAutoTuner creates a new auto-tuner
func NewAutoTuner() *AutoTuner {
	return &AutoTuner{
		parameters: map[string]int{
			"parallel_threshold": 10000,
			"batch_size":         256,
			"cache_size":         10000,
			"worker_count":       4,
			"simd_threshold":     1000,
		},
		baseline:           make(map[string]float64),
		testVariants:       make(map[string]int),
		measurements:       make(map[string][]Measurement),
		adjustmentHistory:  make(map[string][]Adjustment),
		learningRate:       0.1,
		explorationRate:    0.2,
	}
}

// GetParameter returns the current value of a parameter
func (at *AutoTuner) GetParameter(name string) int {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	if value, exists := at.parameters[name]; exists {
		return value
	}
	return 0
}

// SetParameter sets a parameter value
func (at *AutoTuner) SetParameter(name string, value int) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	at.parameters[name] = value
}

// RecordPerformance records a performance measurement
func (at *AutoTuner) RecordPerformance(operation string, parameter string, value int, throughput float64, latency time.Duration, memoryUsage uint64) {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	measurement := Measurement{
		Parameter:   parameter,
		Value:       value,
		Throughput:  throughput,
		Latency:     latency,
		MemoryUsage: memoryUsage,
		Timestamp:   time.Now(),
	}
	
	at.measurements[operation] = append(at.measurements[operation], measurement)
	
	// Keep only recent measurements (last 1000)
	maxMeasurements := 1000
	if len(at.measurements[operation]) > maxMeasurements {
		at.measurements[operation] = at.measurements[operation][len(at.measurements)-maxMeasurements:]
	}
	
	// Periodically adjust parameters
	if len(at.measurements[operation]) >= 50 {
		at.adjustParameters(operation)
	}
}

// adjustParameters adjusts parameters based on performance data
func (at *AutoTuner) adjustParameters(operation string) {
	measurements := at.measurements[operation]
	if len(measurements) < 10 {
		return
	}
	
	// Analyze performance trends
	for parameter := range at.parameters {
		at.optimizeParameter(operation, parameter, measurements)
	}
}

// optimizeParameter optimizes a specific parameter
func (at *AutoTuner) optimizeParameter(operation, parameter string, measurements []Measurement) {
	// Group measurements by parameter value
	valueGroups := make(map[int][]Measurement)
	for _, m := range measurements {
		if m.Parameter == parameter {
			valueGroups[m.Value] = append(valueGroups[m.Value], m)
		}
	}
	
	if len(valueGroups) < 2 {
		return // Need at least 2 different values to compare
	}
	
	// Calculate average performance for each value
	valuePerformance := make(map[int]float64)
	for value, group := range valueGroups {
		if len(group) == 0 {
			continue
		}
		
		totalThroughput := 0.0
		for _, m := range group {
			totalThroughput += m.Throughput
		}
		valuePerformance[value] = totalThroughput / float64(len(group))
	}
	
	// Find best performing value
	bestValue := 0
	bestPerformance := 0.0
	
	for value, performance := range valuePerformance {
		if performance > bestPerformance {
			bestPerformance = performance
			bestValue = value
		}
	}
	
	// Adjust parameter if improvement is significant
	currentValue := at.parameters[parameter]
	if bestValue != currentValue && bestPerformance > valuePerformance[currentValue]*1.05 {
		at.adjustParameter(parameter, currentValue, bestValue, "performance_optimization")
	}
}

// adjustParameter adjusts a parameter and records the change
func (at *AutoTuner) adjustParameter(parameter string, oldValue, newValue int, reason string) {
	at.parameters[parameter] = newValue
	
	adjustment := Adjustment{
		Parameter: parameter,
		OldValue:  oldValue,
		NewValue:  newValue,
		Reason:    reason,
		Timestamp: time.Now(),
		Result:    "pending", // Will be updated based on performance
	}
	
	at.adjustmentHistory[parameter] = append(at.adjustmentHistory[parameter], adjustment)
	
	// Keep only recent adjustments
	maxHistory := 100
	if len(at.adjustmentHistory[parameter]) > maxHistory {
		at.adjustmentHistory[parameter] = at.adjustmentHistory[parameter][len(at.adjustmentHistory[parameter])-maxHistory:]
	}
}

// GetOptimalBatchSize returns the optimal batch size for an operation
func (at *AutoTuner) GetOptimalBatchSize(operation string, totalSize int) int {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	// Use learned batch size if available
	if measurements, exists := at.measurements[operation]; exists && len(measurements) > 0 {
		// Find batch size with best throughput
		bestBatchSize := at.parameters["batch_size"]
		bestThroughput := 0.0
		
		valueGroups := make(map[int][]Measurement)
		for _, m := range measurements {
			if m.Parameter == "batch_size" {
				valueGroups[m.Value] = append(valueGroups[m.Value], m)
			}
		}
		
		for batchSize, group := range valueGroups {
			if len(group) == 0 {
				continue
			}
			
			avgThroughput := 0.0
			for _, m := range group {
				avgThroughput += m.Throughput
			}
			avgThroughput /= float64(len(group))
			
			if avgThroughput > bestThroughput {
				bestThroughput = avgThroughput
				bestBatchSize = batchSize
			}
		}
		
		// Ensure batch size doesn't exceed total size
		if bestBatchSize > totalSize {
			bestBatchSize = totalSize
		}
		
		return bestBatchSize
	}
	
	// Default batch size
	return min(at.parameters["batch_size"], totalSize)
}

// GetOptimalWorkerCount returns the optimal number of workers
func (at *AutoTuner) GetOptimalWorkerCount() int {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	// Use learned worker count
	return at.parameters["worker_count"]
}

// GetPerformanceReport returns a performance report
func (at *AutoTuner) GetPerformanceReport() map[string]interface{} {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	report := make(map[string]interface{})
	report["parameters"] = at.parameters
	report["baseline"] = at.baseline
	
	// Calculate parameter effectiveness
	effectiveness := make(map[string]map[string]float64)
	for operation, measurements := range at.measurements {
		if len(measurements) == 0 {
			continue
		}
		
		effectiveness[operation] = make(map[string]float64)
		
		// Group by parameter
		paramGroups := make(map[string][]Measurement)
		for _, m := range measurements {
			paramGroups[m.Parameter] = append(paramGroups[m.Parameter], m)
		}
		
		// Calculate effectiveness for each parameter
		for param, group := range paramGroups {
			if len(group) == 0 {
				continue
			}
			
			avgThroughput := 0.0
			avgLatency := 0.0
			for _, m := range group {
				avgThroughput += m.Throughput
				avgLatency += float64(m.Latency.Nanoseconds())
			}
			avgThroughput /= float64(len(group))
			avgLatency /= float64(len(group))
			
			effectiveness[operation][param+"_throughput"] = avgThroughput
			effectiveness[operation][param+"_latency_ns"] = avgLatency
		}
	}
	
	report["effectiveness"] = effectiveness
	
	// Recent adjustments
	recentAdjustments := make(map[string][]Adjustment)
	for param, adjustments := range at.adjustmentHistory {
		if len(adjustments) > 0 {
			// Get last 10 adjustments
			start := len(adjustments) - 10
			if start < 0 {
				start = 0
			}
			recentAdjustments[param] = adjustments[start:]
		}
	}
	report["recent_adjustments"] = recentAdjustments
	
	return report
}

// GetRecommendations returns optimization recommendations
func (at *AutoTuner) GetRecommendations() []string {
	at.mu.RLock()
	defer at.mu.RUnlock()
	
	recommendations := make([]string, 0)
	
	// Analyze parameter effectiveness
	for operation, measurements := range at.measurements {
		if len(measurements) < 20 {
			continue // Need more data
		}
		
		// Check for underperforming parameters
		paramPerformance := make(map[string]float64)
		for _, m := range measurements {
			paramPerformance[m.Parameter] += m.Throughput
		}
		
		// Find worst performing parameter
		worstParam := ""
		worstPerformance := math.Inf(1)
		
		for param, performance := range paramPerformance {
			if performance < worstPerformance {
				worstPerformance = performance
				worstParam = param
			}
		}
		
		if worstParam != "" {
			recommendations = append(recommendations, 
				"Consider optimizing "+worstParam+" for "+operation+" (current performance: "+formatFloat(worstPerformance)+" ops/sec)")
		}
	}
	
	// Check for parameter conflicts
	conflicts := at.detectParameterConflicts()
	recommendations = append(recommendations, conflicts...)
	
	return recommendations
}

// detectParameterConflicts detects conflicting parameter settings
func (at *AutoTuner) detectParameterConflicts() []string {
	conflicts := make([]string, 0)
	
	// Check for conflicting thresholds
	parallelThreshold := at.parameters["parallel_threshold"]
	simdThreshold := at.parameters["simd_threshold"]
	
	if simdThreshold >= parallelThreshold {
		conflicts = append(conflicts, 
			"SIMD threshold ("+string(rune(simdThreshold))+") should be less than parallel threshold ("+string(rune(parallelThreshold))+")")
	}
	
	// Check for memory vs performance trade-offs
	cacheSize := at.parameters["cache_size"]
	workerCount := at.parameters["worker_count"]
	
	if cacheSize > 100000 && workerCount > 8 {
		conflicts = append(conflicts, 
			"High cache size and worker count may cause memory pressure")
	}
	
	return conflicts
}

// Reset clears all tuning data
func (at *AutoTuner) Reset() {
	at.mu.Lock()
	defer at.mu.Unlock()
	
	at.measurements = make(map[string][]Measurement)
	at.adjustmentHistory = make(map[string][]Adjustment)
	at.baseline = make(map[string]float64)
}

// Helper functions

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func formatFloat(f float64) string {
	return fmt.Sprintf("%.2f", f)
}

// Global auto-tuner instance
var globalTuner *AutoTuner
var tunerOnce sync.Once

// GetGlobalTuner returns the global auto-tuner
func GetGlobalTuner() *AutoTuner {
	tunerOnce.Do(func() {
		globalTuner = NewAutoTuner()
	})
	return globalTuner
}
