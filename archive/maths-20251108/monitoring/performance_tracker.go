// Package monitoring provides performance tracking and analytics for the maths package
// Build: LYR1-MATH001 | Version: 1.0.0 | Module: monitoring
// Architecture: Layer1-Core | Component: maths-performance-tracking
// Dependencies: tracked
// Last-Modified: 2025-01-19T00:00:00Z

package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// PerformanceTracker tracks performance metrics for maths operations
type PerformanceTracker struct {
	operations map[string]*OperationMetrics
	mu         sync.RWMutex
	startTime  time.Time
}

// OperationMetrics tracks metrics for a specific operation
type OperationMetrics struct {
	Count         int64
	TotalTime     time.Duration
	MinTime       time.Duration
	MaxTime       time.Duration
	AverageTime   time.Duration
	LastCall      time.Time
	ErrorCount    int64
	SuccessRate   float64
	Throughput    float64 // operations per second
	MemoryUsage   int64   // bytes
	SIMDUsage     int64   // number of SIMD operations
	FallbackUsage int64   // number of fallback operations
}

// PerformanceReport provides a comprehensive performance report
type PerformanceReport struct {
	TotalOperations   int64
	TotalTime         time.Duration
	AverageThroughput float64
	TopOperations     []OperationSummary
	SIMDEfficiency    float64
	ErrorRate         float64
	MemoryEfficiency  float64
	GeneratedAt       time.Time
}

// OperationSummary summarizes performance for a single operation
type OperationSummary struct {
	Operation   string
	Count       int64
	AverageTime time.Duration
	Throughput  float64
	SuccessRate float64
	SIMDUsage   int64
	MemoryUsage int64
}

// NewPerformanceTracker creates a new performance tracker
func NewPerformanceTracker() *PerformanceTracker {
	return &PerformanceTracker{
		operations: make(map[string]*OperationMetrics),
		startTime:  time.Now(),
	}
}

// TrackOperation records performance metrics for an operation
func (pt *PerformanceTracker) TrackOperation(operation string, duration time.Duration, success bool, memoryUsage int64, simdUsed bool) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.operations[operation] == nil {
		pt.operations[operation] = &OperationMetrics{
			MinTime: duration,
			MaxTime: duration,
		}
	}

	metrics := pt.operations[operation]
	metrics.Count++
	metrics.TotalTime += duration
	metrics.LastCall = time.Now()

	if duration < metrics.MinTime {
		metrics.MinTime = duration
	}
	if duration > metrics.MaxTime {
		metrics.MaxTime = duration
	}

	metrics.AverageTime = metrics.TotalTime / time.Duration(metrics.Count)

	if success {
		metrics.SuccessRate = float64(metrics.Count-metrics.ErrorCount) / float64(metrics.Count)
	} else {
		metrics.ErrorCount++
		metrics.SuccessRate = float64(metrics.Count-metrics.ErrorCount) / float64(metrics.Count)
	}

	// Calculate throughput (operations per second)
	if metrics.TotalTime > 0 {
		metrics.Throughput = float64(metrics.Count) / metrics.TotalTime.Seconds()
	}

	metrics.MemoryUsage += memoryUsage

	if simdUsed {
		metrics.SIMDUsage++
	} else {
		metrics.FallbackUsage++
	}
}

// GetMetrics returns metrics for a specific operation
func (pt *PerformanceTracker) GetMetrics(operation string) (*OperationMetrics, bool) {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	metrics, exists := pt.operations[operation]
	return metrics, exists
}

// GetAllMetrics returns all operation metrics
func (pt *PerformanceTracker) GetAllMetrics() map[string]*OperationMetrics {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	result := make(map[string]*OperationMetrics)
	for op, metrics := range pt.operations {
		// Create a copy to avoid race conditions
		copy := *metrics
		result[op] = &copy
	}
	return result
}

// GenerateReport generates a comprehensive performance report
func (pt *PerformanceTracker) GenerateReport() *PerformanceReport {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	report := &PerformanceReport{
		GeneratedAt: time.Now(),
	}

	var totalOperations int64
	var totalTime time.Duration
	var totalSIMDUsage int64
	var totalFallbackUsage int64
	var totalErrors int64

	// Calculate totals
	for _, metrics := range pt.operations {
		totalOperations += metrics.Count
		totalTime += metrics.TotalTime
		totalSIMDUsage += metrics.SIMDUsage
		totalFallbackUsage += metrics.FallbackUsage
		totalErrors += metrics.ErrorCount
	}

	report.TotalOperations = totalOperations
	report.TotalTime = totalTime

	if totalTime > 0 {
		report.AverageThroughput = float64(totalOperations) / totalTime.Seconds()
	}

	if totalSIMDUsage+totalFallbackUsage > 0 {
		report.SIMDEfficiency = float64(totalSIMDUsage) / float64(totalSIMDUsage+totalFallbackUsage)
	}

	if totalOperations > 0 {
		report.ErrorRate = float64(totalErrors) / float64(totalOperations)
	}

	// Generate top operations summary
	report.TopOperations = make([]OperationSummary, 0, len(pt.operations))
	for op, metrics := range pt.operations {
		summary := OperationSummary{
			Operation:   op,
			Count:       metrics.Count,
			AverageTime: metrics.AverageTime,
			Throughput:  metrics.Throughput,
			SuccessRate: metrics.SuccessRate,
			SIMDUsage:   metrics.SIMDUsage,
			MemoryUsage: metrics.MemoryUsage,
		}
		report.TopOperations = append(report.TopOperations, summary)
	}

	// Sort by throughput (descending)
	for i := 0; i < len(report.TopOperations)-1; i++ {
		for j := i + 1; j < len(report.TopOperations); j++ {
			if report.TopOperations[i].Throughput < report.TopOperations[j].Throughput {
				report.TopOperations[i], report.TopOperations[j] = report.TopOperations[j], report.TopOperations[i]
			}
		}
	}

	// Limit to top 10
	if len(report.TopOperations) > 10 {
		report.TopOperations = report.TopOperations[:10]
	}

	return report
}

// Reset clears all performance metrics
func (pt *PerformanceTracker) Reset() {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	pt.operations = make(map[string]*OperationMetrics)
	pt.startTime = time.Now()
}

// GetUptime returns the uptime since the tracker was created
func (pt *PerformanceTracker) GetUptime() time.Duration {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	return time.Since(pt.startTime)
}

// GetOperationCount returns the total number of operations tracked
func (pt *PerformanceTracker) GetOperationCount() int64 {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	var total int64
	for _, metrics := range pt.operations {
		total += metrics.Count
	}
	return total
}

// GetSIMDEfficiency returns the overall SIMD efficiency
func (pt *PerformanceTracker) GetSIMDEfficiency() float64 {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	var totalSIMD int64
	var totalFallback int64

	for _, metrics := range pt.operations {
		totalSIMD += metrics.SIMDUsage
		totalFallback += metrics.FallbackUsage
	}

	if totalSIMD+totalFallback == 0 {
		return 0.0
	}

	return float64(totalSIMD) / float64(totalSIMD+totalFallback)
}

// PrintReport prints a formatted performance report
func (pt *PerformanceTracker) PrintReport() {
	report := pt.GenerateReport()

	fmt.Printf("\n=== Maths Package Performance Report ===\n")
	fmt.Printf("Generated at: %s\n", report.GeneratedAt.Format("2006-01-02 15:04:05"))
	fmt.Printf("Uptime: %v\n", pt.GetUptime())
	fmt.Printf("Total Operations: %d\n", report.TotalOperations)
	fmt.Printf("Total Time: %v\n", report.TotalTime)
	fmt.Printf("Average Throughput: %.2f ops/sec\n", report.AverageThroughput)
	fmt.Printf("SIMD Efficiency: %.2f%%\n", report.SIMDEfficiency*100)
	fmt.Printf("Error Rate: %.2f%%\n", report.ErrorRate*100)

	fmt.Printf("\nTop Operations by Throughput:\n")
	fmt.Printf("%-20s %8s %12s %10s %8s %8s %8s\n",
		"Operation", "Count", "Avg Time", "Throughput", "Success%", "SIMD", "Memory")
	fmt.Printf("%-20s %8s %12s %10s %8s %8s %8s\n",
		"---------", "-----", "--------", "----------", "-------", "----", "------")

	for _, op := range report.TopOperations {
		fmt.Printf("%-20s %8d %12v %10.2f %8.1f %8d %8d\n",
			op.Operation, op.Count, op.AverageTime, op.Throughput,
			op.SuccessRate*100, op.SIMDUsage, op.MemoryUsage)
	}
	fmt.Printf("\n")
}

// Global performance tracker instance
var globalTracker = NewPerformanceTracker()

// TrackOperation is a convenience function to track operations globally
func TrackOperation(operation string, duration time.Duration, success bool, memoryUsage int64, simdUsed bool) {
	globalTracker.TrackOperation(operation, duration, success, memoryUsage, simdUsed)
}

// GetGlobalPerformanceReport returns a report for the global tracker
func GetGlobalPerformanceReport() *PerformanceReport {
	return globalTracker.GenerateReport()
}

// PrintGlobalPerformanceReport prints the global performance report
func PrintGlobalPerformanceReport() {
	globalTracker.PrintReport()
}

// ResetGlobalTracker resets the global performance tracker
func ResetGlobalTracker() {
	globalTracker.Reset()
}
