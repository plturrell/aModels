package monitoring

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Dashboard provides real-time performance monitoring
type Dashboard struct {
	mu              sync.RWMutex
	operationCounts map[string]uint64
	latencies       map[string][]time.Duration
	errors          map[string]uint64
	throughput      map[string][]float64
	memoryUsage     map[string][]uint64
	startTime       time.Time
	lastUpdate      time.Time
}

// PerformanceMetrics represents aggregated performance metrics
type PerformanceMetrics struct {
	Operation     string
	Count         uint64
	AvgLatency    time.Duration
	P50Latency    time.Duration
	P95Latency    time.Duration
	P99Latency    time.Duration
	MaxLatency    time.Duration
	ErrorRate     float64
	Throughput    float64
	MemoryUsage   uint64
	LastOperation time.Time
}

// NewDashboard creates a new performance dashboard
func NewDashboard() *Dashboard {
	return &Dashboard{
		operationCounts: make(map[string]uint64),
		latencies:       make(map[string][]time.Duration),
		errors:          make(map[string]uint64),
		throughput:      make(map[string][]float64),
		memoryUsage:     make(map[string][]uint64),
		startTime:       time.Now(),
		lastUpdate:      time.Now(),
	}
}

// RecordOperation records a completed operation
func (d *Dashboard) RecordOperation(opType string, duration time.Duration, err error, memoryUsage uint64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.operationCounts[opType]++
	d.latencies[opType] = append(d.latencies[opType], duration)

	// Calculate throughput (operations per second)
	throughput := 1.0 / duration.Seconds()
	d.throughput[opType] = append(d.throughput[opType], throughput)

	// Record memory usage
	d.memoryUsage[opType] = append(d.memoryUsage[opType], memoryUsage)

	if err != nil {
		d.errors[opType]++
	}

	// Keep only recent data (last 1000 operations per type)
	maxHistory := 1000
	if len(d.latencies[opType]) > maxHistory {
		d.latencies[opType] = d.latencies[opType][len(d.latencies[opType])-maxHistory:]
	}
	if len(d.throughput[opType]) > maxHistory {
		d.throughput[opType] = d.throughput[opType][len(d.throughput[opType])-maxHistory:]
	}
	if len(d.memoryUsage[opType]) > maxHistory {
		d.memoryUsage[opType] = d.memoryUsage[opType][len(d.memoryUsage[opType])-maxHistory:]
	}

	d.lastUpdate = time.Now()
}

// GetMetrics returns current performance metrics
func (d *Dashboard) GetMetrics() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	metrics := make(map[string]interface{})
	metrics["uptime"] = time.Since(d.startTime)
	metrics["last_update"] = d.lastUpdate
	metrics["operation_counts"] = d.operationCounts
	metrics["error_counts"] = d.errors

	// Calculate detailed metrics for each operation
	operationMetrics := make(map[string]PerformanceMetrics)
	for opType := range d.operationCounts {
		operationMetrics[opType] = d.calculateOperationMetrics(opType)
	}
	metrics["operation_metrics"] = operationMetrics

	// Calculate system-wide metrics
	metrics["system_metrics"] = d.calculateSystemMetrics()

	return metrics
}

// calculateOperationMetrics calculates detailed metrics for an operation type
func (d *Dashboard) calculateOperationMetrics(opType string) PerformanceMetrics {
	latencies := d.latencies[opType]
	if len(latencies) == 0 {
		return PerformanceMetrics{Operation: opType}
	}

	// Sort latencies for percentile calculation
	sortedLatencies := make([]time.Duration, len(latencies))
	copy(sortedLatencies, latencies)
	sort.Slice(sortedLatencies, func(i, j int) bool {
		return sortedLatencies[i] < sortedLatencies[j]
	})

	// Calculate percentiles
	p50Index := len(sortedLatencies) * 50 / 100
	p95Index := len(sortedLatencies) * 95 / 100
	p99Index := len(sortedLatencies) * 99 / 100

	// Calculate average latency
	totalLatency := time.Duration(0)
	for _, latency := range latencies {
		totalLatency += latency
	}
	avgLatency := totalLatency / time.Duration(len(latencies))

	// Calculate error rate
	errorRate := float64(d.errors[opType]) / float64(d.operationCounts[opType]) * 100

	// Calculate average throughput
	avgThroughput := 0.0
	if len(d.throughput[opType]) > 0 {
		for _, tp := range d.throughput[opType] {
			avgThroughput += tp
		}
		avgThroughput /= float64(len(d.throughput[opType]))
	}

	// Calculate average memory usage
	avgMemory := uint64(0)
	if len(d.memoryUsage[opType]) > 0 {
		for _, mem := range d.memoryUsage[opType] {
			avgMemory += mem
		}
		avgMemory /= uint64(len(d.memoryUsage[opType]))
	}

	return PerformanceMetrics{
		Operation:     opType,
		Count:         d.operationCounts[opType],
		AvgLatency:    avgLatency,
		P50Latency:    sortedLatencies[p50Index],
		P95Latency:    sortedLatencies[p95Index],
		P99Latency:    sortedLatencies[p99Index],
		MaxLatency:    sortedLatencies[len(sortedLatencies)-1],
		ErrorRate:     errorRate,
		Throughput:    avgThroughput,
		MemoryUsage:   avgMemory,
		LastOperation: d.lastUpdate,
	}
}

// calculateSystemMetrics calculates system-wide performance metrics
func (d *Dashboard) calculateSystemMetrics() map[string]interface{} {
	totalOperations := uint64(0)
	totalErrors := uint64(0)

	for _, count := range d.operationCounts {
		totalOperations += count
	}

	for _, errors := range d.errors {
		totalErrors += errors
	}

	// Calculate overall error rate
	overallErrorRate := 0.0
	if totalOperations > 0 {
		overallErrorRate = float64(totalErrors) / float64(totalOperations) * 100
	}

	// Calculate operations per second
	uptime := time.Since(d.startTime)
	opsPerSecond := float64(totalOperations) / uptime.Seconds()

	return map[string]interface{}{
		"total_operations":   totalOperations,
		"total_errors":       totalErrors,
		"overall_error_rate": overallErrorRate,
		"operations_per_sec": opsPerSecond,
		"uptime_seconds":     uptime.Seconds(),
	}
}

// GetOperationHeatmap returns a heatmap of operation performance
func (d *Dashboard) GetOperationHeatmap() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	heatmap := make(map[string]interface{})

	for opType, latencies := range d.latencies {
		if len(latencies) == 0 {
			continue
		}

		// Create time buckets (last hour, divided into 10-minute intervals)
		now := time.Now()
		buckets := make([]map[string]interface{}, 6) // 6 x 10-minute buckets
		totals := make([]time.Duration, len(buckets))

		for i := range buckets {
			buckets[i] = map[string]interface{}{
				"start_time":  now.Add(-time.Duration(6-i) * 10 * time.Minute),
				"end_time":    now.Add(-time.Duration(5-i) * 10 * time.Minute),
				"count":       0,
				"avg_latency": time.Duration(0),
			}
		}

		// Distribute latencies into buckets
		for _, latency := range latencies {
			// This is a simplified implementation
			// In practice, you'd need to track timestamps for each operation
			bucketIndex := 0 // Placeholder - would calculate based on operation time
			if bucketIndex < len(buckets) {
				buckets[bucketIndex]["count"] = buckets[bucketIndex]["count"].(int) + 1
				totals[bucketIndex] += latency
			}
		}

		for i := range buckets {
			count := buckets[i]["count"].(int)
			if count > 0 {
				buckets[i]["avg_latency"] = totals[i] / time.Duration(count)
			}
		}

		heatmap[opType] = buckets
	}

	return heatmap
}

// GetBottlenecks identifies performance bottlenecks
func (d *Dashboard) GetBottlenecks() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	bottlenecks := make([]string, 0)

	for opType, metrics := range d.calculateAllOperationMetrics() {
		// Check for high latency
		if metrics.P95Latency > 100*time.Millisecond {
			bottlenecks = append(bottlenecks,
				opType+" has high P95 latency: "+metrics.P95Latency.String())
		}

		// Check for high error rate
		if metrics.ErrorRate > 5.0 {
			bottlenecks = append(bottlenecks,
				opType+" has high error rate: "+formatFloat(metrics.ErrorRate)+"%")
		}

		// Check for low throughput
		if metrics.Throughput < 100.0 && metrics.Count > 100 {
			bottlenecks = append(bottlenecks,
				opType+" has low throughput: "+formatFloat(metrics.Throughput)+" ops/sec")
		}

		// Check for high memory usage
		if metrics.MemoryUsage > 100*1024*1024 { // 100MB
			bottlenecks = append(bottlenecks,
				opType+" uses high memory: "+formatBytes(metrics.MemoryUsage))
		}
	}

	return bottlenecks
}

// calculateAllOperationMetrics calculates metrics for all operations
func (d *Dashboard) calculateAllOperationMetrics() map[string]PerformanceMetrics {
	metrics := make(map[string]PerformanceMetrics)

	for opType := range d.operationCounts {
		metrics[opType] = d.calculateOperationMetrics(opType)
	}

	return metrics
}

// GetTopOperations returns the most frequently used operations
func (d *Dashboard) GetTopOperations(limit int) []map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	type opCount struct {
		Operation string
		Count     uint64
	}

	var operations []opCount
	for opType, count := range d.operationCounts {
		operations = append(operations, opCount{opType, count})
	}

	// Sort by count (descending)
	sort.Slice(operations, func(i, j int) bool {
		return operations[i].Count > operations[j].Count
	})

	// Return top N
	if limit > len(operations) {
		limit = len(operations)
	}

	result := make([]map[string]interface{}, limit)
	for i := 0; i < limit; i++ {
		op := operations[i]
		metrics := d.calculateOperationMetrics(op.Operation)
		result[i] = map[string]interface{}{
			"operation":   op.Operation,
			"count":       op.Count,
			"avg_latency": metrics.AvgLatency,
			"error_rate":  metrics.ErrorRate,
			"throughput":  metrics.Throughput,
		}
	}

	return result
}

// Clear clears all dashboard data
func (d *Dashboard) Clear() {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.operationCounts = make(map[string]uint64)
	d.latencies = make(map[string][]time.Duration)
	d.errors = make(map[string]uint64)
	d.throughput = make(map[string][]float64)
	d.memoryUsage = make(map[string][]uint64)
	d.startTime = time.Now()
	d.lastUpdate = time.Now()
}

// Helper functions

func formatFloat(f float64) string {
	return fmt.Sprintf("%.2f", f)
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

// Global dashboard instance
var globalDashboard *Dashboard
var dashboardOnce sync.Once

// GetGlobalDashboard returns the global performance dashboard
func GetGlobalDashboard() *Dashboard {
	dashboardOnce.Do(func() {
		globalDashboard = NewDashboard()
	})
	return globalDashboard
}
