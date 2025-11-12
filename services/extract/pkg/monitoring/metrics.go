package monitoring

import (
	"log"
	"sync"
	"time"
)

// MetricsCollector collects metrics for all improvements
type MetricsCollector struct {
	mu sync.RWMutex
	
	// Validation metrics
	ValidationMetrics struct {
		TotalValidated    int64
		NodesValidated    int64
		EdgesValidated    int64
		NodesRejected     int64
		EdgesRejected     int64
		ValidationErrors  int64
		ValidationTime    time.Duration
		TotalValidationTime time.Duration
	}
	
	// Retry metrics
	RetryMetrics struct {
		TotalRetries      int64
		SuccessfulRetries int64
		FailedRetries     int64
		TotalRetryTime    time.Duration
		AvgRetryTime      time.Duration
	}
	
	// Consistency metrics
	ConsistencyMetrics struct {
		TotalChecks       int64
		ConsistentChecks  int64
		InconsistentChecks int64
		TotalCheckTime    time.Duration
		AvgCheckTime      time.Duration
		NodeVariance      int64
		EdgeVariance      int64
	}
	
	// Neo4j batch metrics
	Neo4jBatchMetrics struct {
		TotalBatches      int64
		TotalNodes        int64
		TotalEdges        int64
		AvgBatchSize      int64
		TotalBatchTime    time.Duration
		AvgBatchTime      time.Duration
		BatchErrors       int64
	}
	
	logger *log.Logger
}

var globalMetrics *MetricsCollector
var metricsOnce sync.Once

// GetMetricsCollector returns the global metrics collector
func GetMetricsCollector(logger *log.Logger) *MetricsCollector {
	metricsOnce.Do(func() {
		globalMetrics = &MetricsCollector{
			logger: logger,
		}
	})
	return globalMetrics
}

// RecordValidation records validation metrics
func (m *MetricsCollector) RecordValidation(result ValidationResult, duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.ValidationMetrics.TotalValidated++
	m.ValidationMetrics.NodesValidated += int64(result.Metrics.NodesValidated)
	m.ValidationMetrics.EdgesValidated += int64(result.Metrics.EdgesValidated)
	m.ValidationMetrics.NodesRejected += int64(result.Metrics.NodesRejected)
	m.ValidationMetrics.EdgesRejected += int64(result.Metrics.EdgesRejected)
	m.ValidationMetrics.ValidationErrors += int64(result.Metrics.ValidationErrors)
	m.ValidationMetrics.TotalValidationTime += duration
	
	// Calculate average
	if m.ValidationMetrics.TotalValidated > 0 {
		m.ValidationMetrics.ValidationTime = m.ValidationMetrics.TotalValidationTime / time.Duration(m.ValidationMetrics.TotalValidated)
	}
}

// RecordRetry records retry metrics
func (m *MetricsCollector) RecordRetry(success bool, duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.RetryMetrics.TotalRetries++
	if success {
		m.RetryMetrics.SuccessfulRetries++
	} else {
		m.RetryMetrics.FailedRetries++
	}
	m.RetryMetrics.TotalRetryTime += duration
	
	// Calculate average
	if m.RetryMetrics.TotalRetries > 0 {
		m.RetryMetrics.AvgRetryTime = m.RetryMetrics.TotalRetryTime / time.Duration(m.RetryMetrics.TotalRetries)
	}
}

// RecordConsistency records consistency check metrics
func (m *MetricsCollector) RecordConsistency(result ConsistencyResult, duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.ConsistencyMetrics.TotalChecks++
	if result.Consistent {
		m.ConsistencyMetrics.ConsistentChecks++
	} else {
		m.ConsistencyMetrics.InconsistentChecks++
	}
	m.ConsistencyMetrics.TotalCheckTime += duration
	m.ConsistencyMetrics.NodeVariance += int64(result.Metrics.NodeVariance)
	m.ConsistencyMetrics.EdgeVariance += int64(result.Metrics.EdgeVariance)
	
	// Calculate average
	if m.ConsistencyMetrics.TotalChecks > 0 {
		m.ConsistencyMetrics.AvgCheckTime = m.ConsistencyMetrics.TotalCheckTime / time.Duration(m.ConsistencyMetrics.TotalChecks)
	}
}

// RecordNeo4jBatch records Neo4j batch processing metrics
func (m *MetricsCollector) RecordNeo4jBatch(nodeCount, edgeCount int, batchCount int, duration time.Duration, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.Neo4jBatchMetrics.TotalBatches += int64(batchCount)
	m.Neo4jBatchMetrics.TotalNodes += int64(nodeCount)
	m.Neo4jBatchMetrics.TotalEdges += int64(edgeCount)
	m.Neo4jBatchMetrics.TotalBatchTime += duration
	
	if err != nil {
		m.Neo4jBatchMetrics.BatchErrors++
	}
	
	// Calculate averages
	if m.Neo4jBatchMetrics.TotalBatches > 0 {
		m.Neo4jBatchMetrics.AvgBatchSize = m.Neo4jBatchMetrics.TotalNodes / m.Neo4jBatchMetrics.TotalBatches
		m.Neo4jBatchMetrics.AvgBatchTime = m.Neo4jBatchMetrics.TotalBatchTime / time.Duration(m.Neo4jBatchMetrics.TotalBatches)
	}
}

// GetMetrics returns all collected metrics
func (m *MetricsCollector) GetMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	return map[string]interface{}{
		"validation": map[string]interface{}{
			"total_validated":     m.ValidationMetrics.TotalValidated,
			"nodes_validated":     m.ValidationMetrics.NodesValidated,
			"edges_validated":     m.ValidationMetrics.EdgesValidated,
			"nodes_rejected":      m.ValidationMetrics.NodesRejected,
			"edges_rejected":      m.ValidationMetrics.EdgesRejected,
			"validation_errors":   m.ValidationMetrics.ValidationErrors,
			"avg_validation_time": m.ValidationMetrics.ValidationTime.String(),
		},
		"retry": map[string]interface{}{
			"total_retries":       m.RetryMetrics.TotalRetries,
			"successful_retries":  m.RetryMetrics.SuccessfulRetries,
			"failed_retries":     m.RetryMetrics.FailedRetries,
			"success_rate":        safeRate(m.RetryMetrics.SuccessfulRetries, m.RetryMetrics.TotalRetries),
			"avg_retry_time":      m.RetryMetrics.AvgRetryTime.String(),
		},
		"consistency": map[string]interface{}{
			"total_checks":        m.ConsistencyMetrics.TotalChecks,
			"consistent_checks":   m.ConsistencyMetrics.ConsistentChecks,
			"inconsistent_checks": m.ConsistencyMetrics.InconsistentChecks,
			"consistency_rate":    safeRate(m.ConsistencyMetrics.ConsistentChecks, m.ConsistencyMetrics.TotalChecks),
			"avg_node_variance":   safeAverage(m.ConsistencyMetrics.NodeVariance, m.ConsistencyMetrics.TotalChecks),
			"avg_edge_variance":   safeAverage(m.ConsistencyMetrics.EdgeVariance, m.ConsistencyMetrics.TotalChecks),
			"avg_check_time":      m.ConsistencyMetrics.AvgCheckTime.String(),
		},
		"neo4j_batch": map[string]interface{}{
			"total_batches":       m.Neo4jBatchMetrics.TotalBatches,
			"total_nodes":         m.Neo4jBatchMetrics.TotalNodes,
			"total_edges":         m.Neo4jBatchMetrics.TotalEdges,
			"avg_batch_size":     m.Neo4jBatchMetrics.AvgBatchSize,
			"avg_batch_time":     m.Neo4jBatchMetrics.AvgBatchTime.String(),
			"batch_errors":       m.Neo4jBatchMetrics.BatchErrors,
			"error_rate":         safeRate(m.Neo4jBatchMetrics.BatchErrors, m.Neo4jBatchMetrics.TotalBatches),
		},
	}
}

func safeRate(num, denom int64) float64 {
	if denom == 0 {
		return 0
	}
	return float64(num) / float64(denom) * 100
}

func safeAverage(sum int64, count int64) float64 {
	if count == 0 {
		return 0
	}
	return float64(sum) / float64(count)
}

// Reset resets all metrics
func (m *MetricsCollector) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.ValidationMetrics = struct {
		TotalValidated    int64
		NodesValidated    int64
		EdgesValidated    int64
		NodesRejected     int64
		EdgesRejected     int64
		ValidationErrors  int64
		ValidationTime    time.Duration
		TotalValidationTime time.Duration
	}{}
	
	m.RetryMetrics = struct {
		TotalRetries      int64
		SuccessfulRetries int64
		FailedRetries     int64
		TotalRetryTime    time.Duration
		AvgRetryTime      time.Duration
	}{}
	
	m.ConsistencyMetrics = struct {
		TotalChecks       int64
		ConsistentChecks  int64
		InconsistentChecks int64
		TotalCheckTime    time.Duration
		AvgCheckTime      time.Duration
		NodeVariance      int64
		EdgeVariance      int64
	}{}
	
	m.Neo4jBatchMetrics = struct {
		TotalBatches      int64
		TotalNodes        int64
		TotalEdges        int64
		AvgBatchSize      int64
		TotalBatchTime    time.Duration
		AvgBatchTime      time.Duration
		BatchErrors       int64
	}{}
}

