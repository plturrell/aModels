package metrics

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Neo4jMetrics tracks Neo4j client performance and health metrics.
type Neo4jMetrics struct {
	// Query metrics
	QueryDuration      *prometheus.HistogramVec
	QueryCount         *prometheus.CounterVec
	QueryErrors        *prometheus.CounterVec
	QueryComplexity    *prometheus.HistogramVec
	
	// Connection metrics
	ConnectionsActive  prometheus.Gauge
	ConnectionsIdle    prometheus.Gauge
	ConnectionErrors   prometheus.Counter
	ConnectionDuration prometheus.Histogram
	
	// Transaction metrics
	TransactionDuration *prometheus.HistogramVec
	TransactionCount    *prometheus.CounterVec
	TransactionErrors   *prometheus.CounterVec
	
	// Batch operation metrics
	BatchSize          *prometheus.HistogramVec
	BatchDuration      *prometheus.HistogramVec
	
	// Cache metrics (if using caching layer)
	CacheHits          prometheus.Counter
	CacheMisses        prometheus.Counter
	
	// Internal state
	mu                 sync.RWMutex
	activeConnections  int64
	idleConnections    int64
}

// NewNeo4jMetrics creates a new Neo4j metrics collector.
func NewNeo4jMetrics(namespace string) *Neo4jMetrics {
	metrics := &Neo4jMetrics{
		// Query metrics
		QueryDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "neo4j_query_duration_seconds",
				Help:      "Duration of Neo4j queries in seconds",
				Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~16s
			},
			[]string{"query_type", "status"},
		),
		QueryCount: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_query_total",
				Help:      "Total number of Neo4j queries executed",
			},
			[]string{"query_type", "status"},
		),
		QueryErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_query_errors_total",
				Help:      "Total number of Neo4j query errors",
			},
			[]string{"query_type", "error_type"},
		),
		QueryComplexity: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "neo4j_query_complexity_score",
				Help:      "Complexity score of Neo4j queries",
				Buckets:   prometheus.LinearBuckets(0, 20, 10), // 0-200 in steps of 20
			},
			[]string{"query_type"},
		),
		
		// Connection metrics
		ConnectionsActive: promauto.NewGauge(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "neo4j_connections_active",
				Help:      "Number of active Neo4j connections",
			},
		),
		ConnectionsIdle: promauto.NewGauge(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "neo4j_connections_idle",
				Help:      "Number of idle Neo4j connections",
			},
		),
		ConnectionErrors: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_connection_errors_total",
				Help:      "Total number of Neo4j connection errors",
			},
		),
		ConnectionDuration: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "neo4j_connection_duration_seconds",
				Help:      "Duration to establish Neo4j connection",
				Buckets:   prometheus.ExponentialBuckets(0.01, 2, 10), // 10ms to ~10s
			},
		),
		
		// Transaction metrics
		TransactionDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "neo4j_transaction_duration_seconds",
				Help:      "Duration of Neo4j transactions",
				Buckets:   prometheus.ExponentialBuckets(0.01, 2, 12), // 10ms to ~40s
			},
			[]string{"type", "status"},
		),
		TransactionCount: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_transaction_total",
				Help:      "Total number of Neo4j transactions",
			},
			[]string{"type", "status"},
		),
		TransactionErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_transaction_errors_total",
				Help:      "Total number of Neo4j transaction errors",
			},
			[]string{"type", "error_type"},
		),
		
		// Batch metrics
		BatchSize: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "neo4j_batch_size",
				Help:      "Size of batched operations",
				Buckets:   prometheus.ExponentialBuckets(10, 2, 10), // 10 to ~10k
			},
			[]string{"operation_type"},
		),
		BatchDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "neo4j_batch_duration_seconds",
				Help:      "Duration of batched operations",
				Buckets:   prometheus.ExponentialBuckets(0.1, 2, 12), // 100ms to ~6min
			},
			[]string{"operation_type"},
		),
		
		// Cache metrics
		CacheHits: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_cache_hits_total",
				Help:      "Total number of cache hits",
			},
		),
		CacheMisses: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "neo4j_cache_misses_total",
				Help:      "Total number of cache misses",
			},
		),
	}
	
	return metrics
}

// QueryTimer is a helper to time query execution.
type QueryTimer struct {
	metrics   *Neo4jMetrics
	queryType string
	start     time.Time
	complexity int
}

// RecordQuery records a query execution with timing and status.
func (m *Neo4jMetrics) RecordQuery(queryType string, duration time.Duration, err error, complexity int) {
	status := "success"
	if err != nil {
		status = "error"
		errorType := "unknown"
		if err != nil {
			errorType = classifyError(err)
		}
		m.QueryErrors.WithLabelValues(queryType, errorType).Inc()
	}
	
	m.QueryDuration.WithLabelValues(queryType, status).Observe(duration.Seconds())
	m.QueryCount.WithLabelValues(queryType, status).Inc()
	m.QueryComplexity.WithLabelValues(queryType).Observe(float64(complexity))
}

// StartQuery returns a timer for measuring query duration.
func (m *Neo4jMetrics) StartQuery(queryType string, complexity int) *QueryTimer {
	return &QueryTimer{
		metrics:    m,
		queryType:  queryType,
		start:      time.Now(),
		complexity: complexity,
	}
}

// End completes the query timer and records metrics.
func (qt *QueryTimer) End(err error) {
	duration := time.Since(qt.start)
	qt.metrics.RecordQuery(qt.queryType, duration, err, qt.complexity)
}

// RecordConnection records connection pool metrics.
func (m *Neo4jMetrics) RecordConnection(active, idle int64) {
	m.mu.Lock()
	m.activeConnections = active
	m.idleConnections = idle
	m.mu.Unlock()
	
	m.ConnectionsActive.Set(float64(active))
	m.ConnectionsIdle.Set(float64(idle))
}

// RecordConnectionError records a connection error.
func (m *Neo4jMetrics) RecordConnectionError() {
	m.ConnectionErrors.Inc()
}

// RecordConnectionTime records time to establish connection.
func (m *Neo4jMetrics) RecordConnectionTime(duration time.Duration) {
	m.ConnectionDuration.Observe(duration.Seconds())
}

// TransactionTimer is a helper to time transaction execution.
type TransactionTimer struct {
	metrics  *Neo4jMetrics
	txType   string
	start    time.Time
}

// StartTransaction returns a timer for measuring transaction duration.
func (m *Neo4jMetrics) StartTransaction(txType string) *TransactionTimer {
	return &TransactionTimer{
		metrics: m,
		txType:  txType,
		start:   time.Now(),
	}
}

// End completes the transaction timer and records metrics.
func (tt *TransactionTimer) End(err error) {
	duration := time.Since(tt.start)
	status := "success"
	if err != nil {
		status = "error"
		errorType := classifyError(err)
		tt.metrics.TransactionErrors.WithLabelValues(tt.txType, errorType).Inc()
	}
	
	tt.metrics.TransactionDuration.WithLabelValues(tt.txType, status).Observe(duration.Seconds())
	tt.metrics.TransactionCount.WithLabelValues(tt.txType, status).Inc()
}

// RecordBatch records batch operation metrics.
func (m *Neo4jMetrics) RecordBatch(operationType string, size int, duration time.Duration) {
	m.BatchSize.WithLabelValues(operationType).Observe(float64(size))
	m.BatchDuration.WithLabelValues(operationType).Observe(duration.Seconds())
}

// RecordCacheHit records a cache hit.
func (m *Neo4jMetrics) RecordCacheHit() {
	m.CacheHits.Inc()
}

// RecordCacheMiss records a cache miss.
func (m *Neo4jMetrics) RecordCacheMiss() {
	m.CacheMisses.Inc()
}

// GetConnectionPoolStats returns current connection pool statistics.
func (m *Neo4jMetrics) GetConnectionPoolStats() (active, idle int64) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeConnections, m.idleConnections
}

// classifyError attempts to classify the error type for metrics.
func classifyError(err error) string {
	if err == nil {
		return "none"
	}
	
	errStr := err.Error()
	
	// Connection errors
	if contains(errStr, "connection") || contains(errStr, "timeout") {
		return "connection"
	}
	
	// Authentication errors
	if contains(errStr, "auth") || contains(errStr, "unauthorized") {
		return "authentication"
	}
	
	// Syntax errors
	if contains(errStr, "syntax") || contains(errStr, "invalid") {
		return "syntax"
	}
	
	// Constraint violations
	if contains(errStr, "constraint") || contains(errStr, "unique") {
		return "constraint"
	}
	
	// Transaction errors
	if contains(errStr, "transaction") || contains(errStr, "deadlock") {
		return "transaction"
	}
	
	return "other"
}

// contains checks if a string contains a substring (case-insensitive).
func contains(s, substr string) bool {
	return len(s) >= len(substr) && containsIgnoreCase(s, substr)
}

func containsIgnoreCase(s, substr string) bool {
	s = toLower(s)
	substr = toLower(substr)
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := range s {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

// MetricsCollector provides a context-aware metrics collector.
type MetricsCollector struct {
	metrics *Neo4jMetrics
}

// NewMetricsCollector creates a new metrics collector.
func NewMetricsCollector(metrics *Neo4jMetrics) *MetricsCollector {
	return &MetricsCollector{
		metrics: metrics,
	}
}

// WithQuery wraps a query execution with metrics collection.
func (mc *MetricsCollector) WithQuery(ctx context.Context, queryType string, complexity int, fn func() error) error {
	timer := mc.metrics.StartQuery(queryType, complexity)
	err := fn()
	timer.End(err)
	return err
}

// WithTransaction wraps a transaction with metrics collection.
func (mc *MetricsCollector) WithTransaction(ctx context.Context, txType string, fn func() error) error {
	timer := mc.metrics.StartTransaction(txType)
	err := fn()
	timer.End(err)
	return err
}
