package observability

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// RegisterMetrics registers all Prometheus metrics.
// This is called automatically when the package is imported due to promauto,
// but we provide this function for explicit registration if needed.
func RegisterMetrics() {
	// Metrics are auto-registered via promauto.New* functions
	// This function exists for explicit registration if needed in the future
}

// Metrics tracks Prometheus metrics for the catalog service.
var (
	// Request metrics
	RequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "catalog_request_duration_seconds",
			Help:    "Request duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 12), // 1ms to 4s
		},
		[]string{"method", "endpoint", "status"},
	)

	RequestCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_requests_total",
			Help: "Total number of requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	RequestSize = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "catalog_request_size_bytes",
			Help:    "Request size in bytes",
			Buckets: prometheus.ExponentialBuckets(100, 10, 6), // 100B to 100MB
		},
		[]string{"method", "endpoint"},
	)

	ResponseSize = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "catalog_response_size_bytes",
			Help:    "Response size in bytes",
			Buckets: prometheus.ExponentialBuckets(100, 10, 6), // 100B to 100MB
		},
		[]string{"method", "endpoint"},
	)

	// Research metrics
	ResearchDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "catalog_research_duration_seconds",
			Help:    "Deep research duration in seconds",
			Buckets: prometheus.ExponentialBuckets(1, 2, 10), // 1s to 512s
		},
	)

	ResearchCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_research_total",
			Help: "Total number of research operations",
		},
		[]string{"status", "topic"},
	)

	// Database metrics
	Neo4jQueryDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "catalog_neo4j_query_duration_seconds",
			Help:    "Neo4j query duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms to 512ms
		},
		[]string{"query_type", "status"},
	)

	Neo4jConnectionPool = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "catalog_neo4j_connection_pool_size",
			Help: "Neo4j connection pool size",
		},
		[]string{"state"}, // "active", "idle", "waiting"
	)

	// Quality metrics
	QualityScore = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "catalog_data_element_quality_score",
			Help: "Quality score for data elements",
		},
		[]string{"element_id", "quality_level"},
	)

	// Cache metrics
	CacheHits = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_cache_hits_total",
			Help: "Total cache hits",
		},
		[]string{"cache_type"},
	)

	CacheMisses = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_cache_misses_total",
			Help: "Total cache misses",
		},
		[]string{"cache_type"},
	)

	// Data product metrics
	DataProductCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_data_products_total",
			Help: "Total number of data products created",
		},
		[]string{"status"}, // "success", "failed"
	)

	DataProductCreationDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "catalog_data_product_creation_duration_seconds",
			Help:    "Data product creation duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10), // 100ms to 51.2s
		},
	)

	// Alert metrics
	AlertsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_alerts_total",
			Help: "Total number of alerts",
		},
		[]string{"alert_type"},
	)

	// Integration metrics
	IntegrationRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "catalog_integration_request_duration_seconds",
			Help:    "Integration request duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 12), // 1ms to 4s
		},
		[]string{"service", "endpoint", "status", "correlation_id"},
	)

	IntegrationRequestCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_integration_requests_total",
			Help: "Total number of integration requests",
		},
		[]string{"service", "endpoint", "status"},
	)

	IntegrationErrorCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_integration_errors_total",
			Help: "Total number of integration errors",
		},
		[]string{"service", "endpoint", "error_type"},
	)

	CircuitBreakerState = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "catalog_circuit_breaker_state",
			Help: "Circuit breaker state (0=closed, 1=open, 2=half-open)",
		},
		[]string{"service"},
	)

	IntegrationRetryCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "catalog_integration_retries_total",
			Help: "Total number of integration retries",
		},
		[]string{"service", "endpoint"},
	)
)

// RecordRequest records metrics for an HTTP request.
func RecordRequest(method, endpoint, status string, duration time.Duration, requestSize, responseSize int64) {
	RequestDuration.WithLabelValues(method, endpoint, status).Observe(duration.Seconds())
	RequestCount.WithLabelValues(method, endpoint, status).Inc()
	if requestSize > 0 {
		RequestSize.WithLabelValues(method, endpoint).Observe(float64(requestSize))
	}
	if responseSize > 0 {
		ResponseSize.WithLabelValues(method, endpoint).Observe(float64(responseSize))
	}
}

// RecordResearch records metrics for a research operation.
func RecordResearch(status, topic string, duration time.Duration) {
	ResearchDuration.Observe(duration.Seconds())
	ResearchCount.WithLabelValues(status, topic).Inc()
}

// RecordNeo4jQuery records metrics for a Neo4j query.
func RecordNeo4jQuery(queryType, status string, duration time.Duration) {
	Neo4jQueryDuration.WithLabelValues(queryType, status).Observe(duration.Seconds())
}

// RecordCacheHit records a cache hit.
func RecordCacheHit(cacheType string) {
	CacheHits.WithLabelValues(cacheType).Inc()
}

// RecordCacheMiss records a cache miss.
func RecordCacheMiss(cacheType string) {
	CacheMisses.WithLabelValues(cacheType).Inc()
}

// RecordDataProductCreation records metrics for data product creation.
func RecordDataProductCreation(status string, duration time.Duration) {
	DataProductCount.WithLabelValues(status).Inc()
	if status == "success" {
		DataProductCreationDuration.Observe(duration.Seconds())
	}
}

// UpdateQualityScore updates the quality score gauge.
func UpdateQualityScore(elementID, qualityLevel string, score float64) {
	QualityScore.WithLabelValues(elementID, qualityLevel).Set(score)
}

// UpdateConnectionPool updates the connection pool size gauge.
func UpdateConnectionPool(state string, size int) {
	Neo4jConnectionPool.WithLabelValues(state).Set(float64(size))
}

// RecordAlert records an alert event.
func RecordAlert(alertType string, details map[string]interface{}) {
	AlertsTotal.WithLabelValues(alertType).Inc()
}

// RecordIntegrationRequest records metrics for an integration request.
func RecordIntegrationRequest(service, endpoint string, statusCode int, latency time.Duration, correlationID string) {
	status := "success"
	if statusCode >= 400 {
		status = "error"
	}
	IntegrationRequestDuration.WithLabelValues(service, endpoint, status, correlationID).Observe(latency.Seconds())
	IntegrationRequestCount.WithLabelValues(service, endpoint, status).Inc()
	
	if statusCode >= 400 {
		errorType := "client_error"
		if statusCode >= 500 {
			errorType = "server_error"
		}
		IntegrationErrorCount.WithLabelValues(service, endpoint, errorType).Inc()
	}
}

// RecordCircuitBreakerState records circuit breaker state.
func RecordCircuitBreakerState(service string, state int) {
	CircuitBreakerState.WithLabelValues(service).Set(float64(state))
}

// RecordIntegrationRetry records a retry attempt.
func RecordIntegrationRetry(service, endpoint string) {
	IntegrationRetryCount.WithLabelValues(service, endpoint).Inc()
}

