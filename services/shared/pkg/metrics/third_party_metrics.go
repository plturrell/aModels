package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Connection pool metrics
	poolSizeGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "third_party_pool_size",
			Help: "Current size of connection pool",
		},
		[]string{"pool_type", "service"},
	)

	poolMaxSizeGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "third_party_pool_max_size",
			Help: "Maximum size of connection pool",
		},
		[]string{"pool_type", "service"},
	)

	poolCreatedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_pool_created_total",
			Help: "Total number of connections created",
		},
		[]string{"pool_type", "service"},
	)

	poolReusedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_pool_reused_total",
			Help: "Total number of connections reused",
		},
		[]string{"pool_type", "service"},
	)

	// Retry metrics
	retryAttemptsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_retry_attempts_total",
			Help: "Total number of retry attempts",
		},
		[]string{"library", "service", "attempt"},
	)

	retrySuccessTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_retry_success_total",
			Help: "Total number of successful retries",
		},
		[]string{"library", "service"},
	)

	// Circuit breaker metrics
	circuitBreakerState = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "third_party_circuit_breaker_state",
			Help: "Circuit breaker state (0=closed, 1=half-open, 2=open)",
		},
		[]string{"breaker_name", "service"},
	)

	circuitBreakerFailures = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_circuit_breaker_failures_total",
			Help: "Total number of circuit breaker failures",
		},
		[]string{"breaker_name", "service"},
	)

	circuitBreakerStateChanges = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_circuit_breaker_state_changes_total",
			Help: "Total number of circuit breaker state changes",
		},
		[]string{"breaker_name", "service", "from_state", "to_state"},
	)

	// Cache metrics
	cacheHitsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_cache_hits_total",
			Help: "Total number of cache hits",
		},
		[]string{"cache_type", "service"},
	)

	cacheMissesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_cache_misses_total",
			Help: "Total number of cache misses",
		},
		[]string{"cache_type", "service"},
	)

	cacheSizeGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "third_party_cache_size",
			Help: "Current size of cache",
		},
		[]string{"cache_type", "service"},
	)

	// API response time metrics
	apiResponseTime = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "third_party_api_response_time_seconds",
			Help:    "Response time for third-party API calls",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"library", "service", "operation"},
	)

	// Error rate metrics
	apiErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "third_party_api_errors_total",
			Help: "Total number of third-party API errors",
		},
		[]string{"library", "service", "operation", "error_type"},
	)

	once sync.Once
)

// UpdatePoolMetrics updates connection pool metrics.
func UpdatePoolMetrics(poolType, service string, currentSize, maxSize int64, created, reused int64) {
	poolSizeGauge.WithLabelValues(poolType, service).Set(float64(currentSize))
	poolMaxSizeGauge.WithLabelValues(poolType, service).Set(float64(maxSize))
	poolCreatedTotal.WithLabelValues(poolType, service).Add(float64(created))
	poolReusedTotal.WithLabelValues(poolType, service).Add(float64(reused))
}

// RecordRetryAttempt records a retry attempt.
func RecordRetryAttempt(library, service string, attempt int) {
	retryAttemptsTotal.WithLabelValues(library, service, string(rune(attempt))).Inc()
}

// RecordRetrySuccess records a successful retry.
func RecordRetrySuccess(library, service string) {
	retrySuccessTotal.WithLabelValues(library, service).Inc()
}

// UpdateCircuitBreakerState updates circuit breaker state metrics.
func UpdateCircuitBreakerState(breakerName, service, fromState, toState string) {
	stateValue := 0.0
	switch toState {
	case "half-open":
		stateValue = 1.0
	case "open":
		stateValue = 2.0
	}
	circuitBreakerState.WithLabelValues(breakerName, service).Set(stateValue)
	circuitBreakerStateChanges.WithLabelValues(breakerName, service, fromState, toState).Inc()
}

// RecordCircuitBreakerFailure records a circuit breaker failure.
func RecordCircuitBreakerFailure(breakerName, service string) {
	circuitBreakerFailures.WithLabelValues(breakerName, service).Inc()
}

// RecordCacheHit records a cache hit.
func RecordCacheHit(cacheType, service string) {
	cacheHitsTotal.WithLabelValues(cacheType, service).Inc()
}

// RecordCacheMiss records a cache miss.
func RecordCacheMiss(cacheType, service string) {
	cacheMissesTotal.WithLabelValues(cacheType, service).Inc()
}

// UpdateCacheSize updates cache size metric.
func UpdateCacheSize(cacheType, service string, size int64) {
	cacheSizeGauge.WithLabelValues(cacheType, service).Set(float64(size))
}

// RecordAPIResponseTime records API response time.
func RecordAPIResponseTime(library, service, operation string, durationSeconds float64) {
	apiResponseTime.WithLabelValues(library, service, operation).Observe(durationSeconds)
}

// RecordAPIError records an API error.
func RecordAPIError(library, service, operation, errorType string) {
	apiErrorsTotal.WithLabelValues(library, service, operation, errorType).Inc()
}

