package api

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gpu_orchestrator_http_request_duration_seconds",
			Help: "Duration of HTTP requests in seconds",
		},
		[]string{"method", "endpoint", "status"},
	)

	httpRequestCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_orchestrator_http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	// GPU allocation metrics
	GPUAllocationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_orchestrator_allocations_total",
			Help: "Total number of GPU allocations",
		},
		[]string{"service", "status"},
	)

	GPUAllocationsActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_allocations_active",
			Help: "Number of currently active GPU allocations",
		},
	)

	GPUQueueDepth = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_queue_depth",
			Help: "Number of requests waiting in queue",
		},
	)

	GPUQueueWaitTime = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "gpu_orchestrator_queue_wait_seconds",
			Help:    "Time requests spend waiting in queue",
			Buckets: []float64{1, 5, 10, 30, 60, 120, 300, 600},
		},
	)

	GPUAvailable = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpus_available",
			Help: "Number of available GPUs",
		},
	)

	GPUTotal = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpus_total",
			Help: "Total number of GPUs",
		},
	)

	GPUUtilization = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpu_utilization_percent",
			Help: "GPU utilization percentage",
		},
		[]string{"gpu_id", "gpu_name"},
	)

	GPUMemoryUsed = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpu_memory_used_mb",
			Help: "GPU memory used in MB",
		},
		[]string{"gpu_id", "gpu_name"},
	)

	GPUMemoryTotal = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpu_memory_total_mb",
			Help: "GPU total memory in MB",
		},
		[]string{"gpu_id", "gpu_name"},
	)

	GPUTemperature = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpu_temperature_celsius",
			Help: "GPU temperature in Celsius",
		},
		[]string{"gpu_id", "gpu_name"},
	)

	GPUPowerDraw = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_orchestrator_gpu_power_draw_watts",
			Help: "GPU power draw in watts",
		},
		[]string{"gpu_id", "gpu_name"},
	)
)

// MetricsMiddleware wraps HTTP handlers with metrics collection
func MetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start).Seconds()
		status := http.StatusText(wrapped.statusCode)

		httpRequestDuration.WithLabelValues(r.Method, r.URL.Path, status).Observe(duration)
		httpRequestCount.WithLabelValues(r.Method, r.URL.Path, status).Inc()
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

