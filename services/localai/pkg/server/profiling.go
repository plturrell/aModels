package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	_ "net/http/pprof" // Import for pprof endpoints
	"runtime"
	"sort"
	"sync"
	"time"
)

// Profiler provides performance profiling and metrics
type Profiler struct {
	requestLatencies []time.Duration
	latencyMu        sync.RWMutex
	maxSamples       int
	startTime        time.Time
	requestCount     int64
	errorCount       int64
	mu               sync.RWMutex
}

// NewProfiler creates a new profiler
func NewProfiler(maxSamples int) *Profiler {
	if maxSamples <= 0 {
		maxSamples = 1000 // Default to 1000 samples
	}

	return &Profiler{
		requestLatencies: make([]time.Duration, 0, maxSamples),
		maxSamples:       maxSamples,
		startTime:        time.Now(),
	}
}

// RecordRequest records a request with its latency
func (p *Profiler) RecordRequest(latency time.Duration) {
	p.mu.Lock()
	p.requestCount++
	p.mu.Unlock()

	p.latencyMu.Lock()
	defer p.latencyMu.Unlock()

	if len(p.requestLatencies) >= p.maxSamples {
		// Remove oldest sample (FIFO)
		p.requestLatencies = p.requestLatencies[1:]
	}
	p.requestLatencies = append(p.requestLatencies, latency)
}

// RecordError records an error
func (p *Profiler) RecordError() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.errorCount++
}

// GetStats returns profiling statistics
func (p *Profiler) GetStats() map[string]interface{} {
	p.mu.RLock()
	requestCount := p.requestCount
	errorCount := p.errorCount
	uptime := time.Since(p.startTime)
	p.mu.RUnlock()

	p.latencyMu.RLock()
	latencies := make([]time.Duration, len(p.requestLatencies))
	copy(latencies, p.requestLatencies)
	p.latencyMu.RUnlock()

	// Calculate latency statistics
	var avgLatency, minLatency, maxLatency time.Duration
	var p50, p95, p99 time.Duration

	if len(latencies) > 0 {
		// Sort latencies for percentile calculation using efficient stdlib sort
		sorted := make([]time.Duration, len(latencies))
		copy(sorted, latencies)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i] < sorted[j]
		})

		minLatency = sorted[0]
		maxLatency = sorted[len(sorted)-1]

		// Calculate average
		var sum time.Duration
		for _, lat := range latencies {
			sum += lat
		}
		avgLatency = sum / time.Duration(len(latencies))

		// Calculate percentiles
		if len(sorted) > 0 {
			p50 = sorted[len(sorted)*50/100]
			if len(sorted) > 1 {
				p95 = sorted[len(sorted)*95/100]
				p99 = sorted[len(sorted)*99/100]
			}
		}
	}

	// Get memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return map[string]interface{}{
		"uptime_seconds":     uptime.Seconds(),
		"request_count":      requestCount,
		"error_count":        errorCount,
		"error_rate":        float64(errorCount) / float64(requestCount+1),
		"latency": map[string]interface{}{
			"avg_ms": avgLatency.Milliseconds(),
			"min_ms": minLatency.Milliseconds(),
			"max_ms": maxLatency.Milliseconds(),
			"p50_ms": p50.Milliseconds(),
			"p95_ms": p95.Milliseconds(),
			"p99_ms": p99.Milliseconds(),
			"samples": len(latencies),
		},
		"memory": map[string]interface{}{
			"alloc_mb":       float64(m.Alloc) / 1024 / 1024,
			"total_alloc_mb": float64(m.TotalAlloc) / 1024 / 1024,
			"sys_mb":         float64(m.Sys) / 1024 / 1024,
			"num_gc":         m.NumGC,
		},
		"goroutines": runtime.NumGoroutine(),
	}
}

// HandleProfilingStats handles the /debug/stats endpoint
func (p *Profiler) HandleProfilingStats(w http.ResponseWriter, r *http.Request) {
	stats := p.GetStats()
	w.Header().Set(HeaderContentType, ContentTypeJSON)
	json.NewEncoder(w).Encode(stats)
}

// HandlePprofRedirect redirects to pprof endpoints
func HandlePprofRedirect(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	if path == "/debug/pprof" {
		http.Redirect(w, r, "/debug/pprof/", http.StatusFound)
		return
	}

	// pprof endpoints are automatically registered by importing net/http/pprof
	// They are available at /debug/pprof/*
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "Available pprof endpoints:\n")
	fmt.Fprintf(w, "  /debug/pprof/\n")
	fmt.Fprintf(w, "  /debug/pprof/goroutine\n")
	fmt.Fprintf(w, "  /debug/pprof/heap\n")
	fmt.Fprintf(w, "  /debug/pprof/profile?seconds=30\n")
	fmt.Fprintf(w, "  /debug/pprof/trace?seconds=5\n")
}

