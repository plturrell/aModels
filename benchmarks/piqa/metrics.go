package piqa

import (
	"sync"
	"sync/atomic"
	"time"
)

// MetricsCollector collects performance metrics for PIQA
type MetricsCollector struct {
	// Retrieval metrics
	totalQueries      atomic.Int64
	successfulQueries atomic.Int64
	failedQueries     atomic.Int64
	totalLatencyNs    atomic.Int64
	cacheHits         atomic.Int64
	cacheMisses       atomic.Int64

	// Embedding metrics
	embeddingsGenerated atomic.Int64
	embeddingErrors     atomic.Int64
	avgEmbeddingTimeNs  atomic.Int64

	// Storage metrics
	vectorsSaved  atomic.Int64
	vectorsLoaded atomic.Int64
	storageErrors atomic.Int64

	// Memory metrics
	currentMemoryMB atomic.Int64
	peakMemoryMB    atomic.Int64

	// Detailed latency tracking
	latencyBuckets map[string]*LatencyBucket
	mu             sync.RWMutex

	startTime time.Time
}

// LatencyBucket tracks latency distribution
type LatencyBucket struct {
	count   atomic.Int64
	totalNs atomic.Int64
	minNs   atomic.Int64
	maxNs   atomic.Int64
}

// MetricsSnapshot represents a point-in-time metrics snapshot
type MetricsSnapshot struct {
	Timestamp time.Time `json:"timestamp"`
	Uptime    string    `json:"uptime"`

	// Query metrics
	TotalQueries      int64   `json:"total_queries"`
	SuccessfulQueries int64   `json:"successful_queries"`
	FailedQueries     int64   `json:"failed_queries"`
	SuccessRate       float64 `json:"success_rate"`
	AvgLatencyMs      float64 `json:"avg_latency_ms"`
	QueriesPerSecond  float64 `json:"queries_per_second"`

	// Cache metrics
	CacheHits    int64   `json:"cache_hits"`
	CacheMisses  int64   `json:"cache_misses"`
	CacheHitRate float64 `json:"cache_hit_rate"`

	// Embedding metrics
	EmbeddingsGenerated int64   `json:"embeddings_generated"`
	EmbeddingErrors     int64   `json:"embedding_errors"`
	AvgEmbeddingTimeMs  float64 `json:"avg_embedding_time_ms"`

	// Storage metrics
	VectorsSaved  int64 `json:"vectors_saved"`
	VectorsLoaded int64 `json:"vectors_loaded"`
	StorageErrors int64 `json:"storage_errors"`

	// Memory metrics
	CurrentMemoryMB int64 `json:"current_memory_mb"`
	PeakMemoryMB    int64 `json:"peak_memory_mb"`

	// Latency distribution
	LatencyP50Ms float64 `json:"latency_p50_ms"`
	LatencyP95Ms float64 `json:"latency_p95_ms"`
	LatencyP99Ms float64 `json:"latency_p99_ms"`
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		latencyBuckets: make(map[string]*LatencyBucket),
		startTime:      time.Now(),
	}
}

// RecordQuery records a query execution
func (m *MetricsCollector) RecordQuery(success bool, latencyNs int64) {
	m.totalQueries.Add(1)
	if success {
		m.successfulQueries.Add(1)
	} else {
		m.failedQueries.Add(1)
	}
	m.totalLatencyNs.Add(latencyNs)

	// Update latency bucket
	m.recordLatency("query", latencyNs)
}

// RecordCacheHit records a cache hit
func (m *MetricsCollector) RecordCacheHit() {
	m.cacheHits.Add(1)
}

// RecordCacheMiss records a cache miss
func (m *MetricsCollector) RecordCacheMiss() {
	m.cacheMisses.Add(1)
}

// RecordEmbedding records embedding generation
func (m *MetricsCollector) RecordEmbedding(success bool, timeNs int64) {
	if success {
		m.embeddingsGenerated.Add(1)
		// Update running average
		current := m.avgEmbeddingTimeNs.Load()
		count := m.embeddingsGenerated.Load()
		newAvg := (current*(count-1) + timeNs) / count
		m.avgEmbeddingTimeNs.Store(newAvg)
	} else {
		m.embeddingErrors.Add(1)
	}
}

// RecordVectorSave records vector storage operation
func (m *MetricsCollector) RecordVectorSave(success bool) {
	if success {
		m.vectorsSaved.Add(1)
	} else {
		m.storageErrors.Add(1)
	}
}

// RecordVectorLoad records vector load operation
func (m *MetricsCollector) RecordVectorLoad(success bool) {
	if success {
		m.vectorsLoaded.Add(1)
	} else {
		m.storageErrors.Add(1)
	}
}

// UpdateMemory updates memory usage metrics
func (m *MetricsCollector) UpdateMemory(currentMB int64) {
	m.currentMemoryMB.Store(currentMB)

	// Update peak if necessary
	for {
		peak := m.peakMemoryMB.Load()
		if currentMB <= peak {
			break
		}
		if m.peakMemoryMB.CompareAndSwap(peak, currentMB) {
			break
		}
	}
}

// recordLatency records latency in a bucket
func (m *MetricsCollector) recordLatency(operation string, latencyNs int64) {
	m.mu.Lock()
	bucket, ok := m.latencyBuckets[operation]
	if !ok {
		bucket = &LatencyBucket{}
		bucket.minNs.Store(latencyNs)
		bucket.maxNs.Store(latencyNs)
		m.latencyBuckets[operation] = bucket
	}
	m.mu.Unlock()

	bucket.count.Add(1)
	bucket.totalNs.Add(latencyNs)

	// Update min
	for {
		min := bucket.minNs.Load()
		if latencyNs >= min {
			break
		}
		if bucket.minNs.CompareAndSwap(min, latencyNs) {
			break
		}
	}

	// Update max
	for {
		max := bucket.maxNs.Load()
		if latencyNs <= max {
			break
		}
		if bucket.maxNs.CompareAndSwap(max, latencyNs) {
			break
		}
	}
}

// Snapshot returns current metrics snapshot
func (m *MetricsCollector) Snapshot() MetricsSnapshot {
	totalQueries := m.totalQueries.Load()
	successQueries := m.successfulQueries.Load()
	totalLatency := m.totalLatencyNs.Load()
	cacheHits := m.cacheHits.Load()
	cacheMisses := m.cacheMisses.Load()
	uptime := time.Since(m.startTime)

	snapshot := MetricsSnapshot{
		Timestamp:           time.Now(),
		Uptime:              uptime.String(),
		TotalQueries:        totalQueries,
		SuccessfulQueries:   successQueries,
		FailedQueries:       m.failedQueries.Load(),
		CacheHits:           cacheHits,
		CacheMisses:         cacheMisses,
		EmbeddingsGenerated: m.embeddingsGenerated.Load(),
		EmbeddingErrors:     m.embeddingErrors.Load(),
		VectorsSaved:        m.vectorsSaved.Load(),
		VectorsLoaded:       m.vectorsLoaded.Load(),
		StorageErrors:       m.storageErrors.Load(),
		CurrentMemoryMB:     m.currentMemoryMB.Load(),
		PeakMemoryMB:        m.peakMemoryMB.Load(),
	}

	// Calculate rates
	if totalQueries > 0 {
		snapshot.SuccessRate = float64(successQueries) / float64(totalQueries) * 100
		snapshot.AvgLatencyMs = float64(totalLatency) / float64(totalQueries) / 1e6
	}

	totalCache := cacheHits + cacheMisses
	if totalCache > 0 {
		snapshot.CacheHitRate = float64(cacheHits) / float64(totalCache) * 100
	}

	if uptime.Seconds() > 0 {
		snapshot.QueriesPerSecond = float64(totalQueries) / uptime.Seconds()
	}

	avgEmbTime := m.avgEmbeddingTimeNs.Load()
	if avgEmbTime > 0 {
		snapshot.AvgEmbeddingTimeMs = float64(avgEmbTime) / 1e6
	}

	// Calculate percentiles (simplified - would use proper histogram in production)
	m.mu.RLock()
	if bucket, ok := m.latencyBuckets["query"]; ok {
		count := bucket.count.Load()
		if count > 0 {
			avg := bucket.totalNs.Load() / count
			snapshot.LatencyP50Ms = float64(avg) / 1e6
			snapshot.LatencyP95Ms = float64(bucket.maxNs.Load()) / 1e6 * 0.95
			snapshot.LatencyP99Ms = float64(bucket.maxNs.Load()) / 1e6 * 0.99
		}
	}
	m.mu.RUnlock()

	return snapshot
}

// Reset resets all metrics
func (m *MetricsCollector) Reset() {
	m.totalQueries.Store(0)
	m.successfulQueries.Store(0)
	m.failedQueries.Store(0)
	m.totalLatencyNs.Store(0)
	m.cacheHits.Store(0)
	m.cacheMisses.Store(0)
	m.embeddingsGenerated.Store(0)
	m.embeddingErrors.Store(0)
	m.avgEmbeddingTimeNs.Store(0)
	m.vectorsSaved.Store(0)
	m.vectorsLoaded.Store(0)
	m.storageErrors.Store(0)
	m.currentMemoryMB.Store(0)
	m.peakMemoryMB.Store(0)

	m.mu.Lock()
	m.latencyBuckets = make(map[string]*LatencyBucket)
	m.mu.Unlock()

	m.startTime = time.Now()
}
