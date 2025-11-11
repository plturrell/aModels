package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// PerplexityAdvancedPipeline provides next-level features:
// - Real-time streaming processing
// - Advanced analytics and metrics
// - Intelligent query optimization
// - Batch processing
// - Advanced caching
// - Performance monitoring
// - Auto-scaling capabilities
type PerplexityAdvancedPipeline struct {
	basePipeline        *PerplexityPipeline
	metricsCollector    *PerplexityMetricsCollector
	queryOptimizer      *QueryOptimizer
	cache               *AdvancedCache
	streamProcessor     *StreamProcessor
	performanceMonitor  *PerformanceMonitor
	autoScaler          *AutoScaler
	logger              *log.Logger
	httpClient          *http.Client
}

// PerplexityAdvancedConfig configures the advanced pipeline.
type PerplexityAdvancedConfig struct {
	PipelineConfig      PerplexityPipelineConfig
	EnableStreaming     bool
	EnableCaching       bool
	EnableAutoScaling   bool
	CacheTTL            time.Duration
	MaxConcurrentQueries int
	Logger              *log.Logger
}

// NewPerplexityAdvancedPipeline creates an advanced pipeline with next-level features.
func NewPerplexityAdvancedPipeline(config PerplexityAdvancedConfig) (*PerplexityAdvancedPipeline, error) {
	// Create base pipeline
	basePipeline, err := NewPerplexityPipeline(config.PipelineConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base pipeline: %w", err)
	}

	// Create metrics collector
	metricsCollector := NewPerplexityMetricsCollector(config.Logger)

	// Create query optimizer
	queryOptimizer := NewQueryOptimizer(metricsCollector, config.Logger)

	// Create advanced cache
	var cache *AdvancedCache
	if config.EnableCaching {
		cache = NewAdvancedCache(config.CacheTTL, config.Logger)
	}

	// Create stream processor
	var streamProcessor *StreamProcessor
	if config.EnableStreaming {
		streamProcessor = NewStreamProcessor(config.Logger)
	}

	// Create performance monitor
	performanceMonitor := NewPerformanceMonitor(config.Logger)

	// Create auto-scaler
	var autoScaler *AutoScaler
	if config.EnableAutoScaling {
		autoScaler = NewAutoScaler(performanceMonitor, config.Logger)
	}

	return &PerplexityAdvancedPipeline{
		basePipeline:       basePipeline,
		metricsCollector:   metricsCollector,
		queryOptimizer:     queryOptimizer,
		cache:              cache,
		streamProcessor:    streamProcessor,
		performanceMonitor: performanceMonitor,
		autoScaler:         autoScaler,
		logger:             config.Logger,
		// Use connection pooling for better performance (Priority 1)
		httpClient: &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				MaxConnsPerHost:     50,
			},
			Timeout: 300 * time.Second,
		},
	}, nil
}

// ProcessDocumentsStreaming processes documents with real-time streaming updates.
func (pap *PerplexityAdvancedPipeline) ProcessDocumentsStreaming(
	ctx context.Context,
	query map[string]interface{},
	streamChan chan<- StreamEvent,
) error {
	if pap.streamProcessor == nil {
		// Fallback to non-streaming
		return pap.basePipeline.ProcessDocuments(ctx, query)
	}

	return pap.streamProcessor.ProcessWithStreaming(ctx, query, streamChan, pap.basePipeline)
}

// ProcessDocumentsBatch processes multiple queries in batch with optimization.
func (pap *PerplexityAdvancedPipeline) ProcessDocumentsBatch(
	ctx context.Context,
	queries []map[string]interface{},
) (*BatchProcessingResult, error) {
	startTime := time.Now()

	// Optimize queries
	optimizedQueries := pap.queryOptimizer.OptimizeBatch(queries)

	// Check cache for cached results
	cachedResults := make(map[int]map[string]interface{})
	uncachedQueries := make([]map[string]interface{}, 0)
	uncachedIndices := make([]int, 0)

	for i, query := range optimizedQueries {
		if pap.cache != nil {
			if cached, found := pap.cache.Get(query); found {
				cachedResults[i] = cached
				continue
			}
		}
		uncachedQueries = append(uncachedQueries, query)
		uncachedIndices = append(uncachedIndices, i)
	}

	// Process uncached queries in parallel
	results := make([]map[string]interface{}, len(optimizedQueries))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for idx, queryIdx := range uncachedIndices {
		wg.Add(1)
		go func(i, origIdx int) {
			defer wg.Done()

			queryStart := time.Now()
			err := pap.basePipeline.ProcessDocuments(ctx, uncachedQueries[i])
			duration := time.Since(queryStart)

			mu.Lock()
			results[origIdx] = map[string]interface{}{
				"query":   uncachedQueries[i],
				"success": err == nil,
				"error": func() string {
					if err != nil {
						return err.Error()
					}
					return ""
				}(),
				"duration": duration,
			}
			mu.Unlock()

			// Cache successful results
			if err == nil && pap.cache != nil {
				pap.cache.Set(uncachedQueries[i], results[origIdx])
			}

			// Record metrics
			pap.metricsCollector.RecordQuery(uncachedQueries[i], duration, err == nil)
		}(idx, queryIdx)
	}

	// Fill in cached results
	for i, cached := range cachedResults {
		results[i] = cached
		pap.metricsCollector.RecordCacheHit()
	}

	wg.Wait()

	totalDuration := time.Since(startTime)

	// Record batch metrics
	pap.metricsCollector.RecordBatch(len(queries), len(cachedResults), totalDuration)

	return &BatchProcessingResult{
		TotalQueries:    len(queries),
		CachedQueries:  len(cachedResults),
		ProcessedQueries: len(uncachedQueries),
		Results:        results,
		TotalDuration:  totalDuration,
		Metrics:        pap.metricsCollector.GetMetrics(),
	}, nil
}

// GetAnalytics returns comprehensive analytics and insights.
func (pap *PerplexityAdvancedPipeline) GetAnalytics() *AnalyticsReport {
	return &AnalyticsReport{
		Metrics:        pap.metricsCollector.GetMetrics(),
		Performance:    pap.performanceMonitor.GetReport(),
		QueryPatterns:  pap.queryOptimizer.GetPatterns(),
		CacheStats:     pap.cache.GetStats(),
		AutoScaleState: pap.autoScaler.GetState(),
	}
}

// OptimizeQuery intelligently optimizes a query for better results.
func (pap *PerplexityAdvancedPipeline) OptimizeQuery(query map[string]interface{}) map[string]interface{} {
	return pap.queryOptimizer.Optimize(query)
}

// PerplexityMetricsCollector collects comprehensive metrics.
type PerplexityMetricsCollector struct {
	queryCount      int64
	cacheHits       int64
	cacheMisses     int64
	totalLatency    time.Duration
	errorCount      int64
	successCount    int64
	queryPatterns   map[string]int64
	latencyHistory  []time.Duration
	mu              sync.RWMutex
	logger          *log.Logger
	startTime       time.Time
}

// NewPerplexityMetricsCollector creates a new metrics collector.
func NewPerplexityMetricsCollector(logger *log.Logger) *PerplexityMetricsCollector {
	return &PerplexityMetricsCollector{
		queryPatterns:  make(map[string]int64),
		latencyHistory: make([]time.Duration, 0, 1000),
		logger:         logger,
		startTime:      time.Now(),
	}
}

// RecordQuery records a query execution.
func (pmc *PerplexityMetricsCollector) RecordQuery(query map[string]interface{}, duration time.Duration, success bool) {
	pmc.mu.Lock()
	defer pmc.mu.Unlock()

	pmc.queryCount++
	pmc.totalLatency += duration
	pmc.latencyHistory = append(pmc.latencyHistory, duration)
	if len(pmc.latencyHistory) > 1000 {
		pmc.latencyHistory = pmc.latencyHistory[1:]
	}

	if success {
		pmc.successCount++
	} else {
		pmc.errorCount++
	}

	// Track query patterns
	queryStr, _ := query["query"].(string)
	if queryStr != "" {
		pmc.queryPatterns[queryStr]++
	}
}

// RecordCacheHit records a cache hit.
func (pmc *PerplexityMetricsCollector) RecordCacheHit() {
	pmc.mu.Lock()
	defer pmc.mu.Unlock()
	pmc.cacheHits++
}

// RecordCacheMiss records a cache miss.
func (pmc *PerplexityMetricsCollector) RecordCacheMiss() {
	pmc.mu.Lock()
	defer pmc.mu.Unlock()
	pmc.cacheMisses++
}

// RecordBatch records batch processing metrics.
func (pmc *PerplexityMetricsCollector) RecordBatch(total, cached int, duration time.Duration) {
	pmc.mu.Lock()
	defer pmc.mu.Unlock()
	// Additional batch-specific metrics can be added here
}

// GetMetrics returns current metrics.
func (pmc *PerplexityMetricsCollector) GetMetrics() map[string]interface{} {
	pmc.mu.RLock()
	defer pmc.mu.RUnlock()

	avgLatency := time.Duration(0)
	if pmc.queryCount > 0 {
		avgLatency = pmc.totalLatency / time.Duration(pmc.queryCount)
	}

	successRate := 0.0
	if pmc.queryCount > 0 {
		successRate = float64(pmc.successCount) / float64(pmc.queryCount) * 100
	}

	cacheHitRate := 0.0
	totalCacheOps := pmc.cacheHits + pmc.cacheMisses
	if totalCacheOps > 0 {
		cacheHitRate = float64(pmc.cacheHits) / float64(totalCacheOps) * 100
	}

	return map[string]interface{}{
		"query_count":      pmc.queryCount,
		"success_count":    pmc.successCount,
		"error_count":     pmc.errorCount,
		"success_rate":     successRate,
		"avg_latency":     avgLatency.String(),
		"cache_hits":      pmc.cacheHits,
		"cache_misses":    pmc.cacheMisses,
		"cache_hit_rate":  cacheHitRate,
		"uptime":          time.Since(pmc.startTime).String(),
		"query_patterns":  pmc.queryPatterns,
	}
}

// QueryOptimizer optimizes queries for better results.
type QueryOptimizer struct {
	metricsCollector *PerplexityMetricsCollector
	patterns         map[string]*QueryPattern
	logger           *log.Logger
	mu               sync.RWMutex
}

// NewQueryOptimizer creates a new query optimizer.
func NewQueryOptimizer(metricsCollector *PerplexityMetricsCollector, logger *log.Logger) *QueryOptimizer {
	return &QueryOptimizer{
		metricsCollector: metricsCollector,
		patterns:         make(map[string]*QueryPattern),
		logger:           logger,
	}
}

// Optimize optimizes a single query.
func (qo *QueryOptimizer) Optimize(query map[string]interface{}) map[string]interface{} {
	optimized := make(map[string]interface{})
	for k, v := range query {
		optimized[k] = v
	}

	// Add intelligent defaults
	if _, ok := optimized["model"]; !ok {
		optimized["model"] = "sonar" // Best default model
	}

	if _, ok := optimized["limit"]; !ok {
		optimized["limit"] = 10 // Optimal default limit
	}

	// Enhance query string if present
	if queryStr, ok := query["query"].(string); ok && queryStr != "" {
		// Add context hints based on patterns
		optimized["query"] = qo.enhanceQuery(queryStr)
	}

	return optimized
}

// OptimizeBatch optimizes multiple queries.
func (qo *QueryOptimizer) OptimizeBatch(queries []map[string]interface{}) []map[string]interface{} {
	optimized := make([]map[string]interface{}, len(queries))
	for i, query := range queries {
		optimized[i] = qo.Optimize(query)
	}
	return optimized
}

// enhanceQuery enhances a query string with intelligent hints.
func (qo *QueryOptimizer) enhanceQuery(query string) string {
	// Add domain-specific enhancements based on learned patterns
	// This is a simplified version - in production, would use ML models
	return query
}

// GetPatterns returns learned query patterns.
func (qo *QueryOptimizer) GetPatterns() map[string]*QueryPattern {
	qo.mu.RLock()
	defer qo.mu.RUnlock()
	return qo.patterns
}

// QueryPattern represents a learned query pattern.
type QueryPattern struct {
	Pattern     string
	Frequency   int64
	AvgLatency  time.Duration
	SuccessRate float64
	LastUsed    time.Time
}

// AdvancedCache provides intelligent caching.
type AdvancedCache struct {
	cache    map[string]*CacheEntry
	ttl      time.Duration
	logger   *log.Logger
	mu       sync.RWMutex
	hits     int64
	misses   int64
}

// NewAdvancedCache creates a new advanced cache.
func NewAdvancedCache(ttl time.Duration, logger *log.Logger) *AdvancedCache {
	return &AdvancedCache{
		cache:  make(map[string]*CacheEntry),
		ttl:    ttl,
		logger: logger,
	}
}

// Get retrieves a value from cache.
func (ac *AdvancedCache) Get(query map[string]interface{}) (map[string]interface{}, bool) {
	ac.mu.RLock()
	defer ac.mu.RUnlock()

	key := ac.generateKey(query)
	entry, found := ac.cache[key]
	if !found {
		ac.misses++
		return nil, false
	}

	if time.Since(entry.CreatedAt) > ac.ttl {
		ac.misses++
		return nil, false
	}

	ac.hits++
	return entry.Value, true
}

// Set stores a value in cache.
func (ac *AdvancedCache) Set(query map[string]interface{}, value map[string]interface{}) {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	key := ac.generateKey(query)
	ac.cache[key] = &CacheEntry{
		Value:     value,
		CreatedAt: time.Now(),
	}
}

// GetStats returns cache statistics.
func (ac *AdvancedCache) GetStats() map[string]interface{} {
	ac.mu.RLock()
	defer ac.mu.RUnlock()

	total := ac.hits + ac.misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(ac.hits) / float64(total) * 100
	}

	return map[string]interface{}{
		"size":      len(ac.cache),
		"hits":      ac.hits,
		"misses":    ac.misses,
		"hit_rate":  hitRate,
		"ttl":       ac.ttl.String(),
	}
}

// generateKey generates a cache key from query.
func (ac *AdvancedCache) generateKey(query map[string]interface{}) string {
	// Create a deterministic key from query
	data, _ := json.Marshal(query)
	return fmt.Sprintf("%x", data)
}

// CacheEntry represents a cache entry.
type CacheEntry struct {
	Value     map[string]interface{}
	CreatedAt time.Time
}

// StreamProcessor handles real-time streaming.
type StreamProcessor struct {
	logger *log.Logger
}

// NewStreamProcessor creates a new stream processor.
func NewStreamProcessor(logger *log.Logger) *StreamProcessor {
	return &StreamProcessor{logger: logger}
}

// ProcessWithStreaming processes documents with streaming updates.
func (sp *StreamProcessor) ProcessWithStreaming(
	ctx context.Context,
	query map[string]interface{},
	streamChan chan<- StreamEvent,
	pipeline *PerplexityPipeline,
) error {
	// Send start event
	streamChan <- StreamEvent{
		Type:    "start",
		Message: "Starting document processing",
		Time:    time.Now(),
	}

	// Process documents with progress updates
	err := pipeline.ProcessDocumentsWithCallback(ctx, query, func(doc map[string]interface{}) error {
		streamChan <- StreamEvent{
			Type:    "progress",
			Message: fmt.Sprintf("Processing document: %v", doc["id"]),
			Data:    doc,
			Time:    time.Now(),
		}
		return nil
	})

	if err != nil {
		streamChan <- StreamEvent{
			Type:    "error",
			Message: err.Error(),
			Time:    time.Now(),
		}
		return err
	}

	streamChan <- StreamEvent{
		Type:    "complete",
		Message: "Document processing completed",
		Time:    time.Now(),
	}

	return nil
}

// StreamEvent represents a streaming event.
type StreamEvent struct {
	Type    string                 `json:"type"`
	Message string                 `json:"message"`
	Data    map[string]interface{} `json:"data,omitempty"`
	Time    time.Time              `json:"time"`
}

// PerformanceMonitor monitors performance metrics.
type PerformanceMonitor struct {
	metrics map[string]*OperationMetrics
	logger  *log.Logger
	mu      sync.RWMutex
}

// NewPerformanceMonitor creates a new performance monitor.
func NewPerformanceMonitor(logger *log.Logger) *PerformanceMonitor {
	return &PerformanceMonitor{
		metrics: make(map[string]*OperationMetrics),
		logger:  logger,
	}
}

// RecordOperation records an operation's performance.
func (pm *PerformanceMonitor) RecordOperation(operation string, duration time.Duration, success bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.metrics[operation] == nil {
		pm.metrics[operation] = &OperationMetrics{
			Operation: operation,
		}
	}

	metric := pm.metrics[operation]
	metric.Count++
	metric.TotalDuration += duration
	if success {
		metric.SuccessCount++
	} else {
		metric.ErrorCount++
	}

	// Update average
	metric.AvgDuration = metric.TotalDuration / time.Duration(metric.Count)
}

// GetReport returns performance report.
func (pm *PerformanceMonitor) GetReport() map[string]interface{} {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	report := make(map[string]interface{})
	for op, metric := range pm.metrics {
		report[op] = map[string]interface{}{
			"count":         metric.Count,
			"avg_duration":  metric.AvgDuration.String(),
			"success_count": metric.SuccessCount,
			"error_count":   metric.ErrorCount,
			"success_rate":  float64(metric.SuccessCount) / float64(metric.Count) * 100,
		}
	}
	return report
}

// OperationMetrics tracks metrics for an operation.
type OperationMetrics struct {
	Operation     string
	Count         int64
	TotalDuration time.Duration
	AvgDuration   time.Duration
	SuccessCount  int64
	ErrorCount    int64
}

// AutoScaler provides auto-scaling capabilities.
type AutoScaler struct {
	performanceMonitor *PerformanceMonitor
	logger              *log.Logger
	currentScale        int
	targetScale         int
	mu                  sync.RWMutex
}

// NewAutoScaler creates a new auto-scaler.
func NewAutoScaler(performanceMonitor *PerformanceMonitor, logger *log.Logger) *AutoScaler {
	return &AutoScaler{
		performanceMonitor: performanceMonitor,
		logger:              logger,
		currentScale:       1,
		targetScale:         1,
	}
}

// EvaluateScale evaluates if scaling is needed.
func (as *AutoScaler) EvaluateScale() int {
	as.mu.Lock()
	defer as.mu.Unlock()

	// Simple scaling logic based on performance
	// In production, would use more sophisticated algorithms
	report := as.performanceMonitor.GetReport()
	
	// Check if we need to scale up
	for _, metrics := range report {
		if m, ok := metrics.(map[string]interface{}); ok {
			if avgDur, ok := m["avg_duration"].(string); ok {
				// Parse duration and decide on scaling
				// Simplified logic
				as.targetScale = 2 // Example: scale to 2 workers
			}
		}
	}

	return as.targetScale
}

// GetState returns current auto-scaling state.
func (as *AutoScaler) GetState() map[string]interface{} {
	as.mu.RLock()
	defer as.mu.RUnlock()

	return map[string]interface{}{
		"current_scale": as.currentScale,
		"target_scale":  as.targetScale,
	}
}

// BatchProcessingResult represents batch processing results.
type BatchProcessingResult struct {
	TotalQueries     int
	CachedQueries    int
	ProcessedQueries int
	Results          []map[string]interface{}
	TotalDuration    time.Duration
	Metrics          map[string]interface{}
}

// AnalyticsReport provides comprehensive analytics.
type AnalyticsReport struct {
	Metrics        map[string]interface{}
	Performance    map[string]interface{}
	QueryPatterns  map[string]*QueryPattern
	CacheStats     map[string]interface{}
	AutoScaleState map[string]interface{}
}

