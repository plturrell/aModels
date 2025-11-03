package glove

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

// MetricsCollector collects and exports training metrics
type MetricsCollector struct {
	// Training metrics
	trainingIterations int64
	trainingCost       float64
	trainingSpeed      float64
	vocabularySize     int64
	cooccurrenceCount  int64

	// Query metrics
	vectorQueries     int64
	similarityQueries int64
	avgQueryLatencyMs float64

	// MCTS metrics
	mctsIterations       int64
	mctsBestScore        float64
	mctsConfigsEvaluated int64

	// Temporal metrics
	temporalUpdates  int64
	semanticDriftAvg float64

	// System metrics
	memoryUsageMB       float64
	dbConnectionsActive int64

	mu        sync.RWMutex
	startTime time.Time
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		startTime: time.Now(),
	}
}

// RecordTrainingIteration records a training iteration
func (mc *MetricsCollector) RecordTrainingIteration(iteration int, cost, speed float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.trainingIterations = int64(iteration)
	mc.trainingCost = cost
	mc.trainingSpeed = speed
}

// RecordVocabularySize records vocabulary size
func (mc *MetricsCollector) RecordVocabularySize(size int) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.vocabularySize = int64(size)
}

// RecordCooccurrenceCount records co-occurrence count
func (mc *MetricsCollector) RecordCooccurrenceCount(count int64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.cooccurrenceCount = count
}

// RecordVectorQuery records a vector query
func (mc *MetricsCollector) RecordVectorQuery(latencyMs float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.vectorQueries++
	// Exponential moving average
	alpha := 0.1
	mc.avgQueryLatencyMs = alpha*latencyMs + (1-alpha)*mc.avgQueryLatencyMs
}

// RecordSimilarityQuery records a similarity query
func (mc *MetricsCollector) RecordSimilarityQuery(latencyMs float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.similarityQueries++
	alpha := 0.1
	mc.avgQueryLatencyMs = alpha*latencyMs + (1-alpha)*mc.avgQueryLatencyMs
}

// RecordMCTSIteration records MCTS optimization progress
func (mc *MetricsCollector) RecordMCTSIteration(iteration int, bestScore float64, configsEvaluated int) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.mctsIterations = int64(iteration)
	mc.mctsBestScore = bestScore
	mc.mctsConfigsEvaluated = int64(configsEvaluated)
}

// RecordTemporalUpdate records a temporal model update
func (mc *MetricsCollector) RecordTemporalUpdate(drift float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.temporalUpdates++
	alpha := 0.1
	mc.semanticDriftAvg = alpha*drift + (1-alpha)*mc.semanticDriftAvg
}

// RecordMemoryUsage records memory usage
func (mc *MetricsCollector) RecordMemoryUsage(mb float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.memoryUsageMB = mb
}

// RecordDBConnections records active database connections
func (mc *MetricsCollector) RecordDBConnections(count int) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.dbConnectionsActive = int64(count)
}

// ExportPrometheus exports metrics in Prometheus format
func (mc *MetricsCollector) ExportPrometheus() string {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	uptime := time.Since(mc.startTime).Seconds()

	return fmt.Sprintf(`# HELP glove_training_iterations Total training iterations completed
# TYPE glove_training_iterations counter
glove_training_iterations %d

# HELP glove_training_cost Current training cost
# TYPE glove_training_cost gauge
glove_training_cost %.6f

# HELP glove_training_speed Training speed in samples per second
# TYPE glove_training_speed gauge
glove_training_speed %.2f

# HELP glove_vocabulary_size Number of words in vocabulary
# TYPE glove_vocabulary_size gauge
glove_vocabulary_size %d

# HELP glove_cooccurrence_count Total co-occurrence entries
# TYPE glove_cooccurrence_count gauge
glove_cooccurrence_count %d

# HELP glove_vector_queries_total Total vector queries
# TYPE glove_vector_queries_total counter
glove_vector_queries_total %d

# HELP glove_similarity_queries_total Total similarity queries
# TYPE glove_similarity_queries_total counter
glove_similarity_queries_total %d

# HELP glove_query_latency_ms Average query latency in milliseconds
# TYPE glove_query_latency_ms gauge
glove_query_latency_ms %.2f

# HELP glove_mcts_iterations MCTS optimization iterations
# TYPE glove_mcts_iterations counter
glove_mcts_iterations %d

# HELP glove_mcts_best_score Best MCTS configuration score
# TYPE glove_mcts_best_score gauge
glove_mcts_best_score %.6f

# HELP glove_mcts_configs_evaluated Total configurations evaluated
# TYPE glove_mcts_configs_evaluated counter
glove_mcts_configs_evaluated %d

# HELP glove_temporal_updates_total Total temporal model updates
# TYPE glove_temporal_updates_total counter
glove_temporal_updates_total %d

# HELP glove_semantic_drift_avg Average semantic drift
# TYPE glove_semantic_drift_avg gauge
glove_semantic_drift_avg %.6f

# HELP glove_memory_usage_mb Memory usage in megabytes
# TYPE glove_memory_usage_mb gauge
glove_memory_usage_mb %.2f

# HELP glove_db_connections_active Active database connections
# TYPE glove_db_connections_active gauge
glove_db_connections_active %d

# HELP glove_uptime_seconds Uptime in seconds
# TYPE glove_uptime_seconds counter
glove_uptime_seconds %.0f
`,
		mc.trainingIterations,
		mc.trainingCost,
		mc.trainingSpeed,
		mc.vocabularySize,
		mc.cooccurrenceCount,
		mc.vectorQueries,
		mc.similarityQueries,
		mc.avgQueryLatencyMs,
		mc.mctsIterations,
		mc.mctsBestScore,
		mc.mctsConfigsEvaluated,
		mc.temporalUpdates,
		mc.semanticDriftAvg,
		mc.memoryUsageMB,
		mc.dbConnectionsActive,
		uptime,
	)
}

// GetSnapshot returns a snapshot of current metrics
func (mc *MetricsCollector) GetSnapshot() MetricsSnapshot {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	return MetricsSnapshot{
		TrainingIterations:   mc.trainingIterations,
		TrainingCost:         mc.trainingCost,
		TrainingSpeed:        mc.trainingSpeed,
		VocabularySize:       mc.vocabularySize,
		CooccurrenceCount:    mc.cooccurrenceCount,
		VectorQueries:        mc.vectorQueries,
		SimilarityQueries:    mc.similarityQueries,
		AvgQueryLatencyMs:    mc.avgQueryLatencyMs,
		MCTSIterations:       mc.mctsIterations,
		MCTSBestScore:        mc.mctsBestScore,
		MCTSConfigsEvaluated: mc.mctsConfigsEvaluated,
		TemporalUpdates:      mc.temporalUpdates,
		SemanticDriftAvg:     mc.semanticDriftAvg,
		MemoryUsageMB:        mc.memoryUsageMB,
		DBConnectionsActive:  mc.dbConnectionsActive,
		UptimeSeconds:        time.Since(mc.startTime).Seconds(),
	}
}

// MetricsSnapshot represents a point-in-time snapshot of metrics
type MetricsSnapshot struct {
	TrainingIterations   int64
	TrainingCost         float64
	TrainingSpeed        float64
	VocabularySize       int64
	CooccurrenceCount    int64
	VectorQueries        int64
	SimilarityQueries    int64
	AvgQueryLatencyMs    float64
	MCTSIterations       int64
	MCTSBestScore        float64
	MCTSConfigsEvaluated int64
	TemporalUpdates      int64
	SemanticDriftAvg     float64
	MemoryUsageMB        float64
	DBConnectionsActive  int64
	UptimeSeconds        float64
}

// MetricsServer serves metrics over HTTP
type MetricsServer struct {
	collector *MetricsCollector
	port      int
	server    *http.Server
}

// NewMetricsServer creates a new metrics server
func NewMetricsServer(collector *MetricsCollector, port int) *MetricsServer {
	return &MetricsServer{
		collector: collector,
		port:      port,
	}
}

// Start starts the metrics server
func (ms *MetricsServer) Start() error {
	mux := http.NewServeMux()

	// Prometheus endpoint
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprint(w, ms.collector.ExportPrometheus())
	})

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"healthy","uptime_seconds":%.0f}`,
			time.Since(ms.collector.startTime).Seconds())
	})

	// Metrics snapshot endpoint (JSON)
	mux.HandleFunc("/metrics/json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		snapshot := ms.collector.GetSnapshot()
		fmt.Fprintf(w, `{
  "training_iterations": %d,
  "training_cost": %.6f,
  "training_speed": %.2f,
  "vocabulary_size": %d,
  "cooccurrence_count": %d,
  "vector_queries": %d,
  "similarity_queries": %d,
  "avg_query_latency_ms": %.2f,
  "mcts_iterations": %d,
  "mcts_best_score": %.6f,
  "mcts_configs_evaluated": %d,
  "temporal_updates": %d,
  "semantic_drift_avg": %.6f,
  "memory_usage_mb": %.2f,
  "db_connections_active": %d,
  "uptime_seconds": %.0f
}`,
			snapshot.TrainingIterations,
			snapshot.TrainingCost,
			snapshot.TrainingSpeed,
			snapshot.VocabularySize,
			snapshot.CooccurrenceCount,
			snapshot.VectorQueries,
			snapshot.SimilarityQueries,
			snapshot.AvgQueryLatencyMs,
			snapshot.MCTSIterations,
			snapshot.MCTSBestScore,
			snapshot.MCTSConfigsEvaluated,
			snapshot.TemporalUpdates,
			snapshot.SemanticDriftAvg,
			snapshot.MemoryUsageMB,
			snapshot.DBConnectionsActive,
			snapshot.UptimeSeconds,
		)
	})

	ms.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", ms.port),
		Handler: mux,
	}

	fmt.Printf("Metrics server starting on port %d\n", ms.port)
	fmt.Printf("  Prometheus: http://localhost:%d/metrics\n", ms.port)
	fmt.Printf("  Health:     http://localhost:%d/health\n", ms.port)
	fmt.Printf("  JSON:       http://localhost:%d/metrics/json\n", ms.port)

	return ms.server.ListenAndServe()
}

// Stop stops the metrics server
func (ms *MetricsServer) Stop() error {
	if ms.server != nil {
		return ms.server.Close()
	}
	return nil
}

// Global metrics collector (optional convenience)
var globalMetrics *MetricsCollector
var metricsOnce sync.Once

// GetGlobalMetrics returns the global metrics collector
func GetGlobalMetrics() *MetricsCollector {
	metricsOnce.Do(func() {
		globalMetrics = NewMetricsCollector()
	})
	return globalMetrics
}

// StartMetricsServer starts the global metrics server
func StartMetricsServer(port int) error {
	collector := GetGlobalMetrics()
	server := NewMetricsServer(collector, port)
	return server.Start()
}
