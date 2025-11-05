package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// RealTimeGleanExporter handles real-time synchronization of graph data to Glean.
// It automatically ingests batches into Glean Catalog as they are created.
type RealTimeGleanExporter struct {
	gleanPersistence *GleanPersistence
	dbName           string
	schemaPath       string
	logger           *log.Logger
	mu               sync.Mutex
	exportQueue      chan *ExportTask
	workerCount      int
	enabled          bool
	lastExportTime   time.Time
	exportStats      *ExportStats
}

// ExportTask represents a graph export task for real-time processing.
type ExportTask struct {
	Nodes     []Node
	Edges     []Edge
	BatchFile string
	CreatedAt time.Time
	ProjectID string
	SystemID  string
}

// ExportStats tracks real-time export statistics.
type ExportStats struct {
	TotalExports      uint64
	SuccessfulExports uint64
	FailedExports     uint64
	LastSuccessTime   time.Time
	LastErrorTime     time.Time
	LastError         string
	mu                sync.RWMutex
}

// NewRealTimeGleanExporter creates a new real-time Glean exporter.
func NewRealTimeGleanExporter(
	gleanPersistence *GleanPersistence,
	dbName string,
	schemaPath string,
	logger *log.Logger,
) *RealTimeGleanExporter {
	enabled := strings.ToLower(strings.TrimSpace(os.Getenv("GLEAN_REALTIME_ENABLE"))) == "true"
	workerCount := 2 // Default: 2 concurrent export workers

	if raw := strings.TrimSpace(os.Getenv("GLEAN_REALTIME_WORKERS")); raw != "" {
		if count, err := parseUint(raw); err == nil && count > 0 {
			workerCount = int(count)
		}
	}

	if dbName == "" {
		dbName = strings.TrimSpace(os.Getenv("GLEAN_DB_NAME"))
	}

	if schemaPath == "" {
		schemaPath = gleanPersistence.schemaPath
	}

	exporter := &RealTimeGleanExporter{
		gleanPersistence: gleanPersistence,
		dbName:           dbName,
		schemaPath:       schemaPath,
		logger:           logger,
		exportQueue:      make(chan *ExportTask, 100), // Buffer up to 100 exports
		workerCount:      workerCount,
		enabled:          enabled && dbName != "",
		exportStats:      &ExportStats{},
	}

	if exporter.enabled {
		// Start worker goroutines for processing exports
		for i := 0; i < workerCount; i++ {
			go exporter.exportWorker(i)
		}

		if logger != nil {
			logger.Printf("Real-time Glean export enabled (workers: %d, db: %s)", workerCount, dbName)
		}
	} else {
		if logger != nil {
			logger.Printf("Real-time Glean export disabled (enable with GLEAN_REALTIME_ENABLE=true and GLEAN_DB_NAME)")
		}
	}

	return exporter
}

// ExportGraph exports graph data to Glean in real-time.
// This method is non-blocking - it queues the export for async processing.
func (r *RealTimeGleanExporter) ExportGraph(ctx context.Context, nodes []Node, edges []Edge, projectID, systemID string) error {
	if !r.enabled {
		return nil // Real-time export disabled, skip silently
	}

	// Create export task
	task := &ExportTask{
		Nodes:     nodes,
		Edges:     edges,
		CreatedAt: time.Now(),
		ProjectID: projectID,
		SystemID:  systemID,
	}

	// Queue for async processing (non-blocking)
	select {
	case r.exportQueue <- task:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	default:
		// Queue full - log warning but don't block
		if r.logger != nil {
			r.logger.Printf("WARNING: Glean export queue full, dropping export (nodes: %d, edges: %d)", len(nodes), len(edges))
		}
		return fmt.Errorf("export queue full")
	}
}

// exportWorker processes export tasks from the queue.
func (r *RealTimeGleanExporter) exportWorker(workerID int) {
	for task := range r.exportQueue {
		if err := r.processExport(task); err != nil {
			if r.logger != nil {
				r.logger.Printf("Worker %d: Export failed: %v", workerID, err)
			}
			r.recordExportFailure(err)
		} else {
			r.recordExportSuccess()
			if r.logger != nil {
				r.logger.Printf("Worker %d: Export successful (nodes: %d, edges: %d)", workerID, len(task.Nodes), len(task.Edges))
			}
		}
	}
}

// processExport processes a single export task.
func (r *RealTimeGleanExporter) processExport(task *ExportTask) error {
	// Track export start time for incremental tracking
	exportStartTime := time.Now()

	// Save graph to batch file using GleanPersistence
	// This creates a new batch file with timestamp and sequence number
	if err := r.gleanPersistence.SaveGraph(task.Nodes, task.Edges); err != nil {
		return fmt.Errorf("save graph to batch: %w", err)
	}

	// Small delay to ensure file is fully written
	time.Sleep(100 * time.Millisecond)

	// Get the most recent batch file (just created by SaveGraph)
	// Uses incremental tracking to find the file created after exportStartTime
	batchFile, err := r.getLatestBatchFileAfter(exportStartTime.Add(-1 * time.Second))
	if err != nil {
		return fmt.Errorf("get latest batch file: %w", err)
	}

	if batchFile == "" {
		return fmt.Errorf("no batch file created (check Glean export directory)")
	}

	task.BatchFile = batchFile

	// Ingest batch into Glean using `glean write` command
	if err := r.ingestBatchToGlean(batchFile); err != nil {
		return fmt.Errorf("ingest to Glean: %w", err)
	}

	// Update last export time for incremental tracking
	r.mu.Lock()
	r.lastExportTime = time.Now()
	r.mu.Unlock()

	return nil
}

// getLatestBatchFileAfter finds the most recent batch file created after the given time.
func (r *RealTimeGleanExporter) getLatestBatchFileAfter(afterTime time.Time) (string, error) {
	exportDir := r.gleanPersistence.exportDir
	entries, err := os.ReadDir(exportDir)
	if err != nil {
		return "", fmt.Errorf("read export directory: %w", err)
	}

	var latestFile string
	var latestTime time.Time

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		// Only consider files created after the specified time
		if info.ModTime().After(afterTime) && info.ModTime().After(latestTime) {
			latestTime = info.ModTime()
			latestFile = filepath.Join(exportDir, entry.Name())
		}
	}

	return latestFile, nil
}

// ingestBatchToGlean ingests a batch file into Glean using the `glean write` command.
func (r *RealTimeGleanExporter) ingestBatchToGlean(batchFile string) error {
	if r.dbName == "" {
		return fmt.Errorf("Glean DB name not configured")
	}

	// Build glean write command
	args := []string{"write", "--db", r.dbName}

	// Add schema path if provided
	if r.schemaPath != "" && fileExists(r.schemaPath) {
		args = append(args, "--schema", r.schemaPath)
	}

	args = append(args, batchFile)

	// Execute glean write command
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "glean", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("glean write command failed: %w", err)
	}

	return nil
}

// recordExportSuccess records a successful export.
func (r *RealTimeGleanExporter) recordExportSuccess() {
	r.exportStats.mu.Lock()
	defer r.exportStats.mu.Unlock()

	r.exportStats.TotalExports++
	r.exportStats.SuccessfulExports++
	r.exportStats.LastSuccessTime = time.Now()
}

// recordExportFailure records a failed export.
func (r *RealTimeGleanExporter) recordExportFailure(err error) {
	r.exportStats.mu.Lock()
	defer r.exportStats.mu.Unlock()

	r.exportStats.TotalExports++
	r.exportStats.FailedExports++
	r.exportStats.LastErrorTime = time.Now()
	r.exportStats.LastError = err.Error()
}

// GetStats returns current export statistics.
func (r *RealTimeGleanExporter) GetStats() ExportStats {
	r.exportStats.mu.RLock()
	defer r.exportStats.mu.RUnlock()

	return ExportStats{
		TotalExports:      r.exportStats.TotalExports,
		SuccessfulExports: r.exportStats.SuccessfulExports,
		FailedExports:     r.exportStats.FailedExports,
		LastSuccessTime:   r.exportStats.LastSuccessTime,
		LastErrorTime:     r.exportStats.LastErrorTime,
		LastError:         r.exportStats.LastError,
	}
}

// Shutdown gracefully shuts down the real-time exporter.
func (r *RealTimeGleanExporter) Shutdown(ctx context.Context) error {
	if !r.enabled {
		return nil
	}

	// Close export queue
	close(r.exportQueue)

	// Wait for all queued exports to complete
	deadline := time.Now().Add(30 * time.Second)
	for len(r.exportQueue) > 0 {
		if time.Now().After(deadline) {
			return fmt.Errorf("shutdown timeout: %d exports still queued", len(r.exportQueue))
		}
		time.Sleep(100 * time.Millisecond)
	}

	if r.logger != nil {
		stats := r.GetStats()
		r.logger.Printf("Real-time Glean exporter shutdown (total: %d, success: %d, failed: %d)",
			stats.TotalExports, stats.SuccessfulExports, stats.FailedExports)
	}

	return nil
}

// Helper functions

func parseUint(s string) (uint, error) {
	var result uint
	_, err := fmt.Sscanf(s, "%d", &result)
	return result, err
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
