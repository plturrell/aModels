package glean

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/git"
)

// GiteaGleanExporter exports Gitea entities to Glean
type GiteaGleanExporter struct {
	exportDir        string
	dbName           string
	predicatePrefix  string
	enabled          bool
	logger           *log.Logger
	stats            ExportStats
	exportQueue      chan ExportTask
	stopChan         chan struct{}
}

// ExportTask represents a Gitea export task
type ExportTask struct {
	Type     string      // repository, commit, file, webhook
	Data     interface{}
	Priority int
}

// ExportStats tracks export statistics
type ExportStats struct {
	TotalExports      int
	SuccessfulExports int
	FailedExports     int
	LastSuccessTime   time.Time
	LastErrorTime     time.Time
	LastError         string
}

// NewGiteaGleanExporter creates a new Gitea→Glean exporter
func NewGiteaGleanExporter(logger *log.Logger) *GiteaGleanExporter {
	exportDir := os.Getenv("GLEAN_EXPORT_DIR")
	if exportDir == "" {
		exportDir = "./data/glean/exports/gitea"
	}

	dbName := os.Getenv("GLEAN_DB_NAME")
	enabled := os.Getenv("GLEAN_REALTIME_ENABLE") == "true" && dbName != ""

	// Create export directory
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		logger.Printf("[glean] Warning: failed to create Gitea export directory: %v", err)
	}

	exporter := &GiteaGleanExporter{
		exportDir:       exportDir,
		dbName:          dbName,
		predicatePrefix: "gitea",
		enabled:         enabled,
		logger:          logger,
		exportQueue:     make(chan ExportTask, 100),
		stopChan:        make(chan struct{}),
	}

	if enabled {
		logger.Printf("[glean] Gitea→Glean real-time export ENABLED (db: %s)", dbName)
		// Start worker pool
		workers := getEnvInt("GLEAN_REALTIME_WORKERS", 2)
		for i := 0; i < workers; i++ {
			go exporter.worker(i)
		}
	} else {
		logger.Printf("[glean] Gitea→Glean export DISABLED")
	}

	return exporter
}

// ExportRepository exports a Gitea repository to Glean
func (e *GiteaGleanExporter) ExportRepository(ctx context.Context, repo *git.Repository, projectID, systemID string) error {
	if !e.enabled {
		return nil
	}

	task := ExportTask{
		Type: "repository",
		Data: map[string]interface{}{
			"repository": repo,
			"projectId":  projectID,
			"systemId":   systemID,
		},
		Priority: 2,
	}

	select {
	case e.exportQueue <- task:
		e.logger.Printf("[glean] Queued repository export: %s", repo.FullName)
		return nil
	default:
		return fmt.Errorf("export queue full")
	}
}

// ExportWebhookEvent exports a Gitea webhook event to Glean
func (e *GiteaGleanExporter) ExportWebhookEvent(ctx context.Context, event interface{}) error {
	if !e.enabled {
		return nil
	}

	task := ExportTask{
		Type:     "webhook",
		Data:     event,
		Priority: 1, // High priority for real-time events
	}

	select {
	case e.exportQueue <- task:
		e.logger.Printf("[glean] Queued webhook event export")
		return nil
	default:
		return fmt.Errorf("export queue full")
	}
}

// ExportFiles exports Gitea files to Glean
func (e *GiteaGleanExporter) ExportFiles(ctx context.Context, files []git.ExtractedFile, repo *git.Repository, branch string) error {
	if !e.enabled {
		return nil
	}

	task := ExportTask{
		Type: "files",
		Data: map[string]interface{}{
			"files":      files,
			"repository": repo,
			"branch":     branch,
		},
		Priority: 3,
	}

	select {
	case e.exportQueue <- task:
		e.logger.Printf("[glean] Queued %d files for export", len(files))
		return nil
	default:
		return fmt.Errorf("export queue full")
	}
}

// worker processes export tasks
func (e *GiteaGleanExporter) worker(id int) {
	e.logger.Printf("[glean] Worker %d started", id)

	for {
		select {
		case task := <-e.exportQueue:
			e.processTask(task)
		case <-e.stopChan:
			e.logger.Printf("[glean] Worker %d stopped", id)
			return
		}
	}
}

// processTask processes a single export task
func (e *GiteaGleanExporter) processTask(task ExportTask) {
	e.stats.TotalExports++

	// Create batch file
	batchFile, err := e.createBatchFile(task)
	if err != nil {
		e.stats.FailedExports++
		e.stats.LastErrorTime = time.Now()
		e.stats.LastError = err.Error()
		e.logger.Printf("[glean] Failed to create batch file: %v", err)
		return
	}

	// Execute glean write
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "glean", "write", "--db", e.dbName, batchFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		e.stats.FailedExports++
		e.stats.LastErrorTime = time.Now()
		e.stats.LastError = fmt.Sprintf("%v: %s", err, output)
		e.logger.Printf("[glean] Export failed: %v\nOutput: %s", err, output)
		return
	}

	e.stats.SuccessfulExports++
	e.stats.LastSuccessTime = time.Now()
	e.logger.Printf("[glean] Successfully exported %s: %s", task.Type, batchFile)
}

// createBatchFile creates a Glean batch file from the task data
func (e *GiteaGleanExporter) createBatchFile(task ExportTask) (string, error) {
	timestamp := time.Now().Format("20060102-150405")
	filename := filepath.Join(e.exportDir, fmt.Sprintf("%s-%s.json", task.Type, timestamp))

	var facts []map[string]interface{}

	switch task.Type {
	case "repository":
		facts = e.convertRepositoryToFacts(task.Data)
	case "webhook":
		facts = e.convertWebhookToFacts(task.Data)
	case "files":
		facts = e.convertFilesToFacts(task.Data)
	default:
		return "", fmt.Errorf("unknown task type: %s", task.Type)
	}

	// Write batch file
	file, err := os.Create(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(facts); err != nil {
		return "", err
	}

	return filename, nil
}

// convertRepositoryToFacts converts a Gitea repository to Glean facts
func (e *GiteaGleanExporter) convertRepositoryToFacts(data interface{}) []map[string]interface{} {
	dataMap := data.(map[string]interface{})
	repo := dataMap["repository"].(*git.Repository)
	projectID := dataMap["projectId"].(string)
	systemID := dataMap["systemId"].(string)

	facts := []map[string]interface{}{
		{
			"predicate": "gitea.GiteaRepository",
			"facts": map[string]interface{}{
				"id":            repo.ID,
				"name":          repo.Name,
				"fullName":      repo.FullName,
				"owner":         repo.Owner.Login,
				"description":   repo.Description,
				"url":           repo.HTMLURL,
				"cloneUrl":      repo.CloneURL,
				"htmlUrl":       repo.HTMLURL,
				"private":       repo.Private,
				"defaultBranch": repo.DefaultBranch,
				"createdAt":     repo.CreatedAt.Format(time.RFC3339),
				"updatedAt":     repo.UpdatedAt.Format(time.RFC3339),
			},
		},
		{
			"predicate": "gitea.RepositoryLinkedToProject",
			"facts": map[string]interface{}{
				"repository": map[string]interface{}{
					"fullName": repo.FullName,
				},
				"projectId": projectID,
				"systemId":  systemID,
			},
		},
	}

	return facts
}

// convertWebhookToFacts converts a webhook event to Glean facts
func (e *GiteaGleanExporter) convertWebhookToFacts(data interface{}) []map[string]interface{} {
	// Type assertion for webhook payload
	eventMap, ok := data.(map[string]interface{})
	if !ok {
		e.logger.Printf("[glean] Warning: invalid webhook data type")
		return []map[string]interface{}{}
	}

	facts := []map[string]interface{}{
		{
			"predicate": "gitea.GiteaWebhookEvent",
			"facts": map[string]interface{}{
				"eventType": eventMap["action"],
				"ref":       eventMap["ref"],
				"before":    eventMap["before"],
				"after":     eventMap["after"],
				"timestamp": time.Now().Format(time.RFC3339),
			},
		},
	}

	return facts
}

// convertFilesToFacts converts Gitea files to Glean facts
func (e *GiteaGleanExporter) convertFilesToFacts(data interface{}) []map[string]interface{} {
	dataMap := data.(map[string]interface{})
	files := dataMap["files"].([]git.ExtractedFile)
	repo := dataMap["repository"].(*git.Repository)
	branch := dataMap["branch"].(string)

	var facts []map[string]interface{}

	for _, file := range files {
		facts = append(facts, map[string]interface{}{
			"predicate": "gitea.GiteaFile",
			"facts": map[string]interface{}{
				"path":        file.Path,
				"name":        filepath.Base(file.Path),
				"contentHash": file.ContentHash,
				"size":        file.Size,
				"type":        "file",
			},
		})

		// Link file to repository
		facts = append(facts, map[string]interface{}{
			"predicate": "gitea.RepositoryContainsFile",
			"facts": map[string]interface{}{
				"repository": map[string]interface{}{
					"fullName": repo.FullName,
				},
				"file": map[string]interface{}{
					"path": file.Path,
				},
			},
		})
	}

	return facts
}

// GetStats returns export statistics
func (e *GiteaGleanExporter) GetStats() ExportStats {
	return e.stats
}

// Stop gracefully stops the exporter
func (e *GiteaGleanExporter) Stop() {
	close(e.stopChan)
	close(e.exportQueue)
}

// Helper function to get integer from environment
func getEnvInt(key string, defaultValue int) int {
	val := os.Getenv(key)
	if val == "" {
		return defaultValue
	}
	var result int
	fmt.Sscanf(val, "%d", &result)
	return result
}
