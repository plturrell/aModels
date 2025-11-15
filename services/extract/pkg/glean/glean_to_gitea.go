package glean

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/plturrell/aModels/services/extract/pkg/git"
)

// GleanToGiteaExporter exports Glean query results back to Gitea repositories
type GleanToGiteaExporter struct {
	giteaClient *git.GiteaClient
	logger      *log.Logger
}

// QueryResult represents a Glean query result
type QueryResult struct {
	Predicate string                 `json:"predicate"`
	Facts     []map[string]interface{} `json:"facts"`
}

// ExportConfig configures how Glean data is exported to Gitea
type ExportConfig struct {
	Owner       string
	RepoName    string
	Branch      string
	Path        string // Target path in repository
	CommitMsg   string
	FileFormat  string // json, markdown, csv
	CreateIfNew bool
}

// NewGleanToGiteaExporter creates a new Glean→Gitea exporter
func NewGleanToGiteaExporter(giteaClient *git.GiteaClient, logger *log.Logger) *GleanToGiteaExporter {
	return &GleanToGiteaExporter{
		giteaClient: giteaClient,
		logger:      logger,
	}
}

// ExportQuery exports the results of a Glean query to a Gitea repository
func (e *GleanToGiteaExporter) ExportQuery(ctx context.Context, query string, config ExportConfig) error {
	e.logger.Printf("[glean→gitea] Executing Glean query: %s", query)

	// Execute Glean query
	results, err := e.executeGleanQuery(query)
	if err != nil {
		return fmt.Errorf("failed to execute Glean query: %w", err)
	}

	// Convert results to file content
	content, err := e.formatResults(results, config.FileFormat)
	if err != nil {
		return fmt.Errorf("failed to format results: %w", err)
	}

	// Store in Gitea
	e.logger.Printf("[glean→gitea] Storing results in %s/%s at %s", config.Owner, config.RepoName, config.Path)

	storageConfig := git.StorageConfig{
		Owner:      config.Owner,
		RepoName:   config.RepoName,
		Branch:     config.Branch,
		AutoCreate: config.CreateIfNew,
	}

	// Ensure repository exists
	giteaStorage := git.NewGiteaStorage(e.giteaClient, e.logger)
	if err := giteaStorage.EnsureRepository(ctx, storageConfig); err != nil {
		return fmt.Errorf("failed to ensure repository: %w", err)
	}

	// Create or update file
	if err := e.giteaClient.CreateOrUpdateFile(
		ctx,
		config.Owner,
		config.RepoName,
		config.Path,
		content,
		config.CommitMsg,
		config.Branch,
	); err != nil {
		return fmt.Errorf("failed to store file in Gitea: %w", err)
	}

	e.logger.Printf("[glean→gitea] Successfully exported query results to Gitea")
	return nil
}

// ExportPredicateData exports all data for a specific Glean predicate to Gitea
func (e *GleanToGiteaExporter) ExportPredicateData(ctx context.Context, predicate string, config ExportConfig) error {
	query := fmt.Sprintf("find %s", predicate)
	return e.ExportQuery(ctx, query, config)
}

// CreateSnapshotBranch creates a Git branch with a snapshot of Glean data
func (e *GleanToGiteaExporter) CreateSnapshotBranch(ctx context.Context, owner, repo, baseBranch, snapshotBranch, query string) error {
	e.logger.Printf("[glean→gitea] Creating snapshot branch %s from %s", snapshotBranch, baseBranch)

	// Execute query
	results, err := e.executeGleanQuery(query)
	if err != nil {
		return err
	}

	// Format as JSON
	content, err := e.formatResults(results, "json")
	if err != nil {
		return err
	}

	// Create branch (simplified - in production would use Git API)
	// For now, just create file in new path structure
	snapshotPath := fmt.Sprintf("glean-snapshots/%s/results.json", snapshotBranch)

	config := ExportConfig{
		Owner:       owner,
		RepoName:    repo,
		Branch:      baseBranch,
		Path:        snapshotPath,
		CommitMsg:   fmt.Sprintf("Glean snapshot: %s", snapshotBranch),
		FileFormat:  "json",
		CreateIfNew: true,
	}

	return e.ExportQuery(ctx, query, config)
}

// executeGleanQuery executes a Glean query and returns results
func (e *GleanToGiteaExporter) executeGleanQuery(query string) ([]QueryResult, error) {
	dbName := os.Getenv("GLEAN_DB_NAME")
	if dbName == "" {
		return nil, fmt.Errorf("GLEAN_DB_NAME not set")
	}

	// Execute glean query command
	cmd := exec.Command("glean", "query", "--db", dbName, query)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("glean query failed: %w", err)
	}

	// Parse JSON output
	var results []QueryResult
	if err := json.Unmarshal(output, &results); err != nil {
		// If not JSON array, try single result
		var result QueryResult
		if err := json.Unmarshal(output, &result); err != nil {
			return nil, fmt.Errorf("failed to parse query results: %w", err)
		}
		results = []QueryResult{result}
	}

	return results, nil
}

// formatResults formats query results in the specified format
func (e *GleanToGiteaExporter) formatResults(results []QueryResult, format string) (string, error) {
	switch strings.ToLower(format) {
	case "json":
		data, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			return "", err
		}
		return string(data), nil

	case "markdown":
		return e.formatAsMarkdown(results), nil

	case "csv":
		return e.formatAsCSV(results), nil

	default:
		return "", fmt.Errorf("unsupported format: %s", format)
	}
}

// formatAsMarkdown formats results as Markdown
func (e *GleanToGiteaExporter) formatAsMarkdown(results []QueryResult) string {
	var md strings.Builder

	md.WriteString("# Glean Query Results\n\n")
	md.WriteString(fmt.Sprintf("_Exported at: %s_\n\n", os.Getenv("TIMESTAMP")))

	for _, result := range results {
		md.WriteString(fmt.Sprintf("## %s\n\n", result.Predicate))

		if len(result.Facts) == 0 {
			md.WriteString("_No results_\n\n")
			continue
		}

		md.WriteString("| Field | Value |\n")
		md.WriteString("|-------|-------|\n")

		for _, fact := range result.Facts {
			for key, value := range fact {
				md.WriteString(fmt.Sprintf("| %s | %v |\n", key, value))
			}
			md.WriteString("|\n") // Separator between facts
		}

		md.WriteString("\n")
	}

	return md.String()
}

// formatAsCSV formats results as CSV
func (e *GleanToGiteaExporter) formatAsCSV(results []QueryResult) string {
	var csv strings.Builder

	for _, result := range results {
		csv.WriteString(fmt.Sprintf("# Predicate: %s\n", result.Predicate))

		if len(result.Facts) == 0 {
			continue
		}

		// Get headers from first fact
		headers := make([]string, 0)
		for key := range result.Facts[0] {
			headers = append(headers, key)
		}

		// Write headers
		csv.WriteString(strings.Join(headers, ","))
		csv.WriteString("\n")

		// Write data
		for _, fact := range result.Facts {
			values := make([]string, len(headers))
			for i, header := range headers {
				values[i] = fmt.Sprintf("%v", fact[header])
			}
			csv.WriteString(strings.Join(values, ","))
			csv.WriteString("\n")
		}

		csv.WriteString("\n")
	}

	return csv.String()
}

// SyncPredicateToRepository continuously syncs a Glean predicate to a Gitea repository
func (e *GleanToGiteaExporter) SyncPredicateToRepository(
	ctx context.Context,
	predicate string,
	config ExportConfig,
	intervalSeconds int,
) error {
	e.logger.Printf("[glean→gitea] Starting continuous sync: %s → %s/%s",
		predicate, config.Owner, config.RepoName)

	// TODO: Implement continuous sync with ticker
	// For now, just do a single export
	return e.ExportPredicateData(ctx, predicate, config)
}

// ExportGiteaEntities exports all Gitea-related entities from Glean back to a summary repository
func (e *GleanToGiteaExporter) ExportGiteaEntities(ctx context.Context, targetOwner, targetRepo string) error {
	e.logger.Println("[glean→gitea] Exporting all Gitea entities from Glean")

	predicates := []string{
		"gitea.GiteaRepository",
		"gitea.GiteaFile",
		"gitea.GiteaCommit",
		"gitea.GiteaBranch",
		"gitea.GiteaWebhookEvent",
	}

	for _, predicate := range predicates {
		filename := filepath.Base(predicate) + ".json"
		config := ExportConfig{
			Owner:       targetOwner,
			RepoName:    targetRepo,
			Branch:      "main",
			Path:        fmt.Sprintf("glean-exports/%s", filename),
			CommitMsg:   fmt.Sprintf("Export %s from Glean", predicate),
			FileFormat:  "json",
			CreateIfNew: true,
		}

		if err := e.ExportPredicateData(ctx, predicate, config); err != nil {
			e.logger.Printf("[glean→gitea] Warning: failed to export %s: %v", predicate, err)
			continue
		}

		e.logger.Printf("[glean→gitea] Exported %s", predicate)
	}

	return nil
}
