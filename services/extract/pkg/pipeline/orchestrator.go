package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Orchestrator orchestrates the code-to-knowledge graph pipeline
type Orchestrator struct {
	config     *Config
	extractURL string
	logger     *log.Logger
	httpClient *http.Client
}

// NewOrchestrator creates a new pipeline orchestrator
func NewOrchestrator(config *Config, logger *log.Logger) *Orchestrator {
	return &Orchestrator{
		config:     config,
		extractURL: config.ExtractURL,
		logger:     logger,
		httpClient: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
}

// Process runs the complete pipeline
func (o *Orchestrator) Process(ctx context.Context) error {
	if o.config.ProjectConfig == nil {
		return fmt.Errorf("project config is required")
	}

	project := o.config.ProjectConfig.Project
	o.logger.Printf("Processing project: %s (ID: %s, System: %s)", project.Name, project.ID, project.SystemID)

	// Build request payload
	payload := o.buildRequestPayload()

	// Submit to extract service
	response, err := o.submitToExtractService(ctx, payload)
	if err != nil {
		return fmt.Errorf("submit to extract service: %w", err)
	}

	o.logger.Printf("Pipeline completed: %d nodes, %d edges", response.Nodes, response.Edges)
	return nil
}

// buildRequestPayload builds the request payload from config
func (o *Orchestrator) buildRequestPayload() map[string]interface{} {
	project := o.config.ProjectConfig.Project
	payload := map[string]interface{}{
		"project_id": project.ID,
		"system_id":  project.SystemID,
	}

	// Add file sources
	if len(o.config.ProjectConfig.Sources.Files) > 0 {
		payload["json_tables"] = o.config.ProjectConfig.Sources.Files
	}

	// Add Git repositories
	if len(o.config.ProjectConfig.Sources.GitRepositories) > 0 {
		gitRepos := make([]map[string]interface{}, len(o.config.ProjectConfig.Sources.GitRepositories))
		for i, repo := range o.config.ProjectConfig.Sources.GitRepositories {
			gitRepos[i] = map[string]interface{}{
				"url":          repo.URL,
				"type":         repo.Type,
				"branch":       repo.Branch,
				"tag":          repo.Tag,
				"commit":       repo.Commit,
				"file_patterns": repo.FilePatterns,
				"auth": map[string]interface{}{
					"type":      repo.Auth.Type,
					"token":     repo.Auth.Token,
					"key_path":  repo.Auth.KeyPath,
					"username":  repo.Auth.Username,
					"password":  repo.Auth.Password,
				},
			}
		}
		payload["git_repositories"] = gitRepos
	}

	// Add AI config
	if o.config.ProjectConfig.AI.Enabled {
		payload["ai_enabled"] = true
		payload["ai_model"] = o.config.ProjectConfig.AI.Model
	}

	return payload
}

// submitToExtractService submits the payload to the extract service
func (o *Orchestrator) submitToExtractService(ctx context.Context, payload map[string]interface{}) (*extractResponse, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	endpoint := strings.TrimRight(o.extractURL, "/") + "/knowledge-graph"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("submit request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("extract service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result extractResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

type extractResponse struct {
	Nodes int `json:"nodes"`
	Edges int `json:"edges"`
}

// ProcessFiles processes files from the filesystem
func (o *Orchestrator) ProcessFiles(ctx context.Context, filePaths []string) error {
	// Expand file patterns
	var expandedFiles []string
	for _, pattern := range filePaths {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			o.logger.Printf("Warning: invalid file pattern %s: %v", pattern, err)
			continue
		}
		if len(matches) == 0 {
			// Try as direct file path
			if _, err := os.Stat(pattern); err == nil {
				expandedFiles = append(expandedFiles, pattern)
			}
		} else {
			expandedFiles = append(expandedFiles, matches...)
		}
	}

	if len(expandedFiles) == 0 {
		return fmt.Errorf("no files found matching patterns")
	}

	o.logger.Printf("Found %d files to process", len(expandedFiles))

	// Update config with expanded files
	if o.config.ProjectConfig == nil {
		o.config.ProjectConfig = &ProjectConfig{}
	}
	o.config.ProjectConfig.Sources.Files = expandedFiles

	return o.Process(ctx)
}

