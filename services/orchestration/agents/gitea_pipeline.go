package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

// GiteaPipelineConfig configures the Gitea pipeline.
type GiteaPipelineConfig struct {
	ExtractServiceURL string
	GiteaURL          string
	GiteaToken        string
	Logger            *log.Logger
}

// GiteaPipeline handles Gitea repository operations in workflows.
type GiteaPipeline struct {
	config       GiteaPipelineConfig
	httpClient   *http.Client
}

// NewGiteaPipeline creates a new Gitea pipeline.
func NewGiteaPipeline(config GiteaPipelineConfig) (*GiteaPipeline, error) {
	if config.ExtractServiceURL == "" {
		config.ExtractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
		if config.ExtractServiceURL == "" {
			config.ExtractServiceURL = "http://localhost:8081"
		}
	}

	if config.GiteaURL == "" {
		config.GiteaURL = os.Getenv("GITEA_URL")
	}
	if config.GiteaToken == "" {
		config.GiteaToken = os.Getenv("GITEA_TOKEN")
	}

	return &GiteaPipeline{
		config: config,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}, nil
}

// CreateRepositoryRequest represents a request to create a repository.
type CreateRepositoryRequest struct {
	Owner       string `json:"owner,omitempty"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Private     bool   `json:"private,omitempty"`
	AutoInit    bool   `json:"auto_init,omitempty"`
	Readme      string `json:"readme,omitempty"`
}

// CloneRepositoryRequest represents a request to clone a repository.
type CloneRepositoryRequest struct {
	Owner  string `json:"owner"`
	Repo   string `json:"repo"`
	Branch string `json:"branch,omitempty"`
	Path   string `json:"path,omitempty"`
}

// ProcessRepositoryRequest represents a request to process a repository.
type ProcessRepositoryRequest struct {
	Owner       string            `json:"owner"`
	Repo        string            `json:"repo"`
	ProjectID   string            `json:"project_id"`
	SystemID    string            `json:"system_id,omitempty"`
	Branch      string            `json:"branch,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

// GiteaWorkflowResult represents the result of a Gitea workflow operation.
type GiteaWorkflowResult struct {
	Success      bool                   `json:"success"`
	Operation    string                 `json:"operation"`
	Repository   map[string]interface{} `json:"repository,omitempty"`
	ClonePath    string                 `json:"clone_path,omitempty"`
	ProcessingID string                 `json:"processing_id,omitempty"`
	Error        string                 `json:"error,omitempty"`
	Duration     time.Duration          `json:"duration"`
}

// CreateRepository creates a new Gitea repository.
func (gp *GiteaPipeline) CreateRepository(ctx context.Context, req CreateRepositoryRequest) (*GiteaWorkflowResult, error) {
	start := time.Now()
	
	url := fmt.Sprintf("%s/gitea/repositories", gp.config.ExtractServiceURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Gitea-URL", gp.config.GiteaURL)
	httpReq.Header.Set("X-Gitea-Token", gp.config.GiteaToken)

	resp, err := gp.httpClient.Do(httpReq)
	if err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "create_repository",
			Error:     err.Error(),
			Duration:  time.Since(start),
		}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		var errorBody map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&errorBody)
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "create_repository",
			Error:     fmt.Sprintf("HTTP %d: %v", resp.StatusCode, errorBody),
			Duration:  time.Since(start),
		}, fmt.Errorf("create repository failed with status %d", resp.StatusCode)
	}

	var repo map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&repo); err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "create_repository",
			Error:     err.Error(),
			Duration:  time.Since(start),
		}, err
	}

	return &GiteaWorkflowResult{
		Success:    true,
		Operation:  "create_repository",
		Repository: repo,
		Duration:   time.Since(start),
	}, nil
}

// CloneRepository clones a Gitea repository for processing.
func (gp *GiteaPipeline) CloneRepository(ctx context.Context, req CloneRepositoryRequest) (*GiteaWorkflowResult, error) {
	start := time.Now()

	url := fmt.Sprintf("%s/gitea/repositories/%s/%s/clone",
		gp.config.ExtractServiceURL,
		req.Owner,
		req.Repo,
	)

	cloneReq := map[string]interface{}{
		"branch": req.Branch,
	}
	if req.Path != "" {
		cloneReq["path"] = req.Path
	}

	jsonData, err := json.Marshal(cloneReq)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Gitea-URL", gp.config.GiteaURL)
	httpReq.Header.Set("X-Gitea-Token", gp.config.GiteaToken)

	resp, err := gp.httpClient.Do(httpReq)
	if err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "clone_repository",
			Error:     err.Error(),
			Duration:  time.Since(start),
		}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorBody map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&errorBody)
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "clone_repository",
			Error:     fmt.Sprintf("HTTP %d: %v", resp.StatusCode, errorBody),
			Duration:  time.Since(start),
		}, fmt.Errorf("clone repository failed with status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "clone_repository",
			Error:     err.Error(),
			Duration:  time.Since(start),
		}, err
	}

	clonePath := ""
	if path, ok := result["clone_path"].(string); ok {
		clonePath = path
	}

	return &GiteaWorkflowResult{
		Success:   true,
		Operation: "clone_repository",
		ClonePath: clonePath,
		Repository: result,
		Duration:  time.Since(start),
	}, nil
}

// ProcessRepository processes a Gitea repository through the extract service.
func (gp *GiteaPipeline) ProcessRepository(ctx context.Context, req ProcessRepositoryRequest) (*GiteaWorkflowResult, error) {
	start := time.Now()

	// First clone the repository
	cloneReq := CloneRepositoryRequest{
		Owner:  req.Owner,
		Repo:   req.Repo,
		Branch: req.Branch,
	}
	if req.Branch == "" {
		cloneReq.Branch = "main"
	}

	cloneResult, err := gp.CloneRepository(ctx, cloneReq)
	if err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "process_repository",
			Error:     fmt.Sprintf("clone failed: %v", err),
			Duration:  time.Since(start),
		}, err
	}

	// Then process through extract service
	extractURL := fmt.Sprintf("%s/graph", gp.config.ExtractServiceURL)
	
	extractReq := map[string]interface{}{
		"project_id": req.ProjectID,
		"system_id":  req.SystemID,
		"gitea_storage": map[string]interface{}{
			"enabled":    true,
			"gitea_url":  gp.config.GiteaURL,
			"gitea_token": gp.config.GiteaToken,
			"owner":      req.Owner,
			"repo_name":  req.Repo,
			"branch":     req.Branch,
		},
	}

	if req.Options != nil {
		for k, v := range req.Options {
			extractReq[k] = v
		}
	}

	jsonData, err := json.Marshal(extractReq)
	if err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "process_repository",
			Error:     fmt.Sprintf("marshal extract request: %v", err),
			Duration:  time.Since(start),
		}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, extractURL, bytes.NewReader(jsonData))
	if err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "process_repository",
			Error:     fmt.Sprintf("create extract request: %v", err),
			Duration:  time.Since(start),
		}, err
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := gp.httpClient.Do(httpReq)
	if err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "process_repository",
			Error:     fmt.Sprintf("extract request failed: %v", err),
			Duration:  time.Since(start),
		}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorBody map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&errorBody)
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "process_repository",
			Error:     fmt.Sprintf("extract failed with status %d: %v", resp.StatusCode, errorBody),
			Duration:  time.Since(start),
		}, fmt.Errorf("extract failed with status %d", resp.StatusCode)
	}

	var extractResult map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&extractResult); err != nil {
		return &GiteaWorkflowResult{
			Success:   false,
			Operation: "process_repository",
			Error:     fmt.Sprintf("decode extract result: %v", err),
			Duration:  time.Since(start),
		}, err
	}

	processingID := ""
	if id, ok := extractResult["processing_id"].(string); ok {
		processingID = id
	}

	return &GiteaWorkflowResult{
		Success:      true,
		Operation:    "process_repository",
		ClonePath:    cloneResult.ClonePath,
		ProcessingID: processingID,
		Repository:   extractResult,
		Duration:     time.Since(start),
	}, nil
}

