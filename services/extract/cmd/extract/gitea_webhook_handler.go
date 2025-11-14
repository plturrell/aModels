package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	handlers "github.com/plturrell/aModels/services/extract/internal/handlers"
	"github.com/plturrell/aModels/services/extract/pkg/git"
)

// GiteaWebhookPayload represents a Gitea webhook push event payload
type GiteaWebhookPayload struct {
	Secret     string `json:"secret"`
	Action     string `json:"action"` // "push", "create", etc.
	Ref        string `json:"ref"`    // "refs/heads/main"
	Before     string `json:"before"` // Previous commit SHA
	After      string `json:"after"`  // New commit SHA
	CompareURL string `json:"compare_url"`
	Commits    []struct {
		ID      string `json:"id"`
		Message string `json:"message"`
		URL     string `json:"url"`
		Author  struct {
			Name     string `json:"name"`
			Email    string `json:"email"`
			Username string `json:"username"`
		} `json:"author"`
		Committer struct {
			Name     string `json:"name"`
			Email    string `json:"email"`
			Username string `json:"username"`
		} `json:"committer"`
		Added    []string `json:"added"`    // Files added
		Removed  []string `json:"removed"`  // Files removed
		Modified []string `json:"modified"` // Files modified
	} `json:"commits"`
	Repository struct {
		ID       int64  `json:"id"`
		Name     string `json:"name"`
		FullName string `json:"full_name"` // "owner/repo"
		HTMLURL  string `json:"html_url"`
		CloneURL string `json:"clone_url"`
		Owner    struct {
			Login     string `json:"login"`
			UserName  string `json:"username"`
			FullName  string `json:"full_name"`
		} `json:"owner"`
	} `json:"repository"`
	Pusher struct {
		Login string `json:"login"`
		ID    int64  `json:"id"`
	} `json:"pusher"`
	Sender struct {
		Login string `json:"login"`
		ID    int64  `json:"id"`
	} `json:"sender"`
}

// Relevant file patterns that should trigger pipeline processing
var relevantFilePatterns = []string{
	"*-config.yaml",
	"*-config.yml",
	"*.yaml",
	"*.yml",
	"*.json",
	"*.hql",
	"*.ddl",
	"*.sql",
	"*.xml",
	"*.xlsx",
	"*.docx",
}

// isRelevantFile checks if a file path matches patterns that should trigger processing
func isRelevantFile(filePath string) bool {
	filePath = strings.ToLower(filePath)
	for _, pattern := range relevantFilePatterns {
		if matched, _ := filepath.Match(pattern, filepath.Base(filePath)); matched {
			return true
		}
		// Also check if file path contains pattern
		if strings.Contains(filePath, pattern[1:]) { // Remove leading *
			return true
		}
	}
	// Check for config files specifically
	if strings.Contains(filePath, "config") && (strings.HasSuffix(filePath, ".yaml") || strings.HasSuffix(filePath, ".yml")) {
		return true
	}
	return false
}

// hasRelevantChanges checks if any commits contain relevant file changes
func hasRelevantChanges(payload *GiteaWebhookPayload) bool {
	for _, commit := range payload.Commits {
		// Check added files
		for _, file := range commit.Added {
			if isRelevantFile(file) {
				return true
			}
		}
		// Check modified files
		for _, file := range commit.Modified {
			if isRelevantFile(file) {
				return true
			}
		}
		// Check removed files (might affect processing)
		for _, file := range commit.Removed {
			if isRelevantFile(file) {
				return true
			}
		}
	}
	return false
}

// extractOwnerRepo extracts owner and repo name from full_name or clone_url
func extractOwnerRepo(fullName, cloneURL string) (owner, repo string) {
	if fullName != "" {
		parts := strings.Split(fullName, "/")
		if len(parts) == 2 {
			return parts[0], parts[1]
		}
	}
	if cloneURL != "" {
		// Extract from clone URL: https://gitea.example.com/owner/repo.git
		parts := strings.Split(strings.TrimSuffix(cloneURL, ".git"), "/")
		if len(parts) >= 2 {
			return parts[len(parts)-2], parts[len(parts)-1]
		}
	}
	return "", ""
}

// extractBranch extracts branch name from ref (e.g., "refs/heads/main" -> "main")
func extractBranch(ref string) string {
	if strings.HasPrefix(ref, "refs/heads/") {
		return strings.TrimPrefix(ref, "refs/heads/")
	}
	return "main" // Default
}

// handleGiteaWebhook handles incoming Gitea webhook events
func (s *extractServer) handleGiteaWebhook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error": "method not allowed",
		})
		return
	}

	// Read webhook payload
	body, err := io.ReadAll(r.Body)
	if err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": fmt.Sprintf("failed to read request body: %v", err),
		})
		return
	}
	defer r.Body.Close()

	// Parse webhook payload
	var payload GiteaWebhookPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": fmt.Sprintf("failed to parse webhook payload: %v", err),
		})
		return
	}

	// Only process push events
	if payload.Action != "push" {
		handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"message": fmt.Sprintf("ignoring non-push event: %s", payload.Action),
		})
		return
	}

	// Check if relevant files changed
	if !hasRelevantChanges(&payload) {
		handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"message": "no relevant file changes detected, skipping pipeline trigger",
		})
		return
	}

	// Extract repository information
	owner, repo := extractOwnerRepo(payload.Repository.FullName, payload.Repository.CloneURL)
	if owner == "" || repo == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": "could not extract owner and repo from webhook payload",
		})
		return
	}

	branch := extractBranch(payload.Ref)

	s.logger.Printf("Gitea webhook: Processing push event for %s/%s@%s (commit: %s)",
		owner, repo, branch, payload.After)

	// Get Gitea credentials from environment or headers
	giteaURL := r.Header.Get("X-Gitea-URL")
	if giteaURL == "" {
		giteaURL = os.Getenv("GITEA_URL")
	}
	giteaToken := r.Header.Get("X-Gitea-Token")
	if giteaToken == "" {
		giteaToken = os.Getenv("GITEA_TOKEN")
	}

	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": "Gitea URL and token are required (set GITEA_URL and GITEA_TOKEN or provide X-Gitea-URL and X-Gitea-Token headers)",
		})
		return
	}

	// Process the repository asynchronously
	go s.processRepositoryFromWebhook(context.Background(), giteaURL, giteaToken, owner, repo, branch, payload)

	// Return immediately
	handlers.WriteJSON(w, http.StatusAccepted, map[string]interface{}{
		"message":    "webhook received, processing repository",
		"repository": fmt.Sprintf("%s/%s", owner, repo),
		"branch":     branch,
		"commit":     payload.After,
	})
}

// processRepositoryFromWebhook processes a repository triggered by webhook
func (s *extractServer) processRepositoryFromWebhook(
	ctx context.Context,
	giteaURL, giteaToken, owner, repo, branch string,
	payload GiteaWebhookPayload,
) {
	startTime := time.Now()
	s.logger.Printf("Starting webhook-triggered processing for %s/%s@%s", owner, repo, branch)

	// Create temp directory for cloning
	tempDir := filepath.Join(os.TempDir(), "extract-webhook", fmt.Sprintf("%s-%s-%d", owner, repo, time.Now().Unix()))
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		s.logger.Printf("Failed to create temp directory: %v", err)
		return
	}
	defer os.RemoveAll(tempDir)

	// Clone repository
	giteaStorage := git.NewGiteaStorage(giteaURL, giteaToken, s.logger)
	clonePath, err := giteaStorage.CloneFromGitea(ctx, payload.Repository.CloneURL, branch, tempDir)
	if err != nil {
		s.logger.Printf("Failed to clone repository %s/%s: %v", owner, repo, err)
		return
	}

	// Look for config file in the repository
	configFiles := []string{
		filepath.Join(clonePath, "*-config.yaml"),
		filepath.Join(clonePath, "*-config.yml"),
		filepath.Join(clonePath, "config.yaml"),
		filepath.Join(clonePath, "config.yml"),
		filepath.Join(clonePath, "sgmi-config.yaml"),
	}

	var configPath string
	for _, pattern := range configFiles {
		matches, _ := filepath.Glob(pattern)
		if len(matches) > 0 {
			configPath = matches[0]
			break
		}
	}

	// If no config found, try to find any YAML file in root
	if configPath == "" {
		entries, _ := os.ReadDir(clonePath)
		for _, entry := range entries {
			if !entry.IsDir() && (strings.HasSuffix(entry.Name(), ".yaml") || strings.HasSuffix(entry.Name(), ".yml")) {
				configPath = filepath.Join(clonePath, entry.Name())
				break
			}
		}
	}

	// Load config if found, otherwise use defaults
	var projectID, systemID string
	if configPath != "" {
		s.logger.Printf("Found config file: %s", configPath)
		// Try to extract project_id and system_id from config
		// For now, we'll use a simple approach - in production, use proper YAML parsing
		configData, err := os.ReadFile(configPath)
		if err == nil {
			configStr := string(configData)
			// Simple extraction (could be improved with proper YAML parsing)
			if strings.Contains(configStr, "project:") {
				// Try to extract project ID
				lines := strings.Split(configStr, "\n")
				for i, line := range lines {
					if strings.Contains(line, "id:") && i > 0 && strings.Contains(lines[i-1], "project:") {
						parts := strings.Split(strings.TrimSpace(line), ":")
						if len(parts) == 2 {
							projectID = strings.Trim(strings.TrimSpace(parts[1]), "\"")
							break
						}
					}
					if strings.Contains(line, "system_id:") {
						parts := strings.Split(strings.TrimSpace(line), ":")
						if len(parts) == 2 {
							systemID = strings.Trim(strings.TrimSpace(parts[1]), "\"")
						}
					}
				}
			}
		}
	}

	// Default project/system IDs if not found
	if projectID == "" {
		projectID = repo // Use repo name as project ID
	}
	if systemID == "" {
		systemID = repo
	}

	// Build file paths from repository
	var jsonFiles, ddlFiles, xmlFiles []string
	err = filepath.Walk(clonePath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}

		relPath, _ := filepath.Rel(clonePath, path)
		ext := strings.ToLower(filepath.Ext(path))

		switch ext {
		case ".json":
			jsonFiles = append(jsonFiles, path)
		case ".hql", ".ddl", ".sql":
			ddlFiles = append(ddlFiles, path)
		case ".xml":
			xmlFiles = append(xmlFiles, path)
		}
		return nil
	})

	if err != nil {
		s.logger.Printf("Error walking repository: %v", err)
	}

	// Build knowledge graph request
	kgRequest := graphRequest{
		JSONTables:    jsonFiles,
		HiveDDLs:      ddlFiles,
		ControlMFiles: xmlFiles,
		ProjectID:     projectID,
		SystemID:      systemID,
		GiteaStorage: &giteaStorageReq{
			Enabled:     true,
			GiteaURL:    giteaURL,
			GiteaToken:  giteaToken,
			Owner:       owner,
			RepoName:    repo,
			Branch:      branch,
			AutoCreate:  false,
			Description: fmt.Sprintf("Auto-processed from webhook (commit: %s)", payload.After),
		},
		IdealDistribution: map[string]float64{},
	}

	// Trigger processing by making an internal HTTP request to /knowledge-graph endpoint
	// This reuses the existing handleGraph logic
	s.logger.Printf("Triggering knowledge graph processing for project=%s, system=%s", projectID, systemID)
	
	// Build the request payload
	requestPayload := map[string]interface{}{
		"project_id":      kgRequest.ProjectID,
		"system_id":       kgRequest.SystemID,
		"json_tables":     kgRequest.JSONTables,
		"hive_ddls":       kgRequest.HiveDDLs,
		"control_m_files": kgRequest.ControlMFiles,
		"gitea_storage": map[string]interface{}{
			"enabled":     kgRequest.GiteaStorage.Enabled,
			"gitea_url":   kgRequest.GiteaStorage.GiteaURL,
			"gitea_token": kgRequest.GiteaStorage.GiteaToken,
			"owner":       kgRequest.GiteaStorage.Owner,
			"repo_name":   kgRequest.GiteaStorage.RepoName,
			"branch":      kgRequest.GiteaStorage.Branch,
		},
		"ideal_distribution": map[string]float64{},
	}

	// Make internal HTTP request to process
	extractURL := os.Getenv("EXTRACT_SERVICE_URL")
	if extractURL == "" {
		extractURL = "http://localhost:8083"
	}
	
	payloadJSON, err := json.Marshal(requestPayload)
	if err != nil {
		s.logger.Printf("Failed to marshal request payload: %v", err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, extractURL+"/knowledge-graph", strings.NewReader(string(payloadJSON)))
	if err != nil {
		s.logger.Printf("Failed to create request: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Minute} // Knowledge graph processing can take time
	resp, err := client.Do(req)
	if err != nil {
		s.logger.Printf("Failed to process repository: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		s.logger.Printf("Knowledge graph processing failed with status %d: %s", resp.StatusCode, string(body))
		return
	}

	var result struct {
		Nodes int `json:"nodes"`
		Edges int `json:"edges"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		s.logger.Printf("Failed to decode response: %v", err)
		return
	}

	duration := time.Since(startTime)
	s.logger.Printf("Successfully processed repository %s/%s: %d nodes, %d edges (duration: %v)",
		owner, repo, result.Nodes, result.Edges, duration)
}

