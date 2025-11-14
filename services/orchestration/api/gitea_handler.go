package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// GiteaHandler provides HTTP handlers for Gitea repository operations in workflows.
type GiteaHandler struct {
	pipeline *agents.GiteaPipeline
	logger   *log.Logger
}

// NewGiteaHandler creates a new Gitea handler.
func NewGiteaHandler(logger *log.Logger) (*GiteaHandler, error) {
	config := agents.GiteaPipelineConfig{
		ExtractServiceURL: getEnvOrDefault("EXTRACT_SERVICE_URL", "http://localhost:8081"),
		GiteaURL:          os.Getenv("GITEA_URL"),
		GiteaToken:        os.Getenv("GITEA_TOKEN"),
		Logger:            logger,
	}

	pipeline, err := agents.NewGiteaPipeline(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Gitea pipeline: %w", err)
	}

	return &GiteaHandler{
		pipeline: pipeline,
		logger:   logger,
	}, nil
}

// HandleCreateRepository handles POST /api/gitea/repositories.
func (h *GiteaHandler) HandleCreateRepository(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req agents.CreateRepositoryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	result, err := h.pipeline.CreateRepository(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, result)
		return
	}

	writeJSON(w, http.StatusOK, result)
}

// HandleCloneRepository handles POST /api/gitea/repositories/{owner}/{repo}/clone.
func (h *GiteaHandler) HandleCloneRepository(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse owner and repo from path
	path := strings.TrimPrefix(r.URL.Path, "/api/gitea/repositories/")
	path = strings.TrimSuffix(path, "/clone")
	parts := strings.Split(path, "/")
	if len(parts) < 2 {
		http.Error(w, "Invalid path: expected /api/gitea/repositories/{owner}/{repo}/clone", http.StatusBadRequest)
		return
	}

	var req agents.CloneRepositoryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Body is optional, use path values
		req = agents.CloneRepositoryRequest{
			Owner: parts[0],
			Repo:  parts[1],
		}
	} else {
		// Override with path values if not provided
		if req.Owner == "" {
			req.Owner = parts[0]
		}
		if req.Repo == "" {
			req.Repo = parts[1]
		}
	}

	result, err := h.pipeline.CloneRepository(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, result)
		return
	}

	writeJSON(w, http.StatusOK, result)
}

// HandleProcessRepository handles POST /api/gitea/repositories/{owner}/{repo}/process.
func (h *GiteaHandler) HandleProcessRepository(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse owner and repo from path
	path := strings.TrimPrefix(r.URL.Path, "/api/gitea/repositories/")
	path = strings.TrimSuffix(path, "/process")
	parts := strings.Split(path, "/")
	if len(parts) < 2 {
		http.Error(w, "Invalid path: expected /api/gitea/repositories/{owner}/{repo}/process", http.StatusBadRequest)
		return
	}

	var req agents.ProcessRepositoryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Override with path values if not provided
	if req.Owner == "" {
		req.Owner = parts[0]
	}
	if req.Repo == "" {
		req.Repo = parts[1]
	}

	if req.ProjectID == "" {
		http.Error(w, "project_id is required", http.StatusBadRequest)
		return
	}

	result, err := h.pipeline.ProcessRepository(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, result)
		return
	}

	writeJSON(w, http.StatusOK, result)
}

