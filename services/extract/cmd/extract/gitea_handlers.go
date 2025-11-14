package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	handlers "github.com/plturrell/aModels/services/extract/internal/handlers"
	"github.com/plturrell/aModels/services/extract/pkg/git"
)

// getGiteaConfig retrieves Gitea configuration from HTTP request.
// It checks headers first (X-Gitea-URL, X-Gitea-Token), then query parameters (for backward compatibility),
// and finally environment variables (GITEA_URL, GITEA_TOKEN).
// Returns the Gitea URL and token, or empty strings if not found.
func getGiteaConfig(r *http.Request) (giteaURL, giteaToken string) {
	// Try headers first (more secure)
	giteaURL = r.Header.Get("X-Gitea-URL")
	giteaToken = r.Header.Get("X-Gitea-Token")
	
	// Fallback to query params for backward compatibility (deprecated)
	if giteaURL == "" {
		giteaURL = r.URL.Query().Get("gitea_url")
	}
	if giteaToken == "" {
		giteaToken = r.URL.Query().Get("gitea_token")
	}
	
	// Fallback to environment variables
	if giteaURL == "" {
		giteaURL = os.Getenv("GITEA_URL")
	}
	if giteaToken == "" {
		giteaToken = os.Getenv("GITEA_TOKEN")
	}
	
	return giteaURL, giteaToken
}

// handleGiteaRepositoryRouter routes requests to specific repository operation handlers.
// It parses the URL path to determine which operation to perform (files, branches, commits, clone).
func (s *extractServer) handleGiteaRepositoryRouter(w http.ResponseWriter, r *http.Request) {
	// Parse path: /gitea/repositories/{owner}/{repo}/...
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 2 {
		http.Error(w, "invalid path", http.StatusBadRequest)
		return
	}

	// Check for specific endpoints
	if len(pathParts) >= 3 {
		endpoint := pathParts[2]
		switch endpoint {
		case "files":
			if len(pathParts) > 3 {
				// File content request: /gitea/repositories/{owner}/{repo}/files/{path}
				s.handleGiteaRepositoryFileContent(w, r)
			} else {
				// List/create files: /gitea/repositories/{owner}/{repo}/files
				s.handleGiteaRepositoryFiles(w, r)
			}
			return
		case "branches":
			s.handleGiteaRepositoryBranches(w, r)
			return
		case "commits":
			s.handleGiteaRepositoryCommits(w, r)
			return
		case "clone":
			s.handleGiteaRepositoryClone(w, r)
			return
		}
	}

	// Default: repository operations
	s.handleGiteaRepository(w, r)
}

// handleGiteaRepositories handles repository list and creation operations.
// GET: Lists repositories with optional owner filter and pagination support.
// POST: Creates a new repository with validation of all input parameters.
func (s *extractServer) handleGiteaRepositories(w http.ResponseWriter, r *http.Request) {
	// Get Gitea config from headers (preferred) or environment
	giteaURL, giteaToken := getGiteaConfig(r)

	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers, or set GITEA_URL and GITEA_TOKEN environment variables",
		})
		return
	}

	client := git.NewGiteaClient(giteaURL, giteaToken)

	switch r.Method {
	case http.MethodGet:
		// List repositories
		owner := r.URL.Query().Get("owner")
		if owner != "" {
			if err := git.ValidateOwnerName(owner); err != nil {
				handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
					"error":   err.Error(),
					"code":    "INVALID_OWNER",
					"details": "Owner name must contain only alphanumeric characters, dots, hyphens, and underscores",
				})
				return
			}
		}
		
		// Parse pagination parameters
		var pagination *git.PaginationOptions
		pageStr := r.URL.Query().Get("page")
		limitStr := r.URL.Query().Get("limit")
		if pageStr != "" || limitStr != "" {
			pagination = &git.PaginationOptions{}
			if pageStr != "" {
				if _, err := fmt.Sscanf(pageStr, "%d", &pagination.Page); err != nil || pagination.Page < 1 {
					handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
						"error":   "Invalid page parameter",
						"code":    "INVALID_PAGE",
						"details": "Page must be a positive integer",
					})
					return
				}
			}
			if limitStr != "" {
				if _, err := fmt.Sscanf(limitStr, "%d", &pagination.Limit); err != nil || pagination.Limit < 1 || pagination.Limit > 100 {
					handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
						"error":   "Invalid limit parameter",
						"code":    "INVALID_LIMIT",
						"details": "Limit must be between 1 and 100",
					})
					return
				}
			}
		}
		
		ctx := r.Context()
		repos, resultPagination, err := client.ListRepositories(ctx, owner, pagination)
		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error":   "Failed to list repositories",
				"code":    "LIST_REPOSITORIES_FAILED",
				"details": err.Error(),
			})
			return
		}
		
		// Return paginated response if pagination was requested
		if pagination != nil {
			hasMore := resultPagination != nil && resultPagination.Limit > 0 && len(repos) == resultPagination.Limit
			handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
				"data":     repos,
				"page":     pagination.Page,
				"limit":    pagination.Limit,
				"has_more": hasMore,
			})
		} else {
			handlers.WriteJSON(w, http.StatusOK, repos)
		}

	case http.MethodPost:
		// Create repository
		var req struct {
			Owner       string `json:"owner"`
			Name        string `json:"name"`
			Description string `json:"description"`
			Private     bool   `json:"private"`
			AutoInit    bool   `json:"auto_init"`
			Readme      string `json:"readme,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   "Invalid request body",
				"code":    "INVALID_REQUEST",
				"details": err.Error(),
			})
			return
		}

		// Validate input
		if err := git.ValidateRepositoryName(req.Name); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   err.Error(),
				"code":    "INVALID_REPO_NAME",
				"details": "Repository name must contain only alphanumeric characters, dots, hyphens, and underscores (max 100 chars)",
			})
			return
		}

		if req.Owner != "" {
			if err := git.ValidateOwnerName(req.Owner); err != nil {
				handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
					"error":   err.Error(),
					"code":    "INVALID_OWNER",
					"details": "Owner name must contain only alphanumeric characters, dots, hyphens, and underscores (max 100 chars)",
				})
				return
			}
		}

		if req.Description != "" {
			if err := git.ValidateDescription(req.Description); err != nil {
				handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
					"error":   err.Error(),
					"code":    "INVALID_DESCRIPTION",
					"details": "Description must not exceed 2000 characters",
				})
				return
			}
		}

		createReq := git.CreateRepositoryRequest{
			Name:        req.Name,
			Description: req.Description,
			Private:     req.Private,
			AutoInit:    req.AutoInit,
			Readme:      req.Readme,
		}

		ctx := r.Context()
		repo, err := client.CreateRepository(ctx, req.Owner, createReq)
		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error":   "Failed to create repository",
				"code":    "CREATE_REPOSITORY_FAILED",
				"details": err.Error(),
			})
			return
		}

		handlers.WriteJSON(w, http.StatusCreated, repo)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleGiteaRepository handles operations on a specific repository.
// GET: Retrieves repository details including clone URLs and metadata.
// DELETE: Permanently deletes a repository (destructive operation).
func (s *extractServer) handleGiteaRepository(w http.ResponseWriter, r *http.Request) {
	// Parse owner and repo from path: /gitea/repositories/{owner}/{repo}
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 2 {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid path",
			"code":    "INVALID_PATH",
			"details": "Expected /gitea/repositories/{owner}/{repo}",
		})
		return
	}

	owner := pathParts[0]
	repo := pathParts[1]

	// Validate owner and repo names
	if err := git.ValidateOwnerName(owner); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_OWNER",
			"details": err.Error(),
		})
		return
	}
	if err := git.ValidateRepositoryName(repo); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_REPO_NAME",
			"details": err.Error(),
		})
		return
	}

	// Get Gitea config from headers
	giteaURL, giteaToken := getGiteaConfig(r)
	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers",
		})
		return
	}

	client := git.NewGiteaClient(giteaURL, giteaToken)
	ctx := r.Context()

	switch r.Method {
	case http.MethodGet:
		// Get repository details
		repoInfo, err := client.GetRepository(ctx, owner, repo)
		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error":   "Failed to get repository",
				"code":    "GET_REPOSITORY_FAILED",
				"details": err.Error(),
			})
			return
		}
		handlers.WriteJSON(w, http.StatusOK, repoInfo)

	case http.MethodDelete:
		// Delete repository
		if err := client.DeleteRepository(ctx, owner, repo); err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error":   "Failed to delete repository",
				"code":    "DELETE_REPOSITORY_FAILED",
				"details": err.Error(),
			})
			return
		}
		handlers.WriteJSON(w, http.StatusNoContent, nil)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleGiteaRepositoryFiles handles file operations within a repository.
// GET: Lists files in a directory with optional path and branch/ref parameters, supports pagination.
// POST: Creates or updates a file with content validation and path sanitization.
func (s *extractServer) handleGiteaRepositoryFiles(w http.ResponseWriter, r *http.Request) {
	// Parse owner and repo from path: /gitea/repositories/{owner}/{repo}/files
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 3 || pathParts[2] != "files" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid path",
			"code":    "INVALID_PATH",
			"details": "Expected /gitea/repositories/{owner}/{repo}/files",
		})
		return
	}

	owner := pathParts[0]
	repo := pathParts[1]

	// Validate owner and repo
	if err := git.ValidateOwnerName(owner); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_OWNER",
			"details": err.Error(),
		})
		return
	}
	if err := git.ValidateRepositoryName(repo); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_REPO_NAME",
			"details": err.Error(),
		})
		return
	}

	// Get Gitea config from headers
	giteaURL, giteaToken := getGiteaConfig(r)
	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers",
		})
		return
	}

	client := git.NewGiteaClient(giteaURL, giteaToken)
	ctx := r.Context()

	switch r.Method {
	case http.MethodGet:
		// List files
		path := r.URL.Query().Get("path")
		ref := r.URL.Query().Get("ref")
		if ref == "" {
			ref = "main"
		}

		// Validate branch name
		if err := git.ValidateBranchName(ref); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   err.Error(),
				"code":    "INVALID_BRANCH",
				"details": err.Error(),
			})
			return
		}

		// Validate and sanitize file path if provided
		if path != "" {
			sanitizedPath, err := git.ValidateFilePath(path)
			if err != nil {
				handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
					"error":   err.Error(),
					"code":    "INVALID_FILE_PATH",
					"details": err.Error(),
				})
				return
			}
			path = sanitizedPath
		}

		// Parse pagination parameters
		var pagination *git.PaginationOptions
		pageStr := r.URL.Query().Get("page")
		limitStr := r.URL.Query().Get("limit")
		if pageStr != "" || limitStr != "" {
			pagination = &git.PaginationOptions{}
			if pageStr != "" {
				if _, err := fmt.Sscanf(pageStr, "%d", &pagination.Page); err != nil || pagination.Page < 1 {
					handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
						"error":   "Invalid page parameter",
						"code":    "INVALID_PAGE",
						"details": "Page must be a positive integer",
					})
					return
				}
			}
			if limitStr != "" {
				if _, err := fmt.Sscanf(limitStr, "%d", &pagination.Limit); err != nil || pagination.Limit < 1 || pagination.Limit > 100 {
					handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
						"error":   "Invalid limit parameter",
						"code":    "INVALID_LIMIT",
						"details": "Limit must be between 1 and 100",
					})
					return
				}
			}
		}

		files, resultPagination, err := client.ListFiles(ctx, owner, repo, path, ref, pagination)
		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error":   "Failed to list files",
				"code":    "LIST_FILES_FAILED",
				"details": err.Error(),
			})
			return
		}
		
		// Return paginated response if pagination was requested
		if pagination != nil {
			hasMore := resultPagination != nil && resultPagination.Limit > 0 && len(files) == resultPagination.Limit
			handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
				"data":     files,
				"page":     pagination.Page,
				"limit":    pagination.Limit,
				"has_more": hasMore,
			})
		} else {
			handlers.WriteJSON(w, http.StatusOK, files)
		}

	case http.MethodPost:
		// Create or update file
		var req struct {
			Path    string `json:"path"`
			Content string `json:"content"`
			Message string `json:"message"`
			Branch  string `json:"branch"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   "Invalid request body",
				"code":    "INVALID_REQUEST",
				"details": err.Error(),
			})
			return
		}

		// Validate required fields
		if req.Path == "" {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   "File path is required",
				"code":    "MISSING_FILE_PATH",
				"details": "The 'path' field is required",
			})
			return
		}

		// Validate and sanitize file path
		sanitizedPath, err := git.ValidateFilePath(req.Path)
		if err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   err.Error(),
				"code":    "INVALID_FILE_PATH",
				"details": err.Error(),
			})
			return
		}
		req.Path = sanitizedPath

		// Set defaults
		if req.Message == "" {
			req.Message = fmt.Sprintf("Update %s", req.Path)
		}
		if req.Branch == "" {
			req.Branch = "main"
		}

		// Validate branch name
		if err := git.ValidateBranchName(req.Branch); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   err.Error(),
				"code":    "INVALID_BRANCH",
				"details": err.Error(),
			})
			return
		}

		if err := client.CreateOrUpdateFile(ctx, owner, repo, req.Path, req.Content, req.Message, req.Branch); err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"error":   "Failed to create/update file",
				"code":    "CREATE_UPDATE_FILE_FAILED",
				"details": err.Error(),
			})
			return
		}

		handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"path":    req.Path,
			"message": "file created/updated successfully",
		})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleGiteaRepositoryFileContent retrieves the raw content of a specific file.
// The file path is extracted from the URL and validated to prevent path traversal attacks.
// Supports optional branch/ref parameter to read from specific commits or branches.
func (s *extractServer) handleGiteaRepositoryFileContent(w http.ResponseWriter, r *http.Request) {
	// Parse owner, repo, and file path from URL
	// Path format: /gitea/repositories/{owner}/{repo}/files/{path}
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 4 || pathParts[2] != "files" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid path",
			"code":    "INVALID_PATH",
			"details": "Expected /gitea/repositories/{owner}/{repo}/files/{path}",
		})
		return
	}

	owner := pathParts[0]
	repo := pathParts[1]
	filePath := strings.Join(pathParts[3:], "/")

	// Validate owner and repo
	if err := git.ValidateOwnerName(owner); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_OWNER",
			"details": err.Error(),
		})
		return
	}
	if err := git.ValidateRepositoryName(repo); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_REPO_NAME",
			"details": err.Error(),
		})
		return
	}

	// Validate and sanitize file path
	sanitizedPath, err := git.ValidateFilePath(filePath)
	if err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_FILE_PATH",
			"details": err.Error(),
		})
		return
	}
	filePath = sanitizedPath

	// Get Gitea config from headers
	giteaURL, giteaToken := getGiteaConfig(r)
	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers",
		})
		return
	}

	client := git.NewGiteaClient(giteaURL, giteaToken)
	ctx := r.Context()

	ref := r.URL.Query().Get("ref")
	if ref == "" {
		ref = "main"
	}

	// Validate branch name
	if err := git.ValidateBranchName(ref); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_BRANCH",
			"details": err.Error(),
		})
		return
	}

	content, err := client.GetFileContent(ctx, owner, repo, filePath, ref)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error":   "Failed to get file content",
			"code":    "GET_FILE_CONTENT_FAILED",
			"details": err.Error(),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"path":    filePath,
		"content": content,
		"ref":     ref,
	})
}

// handleGiteaRepositoryBranches lists all branches for a repository.
// Returns branch information including the latest commit for each branch.
func (s *extractServer) handleGiteaRepositoryBranches(w http.ResponseWriter, r *http.Request) {
	// Parse owner and repo from path: /gitea/repositories/{owner}/{repo}/branches
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 3 || pathParts[2] != "branches" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid path",
			"code":    "INVALID_PATH",
			"details": "Expected /gitea/repositories/{owner}/{repo}/branches",
		})
		return
	}

	owner := pathParts[0]
	repo := pathParts[1]

	// Validate owner and repo
	if err := git.ValidateOwnerName(owner); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_OWNER",
			"details": err.Error(),
		})
		return
	}
	if err := git.ValidateRepositoryName(repo); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_REPO_NAME",
			"details": err.Error(),
		})
		return
	}

	// Get Gitea config from headers
	giteaURL, giteaToken := getGiteaConfig(r)
	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers",
		})
		return
	}

	client := git.NewGiteaClient(giteaURL, giteaToken)
	ctx := r.Context()

	branches, err := client.ListBranches(ctx, owner, repo)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error":   "Failed to list branches",
			"code":    "LIST_BRANCHES_FAILED",
			"details": err.Error(),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, branches)
}

// handleGiteaRepositoryCommits lists commits for a repository branch.
// Supports pagination via page and limit query parameters (default limit: 30, max: 100).
// Returns commit history with author and committer information.
func (s *extractServer) handleGiteaRepositoryCommits(w http.ResponseWriter, r *http.Request) {
	// Parse owner and repo from path: /gitea/repositories/{owner}/{repo}/commits
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 3 || pathParts[2] != "commits" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid path",
			"code":    "INVALID_PATH",
			"details": "Expected /gitea/repositories/{owner}/{repo}/commits",
		})
		return
	}

	owner := pathParts[0]
	repo := pathParts[1]

	// Validate owner and repo
	if err := git.ValidateOwnerName(owner); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_OWNER",
			"details": err.Error(),
		})
		return
	}
	if err := git.ValidateRepositoryName(repo); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_REPO_NAME",
			"details": err.Error(),
		})
		return
	}

	// Get Gitea config from headers
	giteaURL, giteaToken := getGiteaConfig(r)
	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers",
		})
		return
	}

	client := git.NewGiteaClient(giteaURL, giteaToken)
	ctx := r.Context()

	branch := r.URL.Query().Get("branch")
	if branch == "" {
		branch = "main"
	}

	// Validate branch name
	if err := git.ValidateBranchName(branch); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_BRANCH",
			"details": err.Error(),
		})
		return
	}

	// Parse pagination parameters
	var pagination *git.PaginationOptions
	pageStr := r.URL.Query().Get("page")
	limitStr := r.URL.Query().Get("limit")
	limit := 30 // Default limit for backward compatibility
	
	if pageStr != "" || limitStr != "" {
		pagination = &git.PaginationOptions{}
		if pageStr != "" {
			if _, err := fmt.Sscanf(pageStr, "%d", &pagination.Page); err != nil || pagination.Page < 1 {
				handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
					"error":   "Invalid page parameter",
					"code":    "INVALID_PAGE",
					"details": "Page must be a positive integer",
				})
				return
			}
		}
		if limitStr != "" {
			if _, err := fmt.Sscanf(limitStr, "%d", &pagination.Limit); err != nil || pagination.Limit < 1 || pagination.Limit > 100 {
				handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
					"error":   "Invalid limit parameter",
					"code":    "INVALID_LIMIT",
					"details": "Limit must be between 1 and 100",
				})
				return
			}
			limit = pagination.Limit
		}
	} else if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		// Backward compatibility: support limit without page
		if _, err := fmt.Sscanf(limitStr, "%d", &limit); err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   "Invalid limit parameter",
				"code":    "INVALID_LIMIT",
				"details": "Limit must be a positive integer",
			})
			return
		}
		if limit < 1 || limit > 100 {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   "Invalid limit parameter",
				"code":    "INVALID_LIMIT",
				"details": "Limit must be between 1 and 100",
			})
			return
		}
	}

	commits, resultPagination, err := client.ListCommits(ctx, owner, repo, branch, limit, pagination)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error":   "Failed to list commits",
			"code":    "LIST_COMMITS_FAILED",
			"details": err.Error(),
		})
		return
	}

	// Return paginated response if pagination was requested
	if pagination != nil {
		hasMore := resultPagination != nil && resultPagination.Limit > 0 && len(commits) == resultPagination.Limit
		handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"data":     commits,
			"page":     pagination.Page,
			"limit":    pagination.Limit,
			"has_more": hasMore,
		})
	} else {
		handlers.WriteJSON(w, http.StatusOK, commits)
	}
}

// handleGiteaRepositoryClone clones a Gitea repository to the local filesystem for processing.
// This is used by the extraction pipeline to work with repository contents locally.
// Supports optional branch and target path parameters.
func (s *extractServer) handleGiteaRepositoryClone(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse owner and repo from path: /gitea/repositories/{owner}/{repo}/clone
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/gitea/repositories/"), "/")
	if len(pathParts) < 3 || pathParts[2] != "clone" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Invalid path",
			"code":    "INVALID_PATH",
			"details": "Expected /gitea/repositories/{owner}/{repo}/clone",
		})
		return
	}

	owner := pathParts[0]
	repo := pathParts[1]

	// Validate owner and repo
	if err := git.ValidateOwnerName(owner); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_OWNER",
			"details": err.Error(),
		})
		return
	}
	if err := git.ValidateRepositoryName(repo); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_REPO_NAME",
			"details": err.Error(),
		})
		return
	}

	// Get Gitea config from headers
	giteaURL, giteaToken := getGiteaConfig(r)
	if giteaURL == "" || giteaToken == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   "Gitea URL and token are required",
			"code":    "MISSING_CREDENTIALS",
			"details": "Provide X-Gitea-URL and X-Gitea-Token headers",
		})
		return
	}

	var req struct {
		Branch string `json:"branch"`
		Path   string `json:"path,omitempty"` // Optional target path
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Body is optional, use defaults
		req.Branch = "main"
	}

	if req.Branch == "" {
		req.Branch = "main"
	}

	// Validate branch name
	if err := git.ValidateBranchName(req.Branch); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":   err.Error(),
			"code":    "INVALID_BRANCH",
			"details": err.Error(),
		})
		return
	}

	// Validate path if provided
	if req.Path != "" {
		sanitizedPath, err := git.ValidateFilePath(req.Path)
		if err != nil {
			handlers.WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error":   err.Error(),
				"code":    "INVALID_PATH",
				"details": err.Error(),
			})
			return
		}
		req.Path = sanitizedPath
	}

	// Get repository info to get clone URL
	client := git.NewGiteaClient(giteaURL, giteaToken)
	ctx := r.Context()

	repoInfo, err := client.GetRepository(ctx, owner, repo)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error":   "Failed to get repository",
			"code":    "GET_REPOSITORY_FAILED",
			"details": err.Error(),
		})
		return
	}

	// Use GiteaStorage to clone
	giteaStorage := git.NewGiteaStorage(giteaURL, giteaToken, s.logger)
	clonePath, err := giteaStorage.CloneFromGitea(ctx, repoInfo.CloneURL, req.Branch, req.Path)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error":   "Failed to clone repository",
			"code":    "CLONE_REPOSITORY_FAILED",
			"details": err.Error(),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"clone_path": clonePath,
		"repository": repoInfo,
		"branch":     req.Branch,
	})
}

