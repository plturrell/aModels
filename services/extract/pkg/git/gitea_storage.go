package git

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// GiteaStorage handles storing extracted code into Gitea repositories
type GiteaStorage struct {
	giteaClient *GiteaClient
	logger      *log.Logger
}

// NewGiteaStorage creates a new Gitea storage handler
func NewGiteaStorage(giteaURL, giteaToken string, logger *log.Logger) *GiteaStorage {
	return &GiteaStorage{
		giteaClient: NewGiteaClient(giteaURL, giteaToken),
		logger:      logger,
	}
}

// GiteaClient returns the underlying Gitea client
func (s *GiteaStorage) GiteaClient() *GiteaClient {
	return s.giteaClient
}

// StorageConfig configures where to store code in Gitea
type StorageConfig struct {
	Owner       string // Repository owner (user or org)
	RepoName    string // Repository name
	Branch      string // Branch to commit to (default: main)
	BasePath    string // Base path in repository (e.g., "extracted-code/")
	ProjectID   string // Project ID for organization
	SystemID    string // System ID for organization
	AutoCreate  bool   // Automatically create repository if it doesn't exist
	Description string // Repository description
}

// StoreCode stores extracted code files into a Gitea repository
func (s *GiteaStorage) StoreCode(ctx context.Context, config StorageConfig, files []ExtractedFile, repoMeta *RepositoryMetadata) (string, error) {
	// Ensure repository exists
	repo, err := s.ensureRepository(ctx, config)
	if err != nil {
		return "", fmt.Errorf("ensure repository: %w", err)
	}

	branch := config.Branch
	if branch == "" {
		branch = "main"
	}

	// Create commit message
	commitMessage := fmt.Sprintf("Extract code from %s (commit: %s)", repoMeta.URL, repoMeta.Commit)
	if config.ProjectID != "" {
		commitMessage = fmt.Sprintf("[%s] %s", config.ProjectID, commitMessage)
	}

	// Store each file
	basePath := strings.TrimSuffix(config.BasePath, "/")
	if basePath != "" && !strings.HasSuffix(basePath, "/") {
		basePath += "/"
	}

	successCount := 0
	for _, file := range files {
		// Construct file path in repository
		repoPath := basePath + file.Path
		
		// Ensure path uses forward slashes (Gitea API requirement)
		repoPath = filepath.ToSlash(repoPath)

		// Store file via Gitea API
		err := s.giteaClient.CreateOrUpdateFile(
			ctx,
			config.Owner,
			config.RepoName,
			repoPath,
			file.Content,
			fmt.Sprintf("%s: %s", commitMessage, file.Path),
			branch,
		)
		if err != nil {
			s.logger.Printf("Warning: failed to store file %s to Gitea: %v", file.Path, err)
			continue
		}
		successCount++
	}

	s.logger.Printf("Stored %d/%d files to Gitea repository %s/%s", successCount, len(files), config.Owner, config.RepoName)

	return repo.HTMLURL, nil
}

// ensureRepository ensures the repository exists, creating it if needed
func (s *GiteaStorage) ensureRepository(ctx context.Context, config StorageConfig) (*Repository, error) {
	// Try to get existing repository
	repo, err := s.giteaClient.GetRepository(ctx, config.Owner, config.RepoName)
	if err == nil {
		return repo, nil
	}

	// Repository doesn't exist - create if auto-create is enabled
	if !config.AutoCreate {
		return nil, fmt.Errorf("repository %s/%s does not exist and auto-create is disabled", config.Owner, config.RepoName)
	}

	s.logger.Printf("Creating Gitea repository: %s/%s", config.Owner, config.RepoName)

	description := config.Description
	if description == "" {
		description = fmt.Sprintf("Extracted code for project %s", config.ProjectID)
	}

	createReq := CreateRepositoryRequest{
		Name:        config.RepoName,
		Description: description,
		Private:     false, // Can be made configurable
		AutoInit:    true,
		Readme:      "extracted-code",
	}

	repo, err = s.giteaClient.CreateRepository(ctx, config.Owner, createReq)
	if err != nil {
		return nil, fmt.Errorf("create repository: %w", err)
	}

	s.logger.Printf("Created Gitea repository: %s", repo.HTMLURL)
	return repo, nil
}

// CreateRepositoryNode creates a knowledge graph node for the Gitea repository
func (s *GiteaStorage) CreateRepositoryNode(repo *Repository, config StorageConfig, repoMeta *RepositoryMetadata) graph.Node {
	repoID := fmt.Sprintf("gitea_repo:%s/%s", config.Owner, config.RepoName)
	
	return graph.Node{
		ID:    repoID,
		Type:  "GiteaRepository",
		Label: repo.Name,
		Props: map[string]interface{}{
			"gitea_url":     repo.HTMLURL,
			"clone_url":     repo.CloneURL,
			"ssh_url":       repo.SSHURL,
			"owner":         config.Owner,
			"name":          repo.Name,
			"description":   repo.Description,
			"private":       repo.Private,
			"source_url":    repoMeta.URL,
			"source_commit": repoMeta.Commit,
			"source_branch": repoMeta.Branch,
			"project_id":    config.ProjectID,
			"system_id":     config.SystemID,
			"base_path":     config.BasePath,
			"created_at":    time.Now().Format(time.RFC3339),
		},
	}
}

