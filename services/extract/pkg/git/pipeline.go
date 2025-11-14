package git

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
	"github.com/plturrell/aModels/services/extract/pkg/pipeline"
)

// Pipeline orchestrates Git repository processing
type Pipeline struct {
	repoClient  *RepositoryClient
	extractor   *CodeExtractor
	metaExtractor *MetadataExtractor
	logger      *log.Logger
}

// NewPipeline creates a new Git pipeline
func NewPipeline(tempDir string, logger *log.Logger) *Pipeline {
	return &Pipeline{
		repoClient:    NewRepositoryClient(tempDir),
		extractor:     NewCodeExtractor(),
		metaExtractor: NewMetadataExtractor(),
		logger:        logger,
	}
}

// ProcessRepository processes a Git repository and extracts code files
func (p *Pipeline) ProcessRepository(ctx context.Context, repo pipeline.GitRepository) ([]ExtractedFile, *RepositoryMetadata, error) {
	// Clone repository
	p.logger.Printf("Cloning repository: %s (branch: %s)", repo.URL, repo.Branch)
	
	auth := Auth{
		Type:     repo.Auth.Type,
		Token:    repo.Auth.Token,
		KeyPath:  repo.Auth.KeyPath,
		Username: repo.Auth.Username,
		Password: repo.Auth.Password,
	}

	cloneResult, err := p.repoClient.Clone(ctx, repo.URL, repo.Branch, auth)
	if err != nil {
		return nil, nil, fmt.Errorf("clone repository: %w", err)
	}

	// Ensure cleanup
	defer func() {
		if err := p.repoClient.Cleanup(cloneResult.LocalPath); err != nil {
			p.logger.Printf("Warning: failed to cleanup repository: %v", err)
		}
	}()

	// Extract metadata
	meta, err := p.metaExtractor.ExtractMetadata(ctx, cloneResult.LocalPath, repo.URL)
	if err != nil {
		p.logger.Printf("Warning: failed to extract metadata: %v", err)
		meta = &RepositoryMetadata{URL: repo.URL, Branch: cloneResult.Branch, Commit: cloneResult.Commit}
	}

	// Extract files
	files, err := p.extractor.ExtractFiles(cloneResult.LocalPath, repo.FilePatterns)
	if err != nil {
		return nil, meta, fmt.Errorf("extract files: %w", err)
	}

	p.logger.Printf("Extracted %d files from repository", len(files))
	return files, meta, nil
}

// CreateRepositoryNodes creates knowledge graph nodes for a repository
func (p *Pipeline) CreateRepositoryNodes(meta *RepositoryMetadata, projectID, systemID string) ([]graph.Node, []graph.Edge) {
	var nodes []graph.Node
	var edges []graph.Edge

	// Repository node
	repoID := fmt.Sprintf("repository:%s", filepath.Base(meta.URL))
	repoNode := graph.Node{
		ID:    repoID,
		Type:  "Repository",
		Label: filepath.Base(meta.URL),
		Props: map[string]interface{}{
			"url":          meta.URL,
			"branch":       meta.Branch,
			"commit":       meta.Commit,
			"commit_hash":  meta.CommitHash,
			"author":       meta.Author,
			"author_email": meta.AuthorEmail,
			"commit_date":  meta.CommitDate.Format(time.RFC3339),
			"message":      meta.Message,
			"file_count":   meta.FileCount,
			"project_id":   projectID,
			"system_id":    systemID,
		},
	}
	nodes = append(nodes, repoNode)

	return nodes, edges
}

