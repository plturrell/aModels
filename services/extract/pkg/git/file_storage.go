package git

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// FileStorage handles creating file nodes with raw code content
type FileStorage struct {
	logger         *log.Logger
	contentScanner *ContentScanner
}

// NewFileStorage creates a new file storage handler
func NewFileStorage(logger *log.Logger) *FileStorage {
	return &FileStorage{
		logger:         logger,
		contentScanner: NewContentScanner(),
	}
}

// CreateFileNodes creates knowledge graph nodes for files with raw code content
func (f *FileStorage) CreateFileNodes(files []ExtractedFile, repoID, repoURL, commit string, projectID, systemID string) ([]graph.Node, []graph.Edge) {
	var nodes []graph.Node
	var edges []graph.Edge

	for _, file := range files {
		// Skip binary files or very large files
		if !file.IsText {
			f.logger.Printf("Skipping binary file: %s", file.Path)
			continue
		}

		fileID := fmt.Sprintf("file:%s:%s", repoID, file.Path)
		
		// Scan content for secrets
		scanResult := f.contentScanner.Scan(file.Content)

		// Build file node properties
		props := map[string]interface{}{
			"path":          file.Path,
			"size":          file.Size,
			"last_modified": file.LastModified,
			"extension":     file.Extension,
			"repository_id": repoID,
			"repository_url": repoURL,
			"commit":        commit,
			"content_hash":  file.ContentHash,
			"is_large":      file.IsLarge,
			"project_id":    projectID,
			"system_id":     systemID,
			"source":        "git",
			"has_secrets":   scanResult.HasSecrets,
			"risk_level":    scanResult.RiskLevel,
		}

		// Add scan findings if secrets detected
		if scanResult.HasSecrets {
			props["security_findings"] = scanResult.Findings
			f.logger.Printf("Warning: Secrets detected in file %s (risk: %s)", file.Path, scanResult.RiskLevel)
		}

		// Store content inline for small files, reference for large files
		if !file.IsLarge {
			props["content"] = file.Content
		} else {
			// For large files, store a reference (could be to external storage)
			props["content_ref"] = fmt.Sprintf("file://%s/%s", repoID, file.Path)
			props["content_size"] = len(file.Content)
			// Still store truncated content for preview
			previewLen := min(1000, len(file.Content))
			props["content_preview"] = file.Content[:previewLen]
		}

		fileNode := graph.Node{
			ID:    fileID,
			Type:  "File",
			Label: filepath.Base(file.Path),
			Props: props,
		}
		nodes = append(nodes, fileNode)

		// Create edge from repository to file
		edges = append(edges, graph.Edge{
			SourceID: repoID,
			TargetID: fileID,
			Label:    "CONTAINS",
			Props: map[string]interface{}{
				"file_path": file.Path,
				"source":    "git",
			},
		})
	}

	return nodes, edges
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

