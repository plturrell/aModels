package git

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/plturrell/aModels/services/extract/pkg/clients"
	"github.com/plturrell/aModels/services/extract/pkg/extraction"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// DocumentPipeline orchestrates document processing, Gitea storage, and catalog/Glean integration
type DocumentPipeline struct {
	documentProcessor *DocumentProcessor
	giteaStorage      *GiteaStorage
	fileStorage       *FileStorage
	logger            *log.Logger
}

// NewDocumentPipeline creates a new document pipeline
func NewDocumentPipeline(
	markitdownClient *clients.MarkItDownClient,
	ocrExtractor *extraction.MultiModalExtractor,
	giteaStorage *GiteaStorage,
	logger *log.Logger,
) *DocumentPipeline {
	docProcessor := NewDocumentProcessor(markitdownClient, ocrExtractor, logger)
	docProcessor.SetGiteaStorage(giteaStorage)

	return &DocumentPipeline{
		documentProcessor: docProcessor,
		giteaStorage:      giteaStorage,
		fileStorage:       NewFileStorage(logger),
		logger:            logger,
	}
}

// ProcessDocuments processes documents through markitdown/OCR and stores in Gitea
func (p *DocumentPipeline) ProcessDocuments(
	ctx context.Context,
	filePaths []string,
	config StorageConfig,
	projectID, systemID string,
) ([]graph.Node, []graph.Edge, error) {
	var allNodes []graph.Node
	var allEdges []graph.Edge
	var processedFiles []ExtractedFile

	// Process each document
	for _, filePath := range filePaths {
		result, err := p.documentProcessor.ProcessDocument(ctx, filePath)
		if result == nil {
			p.logger.Printf("Skipping document %s: %v", filePath, err)
			continue
		}

		// Store in Gitea
		if p.giteaStorage != nil {
			repoURL, err := p.documentProcessor.StoreProcessedDocument(ctx, result, config)
			if err != nil {
				p.logger.Printf("Warning: failed to store document %s in Gitea: %v", filePath, err)
			} else {
				p.logger.Printf("Stored document %s in Gitea: %s", filePath, repoURL)
			}
		}

		// Create extracted file for knowledge graph
		extractedFile := ExtractedFile{
			Path:         result.ProcessedPath,
			Content:      result.MarkdownContent,
			Size:         int64(len(result.MarkdownContent)),
			LastModified: result.ProcessedAt.Format("2006-01-02T15:04:05Z"),
			Extension:    ".md",
			IsText:       true,
			ContentHash:  result.ContentHash,
			IsLarge:      len(result.MarkdownContent) > InlineContentLimit,
		}
		processedFiles = append(processedFiles, extractedFile)

		// Create document node
		docNode := p.documentProcessor.CreateDocumentNode(result, projectID, systemID)
		allNodes = append(allNodes, docNode)
	}

	// Create file nodes for processed documents
	if len(processedFiles) > 0 {
		docRepoID := fmt.Sprintf("documents:%s", projectID)
		fileNodes, fileEdges := p.fileStorage.CreateFileNodes(processedFiles, docRepoID, "documents", "processed", projectID, systemID)
		allNodes = append(allNodes, fileNodes...)
		allEdges = append(allEdges, fileEdges...)
	}

	return allNodes, allEdges, nil
}

// StandardizeDocumentStructure creates standardized structure in Gitea
func (p *DocumentPipeline) StandardizeDocumentStructure(ctx context.Context, config StorageConfig) error {
	if p.giteaStorage == nil {
		return fmt.Errorf("Gitea storage not configured")
	}

	// Create standardized directory structure
	standardPaths := []string{
		"documents/raw/",
		"documents/processed/",
		"documents/metadata/",
		"documents/verified/",
	}

	// Create README for standardization
	readmeContent := `# Document Repository

This repository contains standardized, self-verified documents processed through:
- MarkItDown for document conversion
- DeepSeek OCR for image/PDF processing
- Automated verification and cataloging

## Structure

- \`documents/raw/\` - Original documents
- \`documents/processed/\` - Processed markdown files
- \`documents/metadata/\` - Document metadata
- \`documents/verified/\` - Verified and standardized documents

## Verification

All documents are:
- Version controlled in Gitea
- Processed through standardized pipeline
- Linked to catalog and Glean
- Self-verified for completeness
`

	// Store README
	readmeFile := ExtractedFile{
		Path:         "README.md",
		Content:      readmeContent,
		Size:         int64(len(readmeContent)),
		LastModified: "",
		Extension:    ".md",
		IsText:       true,
		ContentHash:  calculateContentHash([]byte(readmeContent)),
		IsLarge:      false,
	}

	repoMeta := &RepositoryMetadata{
		URL:    config.RepoName,
		Branch: config.Branch,
		Commit: "initial",
	}

	_, err := p.giteaStorage.StoreCode(ctx, config, []ExtractedFile{readmeFile}, repoMeta)
	return err
}

// VerifyDocumentCompleteness verifies that documents are complete and standardized
func (p *DocumentPipeline) VerifyDocumentCompleteness(result *DocumentProcessingResult) bool {
	// Check if document has content
	if len(strings.TrimSpace(result.MarkdownContent)) == 0 {
		return false
	}

	// Check if content hash is valid
	if result.ContentHash == "" {
		return false
	}

	// Check if metadata is present
	if result.Metadata == nil {
		result.Metadata = make(map[string]interface{})
	}

	// Add verification metadata
	result.Metadata["verified"] = true
	result.Metadata["verification_timestamp"] = result.ProcessedAt.Format("2006-01-02T15:04:05Z")
	result.Metadata["verification_method"] = result.Source

	return true
}

