package git

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/clients"
	"github.com/plturrell/aModels/services/extract/pkg/extraction"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// DocumentProcessor handles document parsing, OCR, and Gitea storage
type DocumentProcessor struct {
	markitdownClient *clients.MarkItDownClient
	ocrExtractor     *extraction.MultiModalExtractor
	logger           *log.Logger
	giteaStorage     *GiteaStorage
}

// NewDocumentProcessor creates a new document processor
func NewDocumentProcessor(markitdownClient *clients.MarkItDownClient, ocrExtractor *extraction.MultiModalExtractor, logger *log.Logger) *DocumentProcessor {
	return &DocumentProcessor{
		markitdownClient: markitdownClient,
		ocrExtractor:     ocrExtractor,
		logger:           logger,
	}
}

// SetGiteaStorage sets the Gitea storage for document versioning
func (d *DocumentProcessor) SetGiteaStorage(storage *GiteaStorage) {
	d.giteaStorage = storage
}

// DocumentProcessingResult represents the result of document processing
type DocumentProcessingResult struct {
	OriginalPath    string
	ProcessedPath   string
	MarkdownContent string
	Metadata        map[string]interface{}
	OCRUsed         bool
	Source          string // "markitdown", "ocr", "direct"
	ContentHash     string
	ProcessedAt     time.Time
}

// ProcessDocument processes a document through markitdown and/or OCR
func (d *DocumentProcessor) ProcessDocument(ctx context.Context, filePath string) (*DocumentProcessingResult, error) {
	ext := strings.ToLower(filepath.Ext(filePath))
	
	result := &DocumentProcessingResult{
		OriginalPath: filePath,
		Metadata:     make(map[string]interface{}),
		ProcessedAt:  time.Now(),
	}

	// Try markitdown first for supported formats
	if d.markitdownClient != nil && d.markitdownClient.IsFormatSupported(ext) {
		markdown, metadata, err := d.processWithMarkitdown(ctx, filePath)
		if err == nil {
			result.MarkdownContent = markdown
			result.Metadata = metadata
			result.Source = "markitdown"
			result.ProcessedPath = filePath + ".md"
			result.ContentHash = calculateContentHash([]byte(markdown))
			return result, nil
		}
		d.logger.Printf("MarkItDown processing failed for %s: %v, trying OCR fallback", filePath, err)
	}

	// Fallback to OCR for images and PDFs
	if d.ocrExtractor != nil && d.isOCRSupported(ext) {
		markdown, err := d.processWithOCR(ctx, filePath)
		if err == nil {
			result.MarkdownContent = markdown
			result.OCRUsed = true
			result.Source = "ocr"
			result.ProcessedPath = filePath + ".md"
			result.ContentHash = calculateContentHash([]byte(markdown))
			return result, nil
		}
		d.logger.Printf("OCR processing failed for %s: %v", filePath, err)
	}

	// If no processing worked, read file directly
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	result.MarkdownContent = string(content)
	result.Source = "direct"
	result.ProcessedPath = filePath
	result.ContentHash = calculateContentHash(content)

	return result, nil
}

// processWithMarkitdown processes document using markitdown service
func (d *DocumentProcessor) processWithMarkitdown(ctx context.Context, filePath string) (string, map[string]interface{}, error) {
	// Use existing markitdown client
	response, err := d.markitdownClient.ConvertFile(ctx, filePath)
	if err != nil {
		return "", nil, fmt.Errorf("markitdown conversion: %w", err)
	}

	metadata := make(map[string]interface{})
	if response.Metadata != nil {
		for k, v := range response.Metadata {
			metadata[k] = v
		}
	}

	return response.TextContent, metadata, nil
}

// processWithOCR processes document using DeepSeek OCR
func (d *DocumentProcessor) processWithOCR(ctx context.Context, filePath string) (string, error) {
	if d.ocrExtractor == nil {
		return "", fmt.Errorf("OCR extractor not configured")
	}

	// Use multi-modal extractor for OCR
	ocrResult, err := d.ocrExtractor.ExtractFromImage(filePath, "")
	if err != nil {
		return "", fmt.Errorf("OCR extraction: %w", err)
	}

	// Combine text and tables into markdown
	markdown := ocrResult.Text
	if len(ocrResult.Tables) > 0 {
		markdown += "\n\n## Extracted Tables\n\n"
		for i, table := range ocrResult.Tables {
			markdown += fmt.Sprintf("### Table %d\n\n", i+1)
			markdown += tableToMarkdown(table)
			markdown += "\n\n"
		}
	}

	return markdown, nil
}

// tableToMarkdown converts ExtractedTable to markdown format
func tableToMarkdown(table extraction.ExtractedTable) string {
	var markdown strings.Builder
	
	// Write headers
	if len(table.Headers) > 0 {
		markdown.WriteString("| ")
		for _, header := range table.Headers {
			markdown.WriteString(header)
			markdown.WriteString(" | ")
		}
		markdown.WriteString("\n|")
		for range table.Headers {
			markdown.WriteString(" --- |")
		}
		markdown.WriteString("\n")
	}
	
	// Write rows
	for _, row := range table.Rows {
		markdown.WriteString("| ")
		for _, cell := range row {
			markdown.WriteString(cell)
			markdown.WriteString(" | ")
		}
		markdown.WriteString("\n")
	}
	
	return markdown.String()
}

// isMarkitdownSupported checks if format is supported by markitdown
func (d *DocumentProcessor) isMarkitdownSupported(ext string) bool {
	if d.markitdownClient == nil {
		return false
	}
	return d.markitdownClient.IsFormatSupported(ext)
}

// isOCRSupported checks if format is supported for OCR
func (d *DocumentProcessor) isOCRSupported(ext string) bool {
	supported := []string{".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
	for _, s := range supported {
		if ext == s {
			return true
		}
	}
	return false
}

// StoreProcessedDocument stores processed document in Gitea
func (d *DocumentProcessor) StoreProcessedDocument(ctx context.Context, result *DocumentProcessingResult, config StorageConfig) (string, error) {
	if d.giteaStorage == nil {
		return "", fmt.Errorf("Gitea storage not configured")
	}

	// Create extracted file from processed document
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

	// Create metadata for document source
	docRepoMeta := &RepositoryMetadata{
		URL:    result.OriginalPath,
		Branch: "main",
		Commit: result.ContentHash[:8],
	}

	// Store in Gitea
	repoURL, err := d.giteaStorage.StoreCode(ctx, config, []ExtractedFile{extractedFile}, docRepoMeta)
	if err != nil {
		return "", fmt.Errorf("store in Gitea: %w", err)
	}

	return repoURL, nil
}

// CreateDocumentNode creates a knowledge graph node for processed document
func (d *DocumentProcessor) CreateDocumentNode(result *DocumentProcessingResult, projectID, systemID string) graph.Node {
	docID := fmt.Sprintf("document:%s", result.ContentHash[:16])

	props := map[string]interface{}{
		"original_path":    result.OriginalPath,
		"processed_path":         result.ProcessedPath,
		"content":               result.MarkdownContent,
		"source":                result.Source,
		"ocr_used":              result.OCRUsed,
		"content_hash":          result.ContentHash,
		"processed_at":           result.ProcessedAt.Format(time.RFC3339),
		"project_id":             projectID,
		"system_id":              systemID,
		"document_type":          filepath.Ext(result.OriginalPath),
	}

	// Add metadata
	for k, v := range result.Metadata {
		props[fmt.Sprintf("meta_%s", k)] = v
	}

	return graph.Node{
		ID:    docID,
		Type:  graph.NodeTypeDocument,
		Label: filepath.Base(result.OriginalPath),
		Props: props,
	}
}


