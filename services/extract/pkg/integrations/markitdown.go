package integrations

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// MarkItDownClientInterface defines the interface for markitdown client to avoid import cycle
type MarkItDownClientInterface interface {
	ConvertFile(ctx context.Context, filePath string) (*MarkItDownResponse, error)
	ConvertBytes(ctx context.Context, fileData []byte, fileExtension string) (*MarkItDownResponse, error)
	IsFormatSupported(fileExtension string) bool
	HealthCheck(ctx context.Context) (bool, error)
}

// MarkItDownResponse represents the response from markitdown service.
type MarkItDownResponse struct {
	TextContent string            `json:"text_content"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	Format      string            `json:"format,omitempty"`
	Error       string            `json:"error,omitempty"`
}

// MarkItDownIntegration provides integration with markitdown service for document conversion.
type MarkItDownIntegration struct {
	client        MarkItDownClientInterface
	logger        *log.Logger
	enabled       bool
	fallbackToOCR bool
}

// NewMarkItDownIntegration creates a new markitdown integration.
// NOTE: client must be created outside this package to avoid import cycle
func NewMarkItDownIntegration(
	client MarkItDownClientInterface,
	logger *log.Logger,
) *MarkItDownIntegration {
	
	enabled := os.Getenv("MARKITDOWN_ENABLED")
	if enabled == "" {
		enabled = "true"
	}
	
	fallbackToOCR := os.Getenv("MARKITDOWN_FALLBACK_TO_OCR")
	if fallbackToOCR == "" {
		fallbackToOCR = "true"
	}

	return &MarkItDownIntegration{
		client:        client,
		logger:        logger,
		enabled:       enabled == "true" && client != nil,
		fallbackToOCR: fallbackToOCR == "true",
	}
}

// ConvertDocument converts a document to markdown using markitdown.
// Returns the markdown text and any error.
func (m *MarkItDownIntegration) ConvertDocument(ctx context.Context, filePath string) (string, error) {
	if !m.enabled || m.client == nil {
		return "", fmt.Errorf("markitdown integration not enabled")
	}

	// Check if format is supported
	ext := strings.ToLower(filepath.Ext(filePath))
	if !m.client.IsFormatSupported(ext) {
		if m.logger != nil {
			m.logger.Printf("Format %s not supported by markitdown, skipping", ext)
		}
		return "", fmt.Errorf("format %s not supported by markitdown", ext)
	}

	// Convert using markitdown
	response, err := m.client.ConvertFile(ctx, filePath)
	if err != nil {
		if m.fallbackToOCR && m.logger != nil {
			m.logger.Printf("MarkItDown conversion failed for %s: %v, will fallback to OCR", filePath, err)
		}
		return "", fmt.Errorf("markitdown conversion failed: %w", err)
	}

	return response.TextContent, nil
}

// ConvertDocumentBytes converts document bytes to markdown.
func (m *MarkItDownIntegration) ConvertDocumentBytes(ctx context.Context, fileData []byte, fileExtension string) (string, error) {
	if !m.enabled || m.client == nil {
		return "", fmt.Errorf("markitdown integration not enabled")
	}

	// Check if format is supported
	ext := strings.ToLower(fileExtension)
	if !m.client.IsFormatSupported(ext) {
		if m.logger != nil {
			m.logger.Printf("Format %s not supported by markitdown, skipping", ext)
		}
		return "", fmt.Errorf("format %s not supported by markitdown", ext)
	}

	// Convert using markitdown
	response, err := m.client.ConvertBytes(ctx, fileData, ext)
	if err != nil {
		if m.fallbackToOCR && m.logger != nil {
			m.logger.Printf("MarkItDown conversion failed (format: %s): %v, will fallback to OCR", ext, err)
		}
		return "", fmt.Errorf("markitdown conversion failed: %w", err)
	}

	return response.TextContent, nil
}

// ShouldUseMarkItDown determines if markitdown should be used for a given file.
func (m *MarkItDownIntegration) ShouldUseMarkItDown(filePath string) bool {
	if !m.enabled || m.client == nil {
		return false
	}

	ext := strings.ToLower(filepath.Ext(filePath))
	return m.client.IsFormatSupported(ext)
}

// HealthCheck checks if markitdown service is available.
func (m *MarkItDownIntegration) HealthCheck(ctx context.Context) (bool, error) {
	if !m.enabled || m.client == nil {
		return false, nil
	}
	return m.client.HealthCheck(ctx)
}

