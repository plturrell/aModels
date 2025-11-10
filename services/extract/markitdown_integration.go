package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// MarkItDownIntegration provides integration with markitdown service for document conversion.
type MarkItDownIntegration struct {
	client        *MarkItDownClient
	logger        *log.Logger
	enabled       bool
	fallbackToOCR bool
}

// NewMarkItDownIntegration creates a new markitdown integration.
func NewMarkItDownIntegration(
	baseURL string,
	logger *log.Logger,
	metricsCollector func(service, endpoint string, statusCode int, latency time.Duration, correlationID string),
) *MarkItDownIntegration {
	client := NewMarkItDownClient(baseURL, logger, metricsCollector)
	
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
		enabled:       enabled == "true",
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

