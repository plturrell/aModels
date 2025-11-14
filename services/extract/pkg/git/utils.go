package git

import (
	"path/filepath"
	"strings"
)

// IsDocumentFile checks if a file is a document that should be processed
func IsDocumentFile(filePath string) bool {
	ext := strings.ToLower(filepath.Ext(filePath))
	
	// Document formats supported by markitdown
	markitdownFormats := []string{
		".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
		".html", ".htm", ".txt", ".md", ".rtf", ".csv", ".json", ".xml", ".epub",
	}
	
	// Image formats for OCR
	ocrFormats := []string{
		".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
	}
	
	for _, format := range markitdownFormats {
		if ext == format {
			return true
		}
	}
	
	for _, format := range ocrFormats {
		if ext == format {
			return true
		}
	}
	
	return false
}

