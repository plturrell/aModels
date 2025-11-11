package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// FilePersistence is the persistence layer for files.
type FilePersistence struct {
	baseDir string
}

// NewFilePersistence creates a new file persistence layer.
func NewFilePersistence(baseDir string) (*FilePersistence, error) {
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}
	return &FilePersistence{baseDir: baseDir}, nil
}

// SaveDocument saves a document to the file system.
func (p *FilePersistence) SaveDocument(path string) error {
	srcFile, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer srcFile.Close()

	destPath := filepath.Join(p.baseDir, filepath.Base(path))
	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, srcFile)
	if err != nil {
		return fmt.Errorf("failed to copy file: %w", err)
	}

	return nil
}
