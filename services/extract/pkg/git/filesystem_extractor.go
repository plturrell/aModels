package git

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// FileSystemExtractor extracts files from the file system
type FileSystemExtractor struct {
	extractor *CodeExtractor
}

// NewFileSystemExtractor creates a new file system extractor
func NewFileSystemExtractor() *FileSystemExtractor {
	return &FileSystemExtractor{
		extractor: NewCodeExtractor(),
	}
}

// ExtractFilesFromPaths extracts files from file system paths
func (f *FileSystemExtractor) ExtractFilesFromPaths(paths []string, patterns []string) ([]ExtractedFile, error) {
	var allFiles []ExtractedFile
	seenPaths := make(map[string]bool)

	for _, path := range paths {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}

		// Resolve absolute path
		absPath, err := filepath.Abs(path)
		if err != nil {
			continue
		}

		// Check if we've already processed this path
		if seenPaths[absPath] {
			continue
		}
		seenPaths[absPath] = true

		// Check if path exists
		info, err := os.Stat(absPath)
		if err != nil {
			continue
		}

		if info.IsDir() {
			// Extract files from directory
			files, err := f.extractFromDirectory(absPath, patterns)
			if err != nil {
				continue
			}
			allFiles = append(allFiles, files...)
		} else {
			// Extract single file
			file, err := f.extractSingleFile(absPath, patterns)
			if err != nil {
				continue
			}
			if file != nil {
				allFiles = append(allFiles, *file)
			}
		}
	}

	return allFiles, nil
}

// extractFromDirectory extracts files from a directory
func (f *FileSystemExtractor) extractFromDirectory(dirPath string, patterns []string) ([]ExtractedFile, error) {
	var files []ExtractedFile
	var totalSize int64

	compiledPatterns := compilePatterns(patterns)

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			// Skip hidden directories
			if strings.HasPrefix(info.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}

		// Check file size limit
		if info.Size() > MaxFileSize {
			return nil // Skip oversized files
		}

		// Check total size limit
		if totalSize+info.Size() > MaxTotalSize {
			return fmt.Errorf("total size limit exceeded")
		}
		totalSize += info.Size()

		// Get relative path from directory
		relPath, err := filepath.Rel(dirPath, path)
		if err != nil {
			return err
		}

		// Check if file matches patterns
		if len(compiledPatterns) > 0 && !matchesPatterns(relPath, compiledPatterns) {
			return nil
		}

		// Read file content
		content, err := f.readFileContent(path, info.Size())
		if err != nil {
			return fmt.Errorf("read file %s: %w", path, err)
		}

		// Detect if file is text
		isText := isTextFile(content)

		// Calculate content hash
		hash := calculateContentHash(content)

		extractedFile := ExtractedFile{
			Path:         relPath,
			Content:      string(content),
			Size:         info.Size(),
			LastModified: info.ModTime().Format("2006-01-02T15:04:05Z"),
			Extension:    strings.ToLower(filepath.Ext(relPath)),
			IsText:       isText,
			ContentHash:  hash,
			IsLarge:      len(content) > InlineContentLimit,
		}

		files = append(files, extractedFile)
		return nil
	})

	return files, err
}

// extractSingleFile extracts a single file
func (f *FileSystemExtractor) extractSingleFile(filePath string, patterns []string) (*ExtractedFile, error) {
	info, err := os.Stat(filePath)
	if err != nil {
		return nil, err
	}

	// Check file size limit
	if info.Size() > MaxFileSize {
		return nil, fmt.Errorf("file size %d exceeds limit", info.Size())
	}

	// Check if file matches patterns
	if len(patterns) > 0 {
		fileName := filepath.Base(filePath)
		compiledPatterns := compilePatterns(patterns)
		if !matchesPatterns(fileName, compiledPatterns) {
			return nil, nil // File doesn't match patterns, skip silently
		}
	}

	// Read file content
	content, err := f.readFileContent(filePath, info.Size())
	if err != nil {
		return nil, err
	}

	// Detect if file is text
	isText := isTextFile(content)

	// Calculate content hash
	hash := calculateContentHash(content)

		// Use a clean path for storage - just filename for single files
		// or relative path if processing directories
		storagePath := filepath.Base(filePath)
		
		extractedFile := &ExtractedFile{
			Path:         storagePath,
			Content:      string(content),
			Size:         info.Size(),
			LastModified: info.ModTime().Format("2006-01-02T15:04:05Z"),
			Extension:    strings.ToLower(filepath.Ext(filePath)),
			IsText:       isText,
			ContentHash:  hash,
			IsLarge:      len(content) > InlineContentLimit,
		}
		
		// Store original path in a separate field if needed (for reference)
		// For now, we'll use the filename as the path

	return extractedFile, nil
}

// readFileContent reads file content with size checking
func (f *FileSystemExtractor) readFileContent(path string, size int64) ([]byte, error) {
	if size > MaxFileSize {
		return nil, fmt.Errorf("file size %d exceeds limit %d", size, MaxFileSize)
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Limit reader to max file size
	limitedReader := io.LimitReader(file, MaxFileSize+1)
	content, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, err
	}

	// Check if file was truncated
	if int64(len(content)) > MaxFileSize {
		return nil, fmt.Errorf("file size exceeds limit")
	}

	return content, nil
}

// ExtractFilesFromFileList extracts files from a list of file paths
// This is used for JSONTables, HiveDDLs, ControlMFiles, etc.
func (f *FileSystemExtractor) ExtractFilesFromFileList(filePaths []string) ([]ExtractedFile, error) {
	var allFiles []ExtractedFile

	for _, filePath := range filePaths {
		filePath = strings.TrimSpace(filePath)
		if filePath == "" {
			continue
		}

		// Resolve absolute path
		absPath, err := filepath.Abs(filePath)
		if err != nil {
			continue
		}

		// Extract single file
		file, err := f.extractSingleFile(absPath, nil) // No pattern filtering for explicit file lists
		if err != nil {
			continue
		}
		if file != nil {
			// For file system files, use relative path from common base
			// If it's a single file, use just the filename
			// If it's from a directory, use relative path
			allFiles = append(allFiles, *file)
		}
	}

	return allFiles, nil
}

