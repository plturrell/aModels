package git

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"
)

const (
	// MaxFileSize is the maximum size for a single file (10MB)
	MaxFileSize = 10 * 1024 * 1024
	// MaxTotalSize is the maximum total size for all files in a repository (100MB)
	MaxTotalSize = 100 * 1024 * 1024
	// InlineContentLimit is the size limit for storing content inline in Neo4j (10KB)
	InlineContentLimit = 10 * 1024
)

// CodeExtractor extracts code files from a cloned repository
type CodeExtractor struct {
	maxFileSize  int64
	maxTotalSize int64
}

// NewCodeExtractor creates a new code extractor
func NewCodeExtractor() *CodeExtractor {
	return &CodeExtractor{
		maxFileSize:  MaxFileSize,
		maxTotalSize: MaxTotalSize,
	}
}

// ExtractedFile represents a file extracted from a repository
type ExtractedFile struct {
	Path         string
	Content      string
	Size         int64
	LastModified string
	Extension    string
	IsText       bool
	ContentHash  string // SHA256 hash for deduplication
	IsLarge      bool   // True if content exceeds inline limit
}

// ExtractFiles extracts files matching patterns from a repository
func (e *CodeExtractor) ExtractFiles(repoPath string, patterns []string) ([]ExtractedFile, error) {
	var files []ExtractedFile
	var totalSize int64
	var mu sync.Mutex

	// Compile patterns for better matching
	compiledPatterns := compilePatterns(patterns)

	err := filepath.Walk(repoPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			// Skip hidden directories and .git
			if strings.HasPrefix(info.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}

		// Check file size limit
		if info.Size() > e.maxFileSize {
			return nil // Skip oversized files
		}

		// Check total size limit
		mu.Lock()
		if totalSize+info.Size() > e.maxTotalSize {
			mu.Unlock()
			return fmt.Errorf("total size limit exceeded")
		}
		totalSize += info.Size()
		mu.Unlock()

		// Check if file matches any pattern
		relPath, err := filepath.Rel(repoPath, path)
		if err != nil {
			return err
		}

		if !matchesPatterns(relPath, compiledPatterns) {
			return nil
		}

		// Read file content
		content, err := e.readFileContent(path, info.Size())
		if err != nil {
			return fmt.Errorf("read file %s: %w", path, err)
		}

		// Detect if file is text
		isText := isTextFile(content)

		// Calculate content hash for deduplication
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

		mu.Lock()
		files = append(files, extractedFile)
		mu.Unlock()

		return nil
	})

	return files, err
}

// ExtractFilesParallel extracts files in parallel for better performance
func (e *CodeExtractor) ExtractFilesParallel(repoPath string, patterns []string, maxWorkers int) ([]ExtractedFile, error) {
	if maxWorkers <= 0 {
		maxWorkers = 10
	}

	var files []ExtractedFile
	var totalSize int64
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, 1)
	fileChan := make(chan os.FileInfo, 100)

	compiledPatterns := compilePatterns(patterns)

	// Walk files and send to channel
	go func() {
		defer close(fileChan)
		filepath.Walk(repoPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() {
				if strings.HasPrefix(info.Name(), ".") {
					return filepath.SkipDir
				}
				return nil
			}

			if info.Size() > e.maxFileSize {
				return nil
			}

			relPath, err := filepath.Rel(repoPath, path)
			if err != nil {
				return err
			}

			if matchesPatterns(relPath, compiledPatterns) {
				fileChan <- info
			}

			return nil
		})
	}()

	// Process files in parallel
	sem := make(chan struct{}, maxWorkers)
	for info := range fileChan {
		wg.Add(1)
		sem <- struct{}{}
		go func(info os.FileInfo) {
			defer wg.Done()
			defer func() { <-sem }()

			mu.Lock()
			if totalSize+info.Size() > e.maxTotalSize {
				mu.Unlock()
				select {
				case errChan <- fmt.Errorf("total size limit exceeded"):
				default:
				}
				return
			}
			totalSize += info.Size()
			mu.Unlock()

			path := filepath.Join(repoPath, info.Name())
			content, err := e.readFileContent(path, info.Size())
			if err != nil {
				return
			}

			hash := calculateContentHash(content)
			extractedFile := ExtractedFile{
				Path:         info.Name(),
				Content:      string(content),
				Size:         info.Size(),
				LastModified: info.ModTime().Format("2006-01-02T15:04:05Z"),
				Extension:    strings.ToLower(filepath.Ext(info.Name())),
				IsText:       isTextFile(content),
				ContentHash:  hash,
				IsLarge:      len(content) > InlineContentLimit,
			}

			mu.Lock()
			files = append(files, extractedFile)
			mu.Unlock()
		}(info)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	select {
	case err := <-errChan:
		return files, err
	default:
		return files, nil
	}
}

// readFileContent reads file content with size checking
func (e *CodeExtractor) readFileContent(path string, size int64) ([]byte, error) {
	if size > e.maxFileSize {
		return nil, fmt.Errorf("file size %d exceeds limit %d", size, e.maxFileSize)
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Limit reader to max file size
	limitedReader := io.LimitReader(file, e.maxFileSize+1)
	content, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, err
	}

	// Check if file was truncated
	if int64(len(content)) > e.maxFileSize {
		return nil, fmt.Errorf("file size exceeds limit")
	}

	return content, nil
}

// isTextFile checks if content is text (not binary)
func isTextFile(content []byte) bool {
	if len(content) == 0 {
		return true
	}

	// Check for null bytes (binary indicator)
	if bytes.Contains(content, []byte{0}) {
		return false
	}

	// Check if content is valid UTF-8
	if !utf8.Valid(content) {
		return false
	}

	// Additional heuristics: check for common text file indicators
	textIndicators := []string{"function", "class", "import", "package", "SELECT", "CREATE", "INSERT"}
	contentStr := strings.ToLower(string(content[:min(1000, len(content))]))
	for _, indicator := range textIndicators {
		if strings.Contains(contentStr, strings.ToLower(indicator)) {
			return true
		}
	}

	return true
}

// calculateContentHash calculates SHA256 hash of content for deduplication
func calculateContentHash(content []byte) string {
	hash := sha256.Sum256(content)
	return hex.EncodeToString(hash[:])
}

// Pattern represents a compiled pattern for matching
type Pattern struct {
	Original string
	IsGlob   bool
	IsExt    bool
	Ext      string
}

// compilePatterns compiles patterns for efficient matching
func compilePatterns(patterns []string) []Pattern {
	if len(patterns) == 0 {
		return nil // Match all if no patterns
	}

	compiled := make([]Pattern, 0, len(patterns))
	for _, p := range patterns {
		pattern := Pattern{Original: p}

		// Check if it's a file extension pattern
		if strings.HasPrefix(p, "*") {
			pattern.IsExt = true
			pattern.Ext = strings.TrimPrefix(p, "*")
		} else if strings.Contains(p, "*") || strings.Contains(p, "?") {
			pattern.IsGlob = true
		}

		compiled = append(compiled, pattern)
	}
	return compiled
}

// matchesPatterns checks if a file path matches any of the compiled patterns
func matchesPatterns(path string, patterns []Pattern) bool {
	if len(patterns) == 0 {
		return true
	}

	pathLower := strings.ToLower(path)

	for _, pattern := range patterns {
		// Extension matching
		if pattern.IsExt {
			if strings.HasSuffix(pathLower, pattern.Ext) {
				return true
			}
		}

		// Glob matching
		if pattern.IsGlob {
			matched, _ := filepath.Match(pattern.Original, path)
			if matched {
				return true
			}
			// Also try with ** support (simple approximation)
			if strings.Contains(pattern.Original, "**") {
				simplePattern := strings.ReplaceAll(pattern.Original, "**/", "")
				simplePattern = strings.ReplaceAll(simplePattern, "**", "*")
				if matched, _ := filepath.Match(simplePattern, filepath.Base(path)); matched {
					return true
				}
			}
		}

		// Substring matching (for patterns like "sql/" or "src/")
		if strings.Contains(pathLower, strings.ToLower(pattern.Original)) {
			return true
		}

		// Exact match
		if pathLower == strings.ToLower(pattern.Original) {
			return true
		}
	}

	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
