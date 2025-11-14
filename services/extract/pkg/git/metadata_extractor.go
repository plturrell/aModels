package git

import (
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"time"
)

// MetadataExtractor extracts Git metadata from a repository
type MetadataExtractor struct {
}

// NewMetadataExtractor creates a new metadata extractor
func NewMetadataExtractor() *MetadataExtractor {
	return &MetadataExtractor{}
}

// RepositoryMetadata represents Git repository metadata
type RepositoryMetadata struct {
	URL         string
	Branch      string
	Commit      string
	CommitHash  string
	Author      string
	AuthorEmail string
	CommitDate  time.Time
	Message     string
	FileCount   int
}

// ExtractMetadata extracts metadata from a cloned repository
func (e *MetadataExtractor) ExtractMetadata(ctx context.Context, repoPath string, url string) (*RepositoryMetadata, error) {
	meta := &RepositoryMetadata{
		URL: url,
	}

	// Get current branch
	branchCmd := exec.CommandContext(ctx, "git", "-C", repoPath, "rev-parse", "--abbrev-ref", "HEAD")
	branchOutput, err := branchCmd.Output()
	if err == nil {
		meta.Branch = strings.TrimSpace(string(branchOutput))
	}

	// Get current commit
	commitCmd := exec.CommandContext(ctx, "git", "-C", repoPath, "rev-parse", "HEAD")
	commitOutput, err := commitCmd.Output()
	if err == nil {
		meta.CommitHash = strings.TrimSpace(string(commitOutput))
		meta.Commit = meta.CommitHash[:8] // Short hash
	}

	// Get commit info
	logCmd := exec.CommandContext(ctx, "git", "-C", repoPath, "log", "-1", "--format=%an|%ae|%ai|%s")
	logOutput, err := logCmd.Output()
	if err == nil {
		parts := strings.SplitN(strings.TrimSpace(string(logOutput)), "|", 4)
		if len(parts) >= 4 {
			meta.Author = parts[0]
			meta.AuthorEmail = parts[1]
			if t, err := time.Parse("2006-01-02 15:04:05 -0700", parts[2]); err == nil {
				meta.CommitDate = t
			}
			meta.Message = parts[3]
		}
	}

	// Count files
	countCmd := exec.CommandContext(ctx, "git", "-C", repoPath, "ls-files")
	countOutput, err := countCmd.Output()
	if err == nil {
		meta.FileCount = len(strings.Split(strings.TrimSpace(string(countOutput)), "\n"))
	}

	return meta, nil
}

// ToJSON converts metadata to JSON
func (m *RepositoryMetadata) ToJSON() (string, error) {
	data, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

