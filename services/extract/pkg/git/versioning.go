package git

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// VersionTracker tracks code versions and changes
type VersionTracker struct {
	logger *log.Logger
}

// NewVersionTracker creates a new version tracker
func NewVersionTracker(logger *log.Logger) *VersionTracker {
	return &VersionTracker{
		logger: logger,
	}
}

// FileVersion represents a version of a file
type FileVersion struct {
	Commit    string
	Content   string
	Timestamp time.Time
	Author    string
	Message   string
	Hash      string
}

// TrackVersion creates version nodes for file changes
func (v *VersionTracker) TrackVersion(ctx context.Context, repoPath, filePath, currentCommit string) (*FileVersion, error) {
	// Get file content at current commit
	content, err := v.getFileAtCommit(ctx, repoPath, filePath, currentCommit)
	if err != nil {
		return nil, fmt.Errorf("get file at commit: %w", err)
	}

	// Get commit information
	commitInfo, err := v.getCommitInfo(ctx, repoPath, currentCommit)
	if err != nil {
		return nil, fmt.Errorf("get commit info: %w", err)
	}

	return &FileVersion{
		Commit:    currentCommit,
		Content:   content,
		Timestamp: commitInfo.Timestamp,
		Author:    commitInfo.Author,
		Message:   commitInfo.Message,
		Hash:      calculateContentHash([]byte(content)),
	}, nil
}

// CommitInfo represents Git commit information
type CommitInfo struct {
	Hash      string
	Author    string
	Email     string
	Timestamp time.Time
	Message   string
}

// getCommitInfo retrieves commit information
func (v *VersionTracker) getCommitInfo(ctx context.Context, repoPath, commit string) (*CommitInfo, error) {
	// Get commit details
	cmd := exec.CommandContext(ctx, "git", "-C", repoPath, "show", "-s", "--format=%H|%an|%ae|%ai|%s", commit)
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	parts := strings.Split(strings.TrimSpace(string(output)), "|")
	if len(parts) < 5 {
		return nil, fmt.Errorf("invalid commit format")
	}

	timestamp, err := time.Parse("2006-01-02 15:04:05 -0700", parts[3])
	if err != nil {
		timestamp = time.Now()
	}

	return &CommitInfo{
		Hash:      parts[0],
		Author:    parts[1],
		Email:     parts[2],
		Timestamp: timestamp,
		Message:   parts[4],
	}, nil
}

// getFileAtCommit retrieves file content at a specific commit
func (v *VersionTracker) getFileAtCommit(ctx context.Context, repoPath, filePath, commit string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", "-C", repoPath, "show", fmt.Sprintf("%s:%s", commit, filePath))
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(output), nil
}

// CreateVersionNode creates a knowledge graph node for a file version
func (v *VersionTracker) CreateVersionNode(fileID string, version *FileVersion) graph.Node {
	versionID := fmt.Sprintf("file_version:%s:%s", fileID, version.Commit)

	return graph.Node{
		ID:    versionID,
		Type:  "FileVersion",
		Label: fmt.Sprintf("Version %s", version.Commit[:8]),
		Props: map[string]interface{}{
			"commit":     version.Commit,
			"content":    version.Content,
			"timestamp":  version.Timestamp.Format(time.RFC3339),
			"author":     version.Author,
			"message":    version.Message,
			"content_hash": version.Hash,
			"file_id":    fileID,
		},
	}
}

// CreateVersionEdge creates an edge linking file to its version
func (v *VersionTracker) CreateVersionEdge(fileID, versionID string) graph.Edge {
	return graph.Edge{
		SourceID: fileID,
		TargetID: versionID,
		Label:    "HAS_VERSION",
		Props: map[string]interface{}{
			"relationship": "version",
		},
	}
}

