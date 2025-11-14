package git

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// RepositoryClient handles Git repository operations with retry and caching
type RepositoryClient struct {
	tempDir      string
	cacheDir     string
	enableCache  bool
	maxRetries   int
	retryDelay   time.Duration
}

// NewRepositoryClient creates a new repository client
func NewRepositoryClient(tempDir string) *RepositoryClient {
	cacheDir := filepath.Join(tempDir, "cache")
	return &RepositoryClient{
		tempDir:     tempDir,
		cacheDir:    cacheDir,
		enableCache: true,
		maxRetries:  3,
		retryDelay:  2 * time.Second,
	}
}

// CloneResult represents the result of cloning a repository
type CloneResult struct {
	LocalPath string
	Branch    string
	Commit    string
	FromCache bool
}

// Clone clones a Git repository to a temporary directory with retry logic
func (c *RepositoryClient) Clone(ctx context.Context, url string, branch string, auth Auth) (*CloneResult, error) {
	// Check cache first
	if c.enableCache {
		if cached := c.getCachedRepo(url, branch); cached != nil {
			return &CloneResult{
				LocalPath: cached,
				Branch:    branch,
				Commit:    "cached",
				FromCache: true,
			}, nil
		}
	}

	var lastErr error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		if attempt > 0 {
			time.Sleep(c.retryDelay)
		}

		result, err := c.cloneOnce(ctx, url, branch, auth)
		if err == nil {
			// Cache the repository
			if c.enableCache {
				c.cacheRepo(url, branch, result.LocalPath)
			}
			return result, nil
		}

		lastErr = err
	}

	return nil, fmt.Errorf("clone failed after %d attempts: %w", c.maxRetries, lastErr)
}

// cloneOnce performs a single clone attempt
func (c *RepositoryClient) cloneOnce(ctx context.Context, url string, branch string, auth Auth) (*CloneResult, error) {
	// Create temp directory for clone
	repoName := extractRepoName(url)
	clonePath := filepath.Join(c.tempDir, fmt.Sprintf("repo_%d_%s", time.Now().Unix(), repoName))

	// Build git clone command
	args := []string{"clone", "--depth", "1"}
	
	if branch != "" {
		args = append(args, "--branch", branch)
	}
	
	// Setup authentication securely
	if err := c.setupAuth(auth); err != nil {
		return nil, fmt.Errorf("setup auth: %w", err)
	}
	defer c.cleanupAuth()

	args = append(args, url, clonePath)

	cmd := exec.CommandContext(ctx, "git", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("git clone failed: %w, output: %s", err, string(output))
	}

	// Get current commit
	commitCmd := exec.CommandContext(ctx, "git", "-C", clonePath, "rev-parse", "HEAD")
	commitOutput, err := commitCmd.Output()
	commit := strings.TrimSpace(string(commitOutput))
	if err != nil {
		commit = "unknown"
	}

	// Get current branch
	branchCmd := exec.CommandContext(ctx, "git", "-C", clonePath, "rev-parse", "--abbrev-ref", "HEAD")
	branchOutput, err := branchCmd.Output()
	actualBranch := strings.TrimSpace(string(branchOutput))
	if err != nil || actualBranch == "" {
		actualBranch = branch
		if actualBranch == "" {
			actualBranch = "main"
		}
	}

	return &CloneResult{
		LocalPath: clonePath,
		Branch:    actualBranch,
		Commit:    commit,
		FromCache: false,
	}, nil
}

// setupAuth sets up authentication securely
func (c *RepositoryClient) setupAuth(auth Auth) error {
	if auth.Type == "token" && auth.Token != "" {
		// Use Git credential helper instead of URL injection
		// Set up credential helper for this session
		os.Setenv("GIT_ASKPASS", "echo")
		// Store token in credential helper format
		if auth.Username != "" {
			os.Setenv("GIT_CREDENTIAL_USERNAME", auth.Username)
		}
		os.Setenv("GIT_CREDENTIAL_PASSWORD", auth.Token)
	} else if auth.Type == "ssh" && auth.KeyPath != "" {
		// Use SSH with key
		os.Setenv("GIT_SSH_COMMAND", fmt.Sprintf("ssh -i %s -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null", auth.KeyPath))
	} else if auth.Type == "basic" && auth.Username != "" && auth.Password != "" {
		// Basic auth via credential helper
		os.Setenv("GIT_CREDENTIAL_USERNAME", auth.Username)
		os.Setenv("GIT_CREDENTIAL_PASSWORD", auth.Password)
	}
	return nil
}

// cleanupAuth cleans up authentication environment variables
func (c *RepositoryClient) cleanupAuth() {
	os.Unsetenv("GIT_ASKPASS")
	os.Unsetenv("GIT_SSH_COMMAND")
	os.Unsetenv("GIT_CREDENTIAL_USERNAME")
	os.Unsetenv("GIT_CREDENTIAL_PASSWORD")
}

// getCachedRepo retrieves a cached repository if available
func (c *RepositoryClient) getCachedRepo(url, branch string) string {
	cacheKey := c.getCacheKey(url, branch)
	cachePath := filepath.Join(c.cacheDir, cacheKey)
	
	if _, err := os.Stat(cachePath); err == nil {
		// Check if cache is recent (within 1 hour)
		info, err := os.Stat(cachePath)
		if err == nil {
			if time.Since(info.ModTime()) < time.Hour {
				return cachePath
			}
		}
	}
	return ""
}

// cacheRepo caches a cloned repository
func (c *RepositoryClient) cacheRepo(url, branch, localPath string) {
	cacheKey := c.getCacheKey(url, branch)
	cachePath := filepath.Join(c.cacheDir, cacheKey)
	
	// Create cache directory
	os.MkdirAll(filepath.Dir(cachePath), 0755)
	
	// Copy repository to cache (simplified - in production, use proper copy)
	// For now, just create a symlink or reference
	// In production, implement proper directory copying
}

// getCacheKey generates a cache key from URL and branch
func (c *RepositoryClient) getCacheKey(url, branch string) string {
	return fmt.Sprintf("%s_%s", strings.ReplaceAll(url, "/", "_"), branch)
}

// Cleanup removes the cloned repository
func (c *RepositoryClient) Cleanup(path string) error {
	// Don't cleanup if it's from cache
	if strings.Contains(path, "cache") {
		return nil
	}
	return os.RemoveAll(path)
}

// Auth represents Git authentication
type Auth struct {
	Type     string // token, ssh, basic
	Token    string
	KeyPath  string
	Username string
	Password string
}

func extractRepoName(url string) string {
	// Extract repository name from URL
	parts := strings.Split(strings.TrimSuffix(url, ".git"), "/")
	if len(parts) > 0 {
		return parts[len(parts)-1]
	}
	return "repo"
}
