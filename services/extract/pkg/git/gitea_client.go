package git

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// GiteaClient handles interactions with Gitea API.
// It provides methods for repository management, file operations, and branch/commit queries.
type GiteaClient struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// NewGiteaClient creates a new Gitea API client.
// baseURL is the base URL of the Gitea instance (e.g., "https://gitea.example.com").
// token is the authentication token for API access.
func NewGiteaClient(baseURL, token string) *GiteaClient {
	return &GiteaClient{
		baseURL: baseURL,
		token:   token,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Token returns the Gitea authentication token
func (c *GiteaClient) Token() string {
	return c.token
}

// isRetryableError checks if an error should be retried
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	
	errStr := err.Error()
	// Retry on network errors, timeouts, and 5xx/429 status codes
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "connection") ||
		strings.Contains(errStr, "status 5") ||
		strings.Contains(errStr, "status 429")
}

// executeWithRetry executes an HTTP request with retry logic
func (c *GiteaClient) executeWithRetry(ctx context.Context, req *http.Request) (*http.Response, error) {
	maxRetries := 3
	initialDelay := 100 * time.Millisecond
	maxDelay := 5 * time.Second
	
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Calculate exponential backoff
			delay := time.Duration(float64(initialDelay) * float64(1<<uint(attempt-1)))
			if delay > maxDelay {
				delay = maxDelay
			}
			
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}
		
		// Clone request for retry (body can only be read once)
		var bodyReader io.Reader
		if req.Body != nil {
			bodyBytes, err := io.ReadAll(req.Body)
			if err != nil {
				return nil, fmt.Errorf("read request body: %w", err)
			}
			bodyReader = bytes.NewReader(bodyBytes)
			req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
		
		// Create new request for this attempt
		newReq := req.Clone(ctx)
		if bodyReader != nil {
			newReq.Body = io.NopCloser(bodyReader)
		}
		
		resp, err := c.httpClient.Do(newReq)
		if err == nil {
			// Check if status code is retryable
			if resp.StatusCode >= 500 || resp.StatusCode == 429 {
				resp.Body.Close()
				lastErr = fmt.Errorf("retryable status %d", resp.StatusCode)
				continue
			}
			return resp, nil
		}
		
		lastErr = err
		if !isRetryableError(err) {
			return nil, err
		}
	}
	
	return nil, fmt.Errorf("request failed after %d retries: %w", maxRetries+1, lastErr)
}

// Repository represents a Gitea repository with its metadata.
type Repository struct {
	ID          int64  `json:"id"`
	Name        string `json:"name"`
	FullName    string `json:"full_name"`
	Description string `json:"description"`
	Private     bool   `json:"private"`
	CloneURL    string `json:"clone_url"`
	SSHURL      string `json:"ssh_url"`
	HTMLURL     string `json:"html_url"`
}

// CreateRepositoryRequest represents a request to create a new repository.
type CreateRepositoryRequest struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Private     bool   `json:"private"`
	AutoInit    bool   `json:"auto_init"`
	Readme      string `json:"readme,omitempty"`
}

// CreateRepository creates a new repository in Gitea.
// If owner is empty, the repository is created for the authenticated user.
// If owner is provided, the repository is created for that organization.
// Returns the created repository or an error if the operation fails.
func (c *GiteaClient) CreateRepository(ctx context.Context, owner string, req CreateRepositoryRequest) (*Repository, error) {
	url := fmt.Sprintf("%s/api/v1/user/repos", c.baseURL)
	if owner != "" {
		url = fmt.Sprintf("%s/api/v1/orgs/%s/repos", c.baseURL, owner)
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("create repository failed with status %d: %s", resp.StatusCode, string(body))
	}

	var repo Repository
	if err := json.NewDecoder(resp.Body).Decode(&repo); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &repo, nil
}

// GetRepository retrieves a repository by owner and name.
// Returns the repository details or an error if not found or access is denied.
func (c *GiteaClient) GetRepository(ctx context.Context, owner, name string) (*Repository, error) {
	url := fmt.Sprintf("%s/api/v1/repos/%s/%s", c.baseURL, owner, name)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("get repository failed with status %d: %s", resp.StatusCode, string(body))
	}

	var repo Repository
	if err := json.NewDecoder(resp.Body).Decode(&repo); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &repo, nil
}

// FileContent represents file content for Gitea API
type FileContent struct {
	Content string `json:"content"` // Base64 encoded
	Message string `json:"message"`
	Branch  string `json:"branch,omitempty"`
}

// CreateOrUpdateFile creates or updates a file in the repository.
// If the file exists, it updates it; otherwise, it creates a new file.
// path is the file path within the repository.
// content is the file content (will be base64 encoded).
// message is the commit message.
// branch is the target branch (defaults to "main" if empty).
func (c *GiteaClient) CreateOrUpdateFile(ctx context.Context, owner, repo, path, content, message, branch string) error {
	url := fmt.Sprintf("%s/api/v1/repos/%s/%s/contents/%s", c.baseURL, owner, repo, path)

	// Check if file exists
	existingFile, err := c.GetFile(ctx, owner, repo, path, branch)
	if err == nil && existingFile != nil {
		// Update existing file
		updateReq := map[string]interface{}{
			"content":  encodeBase64(content),
			"message":  message,
			"branch":   branch,
			"sha":      existingFile.SHA,
		}
		return c.updateFile(ctx, url, updateReq)
	}

	// Create new file
	createReq := map[string]interface{}{
		"content": encodeBase64(content),
		"message": message,
		"branch":  branch,
	}
	return c.createFile(ctx, url, createReq)
}

// FileInfo represents file or directory information from Gitea.
type FileInfo struct {
	Name string `json:"name"`
	Path string `json:"path"`
	SHA  string `json:"sha"`
	Size int64  `json:"size"`
	Type string `json:"type"`
}

// GetFile retrieves file information from the repository.
// path is the file path within the repository.
// ref is the branch, tag, or commit SHA (defaults to default branch if empty).
// Returns file metadata including SHA, size, and type (file or dir).
func (c *GiteaClient) GetFile(ctx context.Context, owner, repo, path, ref string) (*FileInfo, error) {
	url := fmt.Sprintf("%s/api/v1/repos/%s/%s/contents/%s", c.baseURL, owner, repo, path)
	if ref != "" {
		url += "?ref=" + ref
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("get file failed with status %d", resp.StatusCode)
	}

	var file FileInfo
	if err := json.NewDecoder(resp.Body).Decode(&file); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &file, nil
}

// createFile creates a new file in the repository.
// This is an internal helper method used by CreateOrUpdateFile.
func (c *GiteaClient) createFile(ctx context.Context, url string, req map[string]interface{}) error {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("create file failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// updateFile updates an existing file in the repository.
// This is an internal helper method used by CreateOrUpdateFile.
func (c *GiteaClient) updateFile(ctx context.Context, url string, req map[string]interface{}) error {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(jsonData))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("update file failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// encodeBase64 encodes content to base64 as required by the Gitea API.
func encodeBase64(content string) string {
	return base64.StdEncoding.EncodeToString([]byte(content))
}

// PaginationOptions represents pagination parameters for list operations.
// Page is 1-based (first page is 1).
// Limit specifies the maximum number of items per page (typically 1-100).
type PaginationOptions struct {
	Page  int // Page number (1-based)
	Limit int // Items per page
}

// PaginatedResponse represents a paginated API response.
// This generic type can be used for any paginated data.
type PaginatedResponse[T any] struct {
	Data     []T  `json:"data"`
	Page     int  `json:"page"`
	Limit    int  `json:"limit"`
	Total    int  `json:"total,omitempty"`
	HasMore  bool `json:"has_more"`
}

// ListRepositories lists repositories for a user or organization
// If pagination is provided, returns paginated results
func (c *GiteaClient) ListRepositories(ctx context.Context, owner string, pagination *PaginationOptions) ([]Repository, *PaginationOptions, error) {
	baseURL := fmt.Sprintf("%s/api/v1/user/repos", c.baseURL)
	if owner != "" {
		baseURL = fmt.Sprintf("%s/api/v1/users/%s/repos", c.baseURL, owner)
	}

	// Build URL with pagination
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, nil, fmt.Errorf("parse URL: %w", err)
	}
	
	params := url.Values{}
	if pagination != nil {
		if pagination.Page > 0 {
			params.Set("page", fmt.Sprintf("%d", pagination.Page))
		}
		if pagination.Limit > 0 {
			params.Set("limit", fmt.Sprintf("%d", pagination.Limit))
		}
	}
	if len(params) > 0 {
		u.RawQuery = params.Encode()
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.executeWithRetry(ctx, httpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, nil, fmt.Errorf("list repositories failed with status %d: %s", resp.StatusCode, string(body))
	}

	var repos []Repository
	if err := json.NewDecoder(resp.Body).Decode(&repos); err != nil {
		return nil, nil, fmt.Errorf("decode response: %w", err)
	}

	// Determine if there are more pages (if limit was set and we got limit items)
	hasMore := false
	if pagination != nil && pagination.Limit > 0 && len(repos) == pagination.Limit {
		hasMore = true
	}

	resultPagination := &PaginationOptions{
		Page:  pagination.Page,
		Limit: pagination.Limit,
	}
	if pagination == nil {
		resultPagination = nil
	}

	return repos, resultPagination, nil
}

// Branch represents a Gitea branch with its latest commit information.
type Branch struct {
	Name   string `json:"name"`
	Commit struct {
		ID      string `json:"id"`
		Message string `json:"message"`
		Author  struct {
			Name  string `json:"name"`
			Email string `json:"email"`
			Date  string `json:"date"`
		} `json:"author"`
	} `json:"commit"`
}

// ListBranches lists all branches for a repository.
// Returns a slice of branches with their commit information.
func (c *GiteaClient) ListBranches(ctx context.Context, owner, repo string) ([]Branch, error) {
	url := fmt.Sprintf("%s/api/v1/repos/%s/%s/branches", c.baseURL, owner, repo)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("list branches failed with status %d: %s", resp.StatusCode, string(body))
	}

	var branches []Branch
	if err := json.NewDecoder(resp.Body).Decode(&branches); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return branches, nil
}

// Commit represents a Gitea commit with author and committer information.
type Commit struct {
	ID      string `json:"id"`
	Message string `json:"message"`
	Author  struct {
		Name  string `json:"name"`
		Email string `json:"email"`
		Date  string `json:"date"`
	} `json:"author"`
	Committer struct {
		Name  string `json:"name"`
		Email string `json:"email"`
		Date  string `json:"date"`
	} `json:"committer"`
	URL string `json:"url"`
}

// ListCommits lists commits for a repository
// If pagination is provided, uses page and limit. Otherwise uses limit parameter for backward compatibility.
func (c *GiteaClient) ListCommits(ctx context.Context, owner, repo, branch string, limit int, pagination *PaginationOptions) ([]Commit, *PaginationOptions, error) {
	baseURL := fmt.Sprintf("%s/api/v1/repos/%s/%s/commits", c.baseURL, owner, repo)
	
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, nil, fmt.Errorf("parse URL: %w", err)
	}
	
	params := url.Values{}
	if branch != "" {
		params.Set("sha", branch)
	}
	
	// Use pagination if provided, otherwise use limit for backward compatibility
	if pagination != nil {
		if pagination.Page > 0 {
			params.Set("page", fmt.Sprintf("%d", pagination.Page))
		}
		if pagination.Limit > 0 {
			params.Set("limit", fmt.Sprintf("%d", pagination.Limit))
		}
	} else if limit > 0 {
		params.Set("limit", fmt.Sprintf("%d", limit))
	}
	
	u.RawQuery = params.Encode()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.executeWithRetry(ctx, httpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, nil, fmt.Errorf("list commits failed with status %d: %s", resp.StatusCode, string(body))
	}

	var commits []Commit
	if err := json.NewDecoder(resp.Body).Decode(&commits); err != nil {
		return nil, nil, fmt.Errorf("decode response: %w", err)
	}

	// Determine if there are more pages
	hasMore := false
	if pagination != nil && pagination.Limit > 0 && len(commits) == pagination.Limit {
		hasMore = true
	}

	resultPagination := &PaginationOptions{
		Page:  pagination.Page,
		Limit: pagination.Limit,
	}
	if pagination == nil {
		resultPagination = nil
	}

	return commits, resultPagination, nil
}

// ListFiles lists files in a directory (or returns single file info)
// Note: Gitea API doesn't support pagination for file listings, but we accept pagination for consistency
func (c *GiteaClient) ListFiles(ctx context.Context, owner, repo, path, ref string, pagination *PaginationOptions) ([]FileInfo, *PaginationOptions, error) {
	baseURL := fmt.Sprintf("%s/api/v1/repos/%s/%s/contents/%s", c.baseURL, owner, repo, path)
	
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, nil, fmt.Errorf("parse URL: %w", err)
	}
	
	params := url.Values{}
	if ref != "" {
		params.Set("ref", ref)
	}
	// Note: Gitea doesn't support pagination for file listings, but we keep the interface consistent
	if pagination != nil && pagination.Page > 0 {
		params.Set("page", fmt.Sprintf("%d", pagination.Page))
	}
	if pagination != nil && pagination.Limit > 0 {
		params.Set("limit", fmt.Sprintf("%d", pagination.Limit))
	}
	
	u.RawQuery = params.Encode()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.executeWithRetry(ctx, httpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, nil, fmt.Errorf("list files failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Gitea API returns either a single FileInfo or an array of FileInfo
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("read response: %w", err)
	}

	// Try to decode as array first
	var files []FileInfo
	if err := json.Unmarshal(bodyBytes, &files); err == nil {
		// Apply client-side pagination if needed (Gitea doesn't support server-side pagination for files)
		resultPagination := pagination
		if pagination != nil && pagination.Limit > 0 {
			start := (pagination.Page - 1) * pagination.Limit
			end := start + pagination.Limit
			if start >= len(files) {
				files = []FileInfo{}
			} else {
				if end > len(files) {
					end = len(files)
				}
				files = files[start:end]
			}
		}
		return files, resultPagination, nil
	}

	// If not an array, try as single file
	var file FileInfo
	if err := json.Unmarshal(bodyBytes, &file); err == nil {
		return []FileInfo{file}, pagination, nil
	}

	return nil, nil, fmt.Errorf("unable to decode response as file or file array")
}

// GetFileContent retrieves the raw content of a file from the repository.
// path is the file path within the repository.
// ref is the branch, tag, or commit SHA (defaults to default branch if empty).
// Returns the file content as a string (decoded from base64).
func (c *GiteaClient) GetFileContent(ctx context.Context, owner, repo, path, ref string) (string, error) {
	url := fmt.Sprintf("%s/api/v1/repos/%s/%s/contents/%s", c.baseURL, owner, repo, path)
	if ref != "" {
		url += "?ref=" + ref
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("get file content failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Gitea returns file content in a ContentResponse structure
	type ContentResponse struct {
		Type     string `json:"type"`
		Encoding string `json:"encoding"`
		Size     int64  `json:"size"`
		Name     string `json:"name"`
		Path     string `json:"path"`
		Content  string `json:"content"` // Base64 encoded
		SHA      string `json:"sha"`
	}

	var contentResp ContentResponse
	if err := json.NewDecoder(resp.Body).Decode(&contentResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	// Decode base64 content
	if contentResp.Encoding == "base64" {
		decoded, err := base64.StdEncoding.DecodeString(contentResp.Content)
		if err != nil {
			return "", fmt.Errorf("decode base64 content: %w", err)
		}
		return string(decoded), nil
	}

	return contentResp.Content, nil
}

// DeleteRepository deletes a repository from Gitea.
// This is a destructive operation that cannot be undone.
// owner is the repository owner (user or organization).
// repo is the repository name.
// Returns an error if the repository doesn't exist or access is denied.
func (c *GiteaClient) DeleteRepository(ctx context.Context, owner, repo string) error {
	url := fmt.Sprintf("%s/api/v1/repos/%s/%s", c.baseURL, owner, repo)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("token %s", c.token))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("delete repository failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

