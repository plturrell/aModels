package git

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// GiteaClient handles interactions with Gitea API
type GiteaClient struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// NewGiteaClient creates a new Gitea API client
func NewGiteaClient(baseURL, token string) *GiteaClient {
	return &GiteaClient{
		baseURL: baseURL,
		token:   token,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Repository represents a Gitea repository
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

// CreateRepositoryRequest represents a request to create a repository
type CreateRepositoryRequest struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Private     bool   `json:"private"`
	AutoInit    bool   `json:"auto_init"`
	Readme      string `json:"readme,omitempty"`
}

// CreateRepository creates a new repository in Gitea
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

// GetRepository retrieves a repository by owner and name
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

// CreateOrUpdateFile creates or updates a file in the repository
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

// FileInfo represents file information from Gitea
type FileInfo struct {
	Name string `json:"name"`
	Path string `json:"path"`
	SHA  string `json:"sha"`
	Size int64  `json:"size"`
	Type string `json:"type"`
}

// GetFile retrieves file information
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

// encodeBase64 encodes content to base64
func encodeBase64(content string) string {
	return base64.StdEncoding.EncodeToString([]byte(content))
}

