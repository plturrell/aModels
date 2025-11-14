package git

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewGiteaClient(t *testing.T) {
	client := NewGiteaClient("https://gitea.example.com", "test-token")
	if client == nil {
		t.Fatal("NewGiteaClient returned nil")
	}
	if client.baseURL != "https://gitea.example.com" {
		t.Errorf("Expected baseURL 'https://gitea.example.com', got '%s'", client.baseURL)
	}
	if client.token != "test-token" {
		t.Errorf("Expected token 'test-token', got '%s'", client.token)
	}
}

func TestGiteaClient_ListRepositories(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/user/repos" {
			t.Errorf("Unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "token test-token" {
			t.Errorf("Missing or invalid Authorization header")
		}
		
		repos := []Repository{
			{ID: 1, Name: "repo1", FullName: "user/repo1"},
			{ID: 2, Name: "repo2", FullName: "user/repo2"},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(repos)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	repos, pagination, err := client.ListRepositories(ctx, "", nil)
	if err != nil {
		t.Fatalf("ListRepositories failed: %v", err)
	}
	if len(repos) != 2 {
		t.Errorf("Expected 2 repositories, got %d", len(repos))
	}
	if pagination != nil {
		t.Error("Expected nil pagination when not requested")
	}
}

func TestGiteaClient_ListRepositories_WithPagination(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify pagination parameters
		if r.URL.Query().Get("page") != "1" {
			t.Errorf("Expected page=1, got %s", r.URL.Query().Get("page"))
		}
		if r.URL.Query().Get("limit") != "10" {
			t.Errorf("Expected limit=10, got %s", r.URL.Query().Get("limit"))
		}
		
		repos := make([]Repository, 10)
		for i := 0; i < 10; i++ {
			repos[i] = Repository{ID: int64(i + 1), Name: fmt.Sprintf("repo%d", i+1)}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(repos)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	pagination := &PaginationOptions{Page: 1, Limit: 10}
	repos, resultPagination, err := client.ListRepositories(ctx, "", pagination)
	if err != nil {
		t.Fatalf("ListRepositories failed: %v", err)
	}
	if len(repos) != 10 {
		t.Errorf("Expected 10 repositories, got %d", len(repos))
	}
	if resultPagination == nil {
		t.Error("Expected pagination result")
	}
	if resultPagination.Page != 1 {
		t.Errorf("Expected page 1, got %d", resultPagination.Page)
	}
}

func TestGiteaClient_GetRepository(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/repos/owner/repo" {
			t.Errorf("Unexpected path: %s", r.URL.Path)
		}
		
		repo := Repository{
			ID:          1,
			Name:        "repo",
			FullName:    "owner/repo",
			Description: "Test repository",
			Private:     false,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(repo)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	repo, err := client.GetRepository(ctx, "owner", "repo")
	if err != nil {
		t.Fatalf("GetRepository failed: %v", err)
	}
	if repo.Name != "repo" {
		t.Errorf("Expected name 'repo', got '%s'", repo.Name)
	}
	if repo.FullName != "owner/repo" {
		t.Errorf("Expected full name 'owner/repo', got '%s'", repo.FullName)
	}
}

func TestGiteaClient_ListBranches(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		branches := []Branch{
			{Name: "main"},
			{Name: "develop"},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(branches)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	branches, err := client.ListBranches(ctx, "owner", "repo")
	if err != nil {
		t.Fatalf("ListBranches failed: %v", err)
	}
	if len(branches) != 2 {
		t.Errorf("Expected 2 branches, got %d", len(branches))
	}
}

func TestGiteaClient_CreateRepository(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if !strings.Contains(r.URL.Path, "/api/v1/") {
			t.Errorf("Unexpected path: %s", r.URL.Path)
		}
		
		repo := Repository{
			ID:          1,
			Name:        "test-repo",
			FullName:    "owner/test-repo",
			Description: "Test repository",
			Private:     false,
			CloneURL:    "https://gitea.example.com/owner/test-repo.git",
		}
		w.WriteHeader(http.StatusCreated)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(repo)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	req := CreateRepositoryRequest{
		Name:        "test-repo",
		Description: "Test repository",
		Private:     false,
		AutoInit:    true,
	}

	repo, err := client.CreateRepository(ctx, "owner", req)
	if err != nil {
		t.Fatalf("CreateRepository failed: %v", err)
	}
	if repo.Name != "test-repo" {
		t.Errorf("Expected name 'test-repo', got '%s'", repo.Name)
	}
}

func TestGiteaClient_CreateOrUpdateFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			// File doesn't exist
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method == http.MethodPost {
			w.WriteHeader(http.StatusCreated)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"message": "created"})
			return
		}
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	err := client.CreateOrUpdateFile(ctx, "owner", "repo", "test.txt", "content", "Add test file", "main")
	if err != nil {
		t.Fatalf("CreateOrUpdateFile failed: %v", err)
	}
}

func TestGiteaClient_RetryLogic(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(Repository{ID: 1, Name: "test"})
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	_, err := client.GetRepository(ctx, "owner", "repo")
	if err != nil {
		t.Fatalf("Expected retry to succeed, got error: %v", err)
	}
	if attempts != 3 {
		t.Errorf("Expected 3 attempts, got %d", attempts)
	}
}

func TestGiteaClient_ListCommits_Pagination(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		page := r.URL.Query().Get("page")
		limit := r.URL.Query().Get("limit")
		
		if page != "2" || limit != "5" {
			t.Errorf("Expected page=2 and limit=5, got page=%s limit=%s", page, limit)
		}
		
		commits := make([]Commit, 5)
		for i := 0; i < 5; i++ {
			commits[i] = Commit{ID: fmt.Sprintf("commit%d", i+6)}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(commits)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	pagination := &PaginationOptions{Page: 2, Limit: 5}
	commits, _, err := client.ListCommits(ctx, "owner", "repo", "main", 0, pagination)
	if err != nil {
		t.Fatalf("ListCommits failed: %v", err)
	}
	if len(commits) != 5 {
		t.Errorf("Expected 5 commits, got %d", len(commits))
	}
}

func TestGiteaClient_ErrorHandling(t *testing.T) {
	tests := []struct {
		name           string
		statusCode     int
		expectError    bool
	}{
		{"unauthorized", http.StatusUnauthorized, true},
		{"not found", http.StatusNotFound, true},
		{"rate limited", http.StatusTooManyRequests, true},
		{"server error", http.StatusInternalServerError, true},
		{"success", http.StatusOK, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				if tt.statusCode == http.StatusOK {
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(Repository{ID: 1, Name: "test"})
				}
			}))
			defer server.Close()

			client := NewGiteaClient(server.URL, "test-token")
			ctx := context.Background()

			_, err := client.GetRepository(ctx, "owner", "repo")
			if tt.expectError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

func TestGiteaClient_DeleteRepository(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Errorf("Expected DELETE, got %s", r.Method)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	err := client.DeleteRepository(ctx, "owner", "repo")
	if err != nil {
		t.Fatalf("DeleteRepository failed: %v", err)
	}
}

func TestGiteaClient_GetFileContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		type ContentResponse struct {
			Type     string `json:"type"`
			Encoding string `json:"encoding"`
			Content  string `json:"content"`
		}
		
		response := ContentResponse{
			Type:     "file",
			Encoding: "base64",
			Content:  "SGVsbG8gV29ybGQh", // "Hello World!" in base64
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewGiteaClient(server.URL, "test-token")
	ctx := context.Background()

	content, err := client.GetFileContent(ctx, "owner", "repo", "test.txt", "main")
	if err != nil {
		t.Fatalf("GetFileContent failed: %v", err)
	}
	if content != "Hello World!" {
		t.Errorf("Expected 'Hello World!', got '%s'", content)
	}
}

