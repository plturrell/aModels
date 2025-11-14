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

func TestResponseCache(t *testing.T) {
	cache := NewResponseCache(1 * time.Second)
	
	// Test Set and Get
	cache.Set("key1", "value1")
	value, found := cache.Get("key1")
	if !found {
		t.Error("Expected to find key1 in cache")
	}
	if value != "value1" {
		t.Errorf("Expected 'value1', got '%v'", value)
	}
	
	// Test expiration
	time.Sleep(2 * time.Second)
	_, found = cache.Get("key1")
	if found {
		t.Error("Expected key1 to be expired")
	}
	
	// Test Delete
	cache.Set("key2", "value2")
	cache.Delete("key2")
	_, found = cache.Get("key2")
	if found {
		t.Error("Expected key2 to be deleted")
	}
	
	// Test Clear
	cache.Set("key3", "value3")
	cache.Clear()
	_, found = cache.Get("key3")
	if found {
		t.Error("Expected cache to be cleared")
	}
}

