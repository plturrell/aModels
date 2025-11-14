package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// TestVerifyWebhookSignature tests the webhook signature verification function
func TestVerifyWebhookSignature(t *testing.T) {
	tests := []struct {
		name      string
		payload   []byte
		secret    string
		signature string
		want      bool
	}{
		{
			name:      "valid signature",
			payload:   []byte(`{"test":"data"}`),
			secret:    "my-secret",
			signature: computeSignature([]byte(`{"test":"data"}`), "my-secret"),
			want:      true,
		},
		{
			name:      "valid signature with sha256 prefix",
			payload:   []byte(`{"test":"data"}`),
			secret:    "my-secret",
			signature: "sha256=" + computeSignature([]byte(`{"test":"data"}`), "my-secret"),
			want:      true,
		},
		{
			name:      "invalid signature",
			payload:   []byte(`{"test":"data"}`),
			secret:    "my-secret",
			signature: "invalid-signature",
			want:      false,
		},
		{
			name:      "empty secret",
			payload:   []byte(`{"test":"data"}`),
			secret:    "",
			signature: "any-signature",
			want:      false,
		},
		{
			name:      "empty signature",
			payload:   []byte(`{"test":"data"}`),
			secret:    "my-secret",
			signature: "",
			want:      false,
		},
		{
			name:      "wrong secret",
			payload:   []byte(`{"test":"data"}`),
			secret:    "wrong-secret",
			signature: computeSignature([]byte(`{"test":"data"}`), "my-secret"),
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := verifyWebhookSignature(tt.payload, tt.signature, tt.secret)
			if got != tt.want {
				t.Errorf("verifyWebhookSignature() = %v, want %v", got, tt.want)
			}
		})
	}
}

// computeSignature computes HMAC-SHA256 signature for testing
func computeSignature(payload []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(payload)
	return hex.EncodeToString(mac.Sum(nil))
}

// TestHandleGiteaWebhook_SignatureVerification tests webhook signature verification
func TestHandleGiteaWebhook_SignatureVerification(t *testing.T) {
	// Create test server
	logger := log.New(io.Discard, "", 0)
	server := &extractServer{
		logger: logger,
	}

	// Test payload
	payload := GiteaWebhookPayload{
		Action: "push",
		Ref:    "refs/heads/main",
		After:  "abc123",
		Repository: struct {
			ID       int64  `json:"id"`
			Name     string `json:"name"`
			FullName string `json:"full_name"`
			HTMLURL  string `json:"html_url"`
			CloneURL string `json:"clone_url"`
			Owner    struct {
				Login    string `json:"login"`
				UserName string `json:"username"`
				FullName string `json:"full_name"`
			} `json:"owner"`
		}{
			FullName: "owner/repo",
			CloneURL: "https://gitea.example.com/owner/repo.git",
		},
		Commits: []struct {
			ID      string `json:"id"`
			Message string `json:"message"`
			URL     string `json:"url"`
			Author  struct {
				Name     string `json:"name"`
				Email    string `json:"email"`
				Username string `json:"username"`
			} `json:"author"`
			Committer struct {
				Name     string `json:"name"`
				Email    string `json:"email"`
				Username string `json:"username"`
			} `json:"committer"`
			Added    []string `json:"added"`
			Removed  []string `json:"removed"`
			Modified []string `json:"modified"`
		}{
			{
				Added: []string{"test.yaml"},
			},
		},
	}

	payloadBytes, _ := json.Marshal(payload)

	tests := []struct {
		name           string
		webhookSecret  string
		signature      string
		expectedStatus int
	}{
		{
			name:           "valid signature",
			webhookSecret:  "test-secret",
			signature:      computeSignature(payloadBytes, "test-secret"),
			expectedStatus: http.StatusAccepted,
		},
		{
			name:           "invalid signature",
			webhookSecret:  "test-secret",
			signature:      "invalid-signature",
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "no secret configured (skip verification)",
			webhookSecret:  "",
			signature:      "",
			expectedStatus: http.StatusAccepted,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set environment variable
			if tt.webhookSecret != "" {
				os.Setenv("GITEA_WEBHOOK_SECRET", tt.webhookSecret)
			} else {
				os.Unsetenv("GITEA_WEBHOOK_SECRET")
			}
			defer os.Unsetenv("GITEA_WEBHOOK_SECRET")

			// Create request
			req := httptest.NewRequest(http.MethodPost, "/webhooks/gitea", bytes.NewReader(payloadBytes))
			req.Header.Set("Content-Type", "application/json")
			if tt.signature != "" {
				req.Header.Set("X-Gitea-Signature", tt.signature)
			}

			// Record response
			rr := httptest.NewRecorder()

			// Call handler
			server.handleGiteaWebhook(rr, req)

			// Check status code
			if status := rr.Code; status != tt.expectedStatus {
				t.Errorf("handler returned wrong status code: got %v want %v", status, tt.expectedStatus)
				t.Errorf("response body: %s", rr.Body.String())
			}
		})
	}
}

// TestIsRelevantFile tests the file relevance checker
func TestIsRelevantFile(t *testing.T) {
	tests := []struct {
		name string
		path string
		want bool
	}{
		{"YAML config", "project-config.yaml", true},
		{"YML config", "settings-config.yml", true},
		{"JSON file", "data.json", true},
		{"HQL file", "query.hql", true},
		{"DDL file", "schema.ddl", true},
		{"SQL file", "script.sql", true},
		{"XML file", "workflow.xml", true},
		{"Config in subdirectory", "configs/app-config.yaml", true},
		{"Random text file", "readme.txt", false},
		{"Go source", "main.go", false},
		{"Python source", "script.py", false},
		{"Markdown", "README.md", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRelevantFile(tt.path)
			if got != tt.want {
				t.Errorf("isRelevantFile(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

// TestExtractOwnerRepo tests owner/repo extraction from various formats
func TestExtractOwnerRepo(t *testing.T) {
	tests := []struct {
		name      string
		fullName  string
		cloneURL  string
		wantOwner string
		wantRepo  string
	}{
		{
			name:      "from full name",
			fullName:  "myuser/myrepo",
			cloneURL:  "",
			wantOwner: "myuser",
			wantRepo:  "myrepo",
		},
		{
			name:      "from clone URL",
			fullName:  "",
			cloneURL:  "https://gitea.example.com/owner/repo.git",
			wantOwner: "owner",
			wantRepo:  "repo",
		},
		{
			name:      "full name takes precedence",
			fullName:  "user1/repo1",
			cloneURL:  "https://gitea.example.com/user2/repo2.git",
			wantOwner: "user1",
			wantRepo:  "repo1",
		},
		{
			name:      "invalid formats",
			fullName:  "invalid",
			cloneURL:  "not-a-url",
			wantOwner: "",
			wantRepo:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOwner, gotRepo := extractOwnerRepo(tt.fullName, tt.cloneURL)
			if gotOwner != tt.wantOwner || gotRepo != tt.wantRepo {
				t.Errorf("extractOwnerRepo() = (%v, %v), want (%v, %v)",
					gotOwner, gotRepo, tt.wantOwner, tt.wantRepo)
			}
		})
	}
}

// TestExtractBranch tests branch name extraction from refs
func TestExtractBranch(t *testing.T) {
	tests := []struct {
		name string
		ref  string
		want string
	}{
		{"main branch", "refs/heads/main", "main"},
		{"develop branch", "refs/heads/develop", "develop"},
		{"feature branch", "refs/heads/feature/new-feature", "feature/new-feature"},
		{"invalid ref", "invalid-ref", "main"},
		{"empty ref", "", "main"},
		{"tag ref", "refs/tags/v1.0.0", "main"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractBranch(tt.ref)
			if got != tt.want {
				t.Errorf("extractBranch(%q) = %v, want %v", tt.ref, got, tt.want)
			}
		})
	}
}

// TestHasRelevantChanges tests the relevant changes detector
func TestHasRelevantChanges(t *testing.T) {
	tests := []struct {
		name    string
		payload *GiteaWebhookPayload
		want    bool
	}{
		{
			name: "has relevant added files",
			payload: &GiteaWebhookPayload{
				Commits: []struct {
					ID      string `json:"id"`
					Message string `json:"message"`
					URL     string `json:"url"`
					Author  struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"author"`
					Committer struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"committer"`
					Added    []string `json:"added"`
					Removed  []string `json:"removed"`
					Modified []string `json:"modified"`
				}{
					{
						Added: []string{"config.yaml", "readme.md"},
					},
				},
			},
			want: true,
		},
		{
			name: "has relevant modified files",
			payload: &GiteaWebhookPayload{
				Commits: []struct {
					ID      string `json:"id"`
					Message string `json:"message"`
					URL     string `json:"url"`
					Author  struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"author"`
					Committer struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"committer"`
					Added    []string `json:"added"`
					Removed  []string `json:"removed"`
					Modified []string `json:"modified"`
				}{
					{
						Modified: []string{"script.sql"},
					},
				},
			},
			want: true,
		},
		{
			name: "no relevant changes",
			payload: &GiteaWebhookPayload{
				Commits: []struct {
					ID      string `json:"id"`
					Message string `json:"message"`
					URL     string `json:"url"`
					Author  struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"author"`
					Committer struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"committer"`
					Added    []string `json:"added"`
					Removed  []string `json:"removed"`
					Modified []string `json:"modified"`
				}{
					{
						Added: []string{"README.md", "main.go"},
					},
				},
			},
			want: false,
		},
		{
			name: "empty commits",
			payload: &GiteaWebhookPayload{
				Commits: []struct {
					ID      string `json:"id"`
					Message string `json:"message"`
					URL     string `json:"url"`
					Author  struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"author"`
					Committer struct {
						Name     string `json:"name"`
						Email    string `json:"email"`
						Username string `json:"username"`
					} `json:"committer"`
					Added    []string `json:"added"`
					Removed  []string `json:"removed"`
					Modified []string `json:"modified"`
				}{},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasRelevantChanges(tt.payload)
			if got != tt.want {
				t.Errorf("hasRelevantChanges() = %v, want %v", got, tt.want)
			}
		})
	}
}
