package sdk_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/langchain-ai/langgraph-go/pkg/sdk"
)

func TestHealthSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client, err := sdk.NewClient(sdk.Config{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health unexpected error: %v", err)
	}
}

func TestRunGraph(t *testing.T) {
	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/graphs/run" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		capturedAuth = r.Header.Get("Authorization")
		var req sdk.RunGraphRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.GraphID != "demo" {
			t.Fatalf("expected graph id demo, got %s", req.GraphID)
		}
		_ = json.NewEncoder(w).Encode(sdk.RunGraphResponse{RunID: "123", Status: "submitted"})
	}))
	defer srv.Close()

	client, err := sdk.NewClient(sdk.Config{BaseURL: srv.URL, APIKey: "token"})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	resp, err := client.RunGraph(context.Background(), sdk.RunGraphRequest{GraphID: "demo", Input: map[string]any{"value": 1}})
	if err != nil {
		t.Fatalf("RunGraph error: %v", err)
	}
	if resp.RunID != "123" {
		t.Fatalf("unexpected run id %s", resp.RunID)
	}
	if capturedAuth != "Bearer token" {
		t.Fatalf("expected auth header, got %s", capturedAuth)
	}
}

func TestGetRun(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || !strings.HasPrefix(r.URL.Path, "/api/runs/") {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(sdk.GetRunResponse{RunID: "123", Status: "completed", Output: map[string]any{"ok": true}})
	}))
	defer srv.Close()

	client, err := sdk.NewClient(sdk.Config{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	resp, err := client.GetRun(context.Background(), "123")
	if err != nil {
		t.Fatalf("GetRun error: %v", err)
	}
	if resp.Status != "completed" {
		t.Fatalf("unexpected status %s", resp.Status)
	}
}

func TestDeployAndDelete(t *testing.T) {
	var lastMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lastMethod = r.Method
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/deployments":
			_ = json.NewEncoder(w).Encode(sdk.DeployResponse{DeploymentID: "dep-1", Status: "created"})
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/api/deployments/"):
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client, err := sdk.NewClient(sdk.Config{BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	resp, err := client.DeployGraph(context.Background(), sdk.DeployRequest{GraphID: "g", Name: "demo"})
	if err != nil {
		t.Fatalf("DeployGraph error: %v", err)
	}
	if resp.DeploymentID != "dep-1" {
		t.Fatalf("unexpected deployment id %s", resp.DeploymentID)
	}
	if lastMethod != http.MethodPost {
		t.Fatalf("expected POST, got %s", lastMethod)
	}

	if err := client.DeleteDeployment(context.Background(), "dep-1"); err != nil {
		t.Fatalf("DeleteDeployment error: %v", err)
	}
	if lastMethod != http.MethodDelete {
		t.Fatalf("expected DELETE, got %s", lastMethod)
	}
}

func TestClientBasePathPreserved(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client, err := sdk.NewClient(sdk.Config{BaseURL: srv.URL + "/root"})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health error: %v", err)
	}
	if gotPath != "/root/health" {
		t.Fatalf("expected request path /root/health, got %s", gotPath)
	}
}
