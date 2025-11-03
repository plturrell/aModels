
package server

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/langchain-ai/langgraph-go/pkg/cli"
)

// Server is an HTTP server for running graphs.
type Server struct {
	addr   string
	logger cli.Logger
	apiKey string
}

// NewServer creates a new server.
func NewServer(addr string, logger cli.Logger, apiKey string) *Server {
	return &Server{
		addr:   addr,
		logger: logger,
		apiKey: apiKey,
	}
}

// Run starts the server.
func (s *Server) Run() error {
	http.HandleFunc("/runs", s.handleRun)
	return http.ListenAndServe(s.addr, nil)
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	if s.apiKey != "" && r.Header.Get("X-API-Key") != s.apiKey {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req cli.RunConfig
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := cli.RunProject(context.Background(), req, s.logger); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}
