package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/plturrell/aModels/services/orchestration/api"
)

func main() {
	// Create logger
	logger := log.New(os.Stdout, "[orchestration] ", log.LstdFlags|log.Lshortfile)

	// Create Perplexity handler
	perplexityHandler, err := api.NewPerplexityHandler(logger)
	if err != nil {
		logger.Fatalf("Failed to create Perplexity handler: %v", err)
	}

	// Create DMS handler
	dmsHandler, err := api.NewDMSHandler(logger)
	if err != nil {
		logger.Fatalf("Failed to create DMS handler: %v", err)
	}

	// Create Relational handler
	relationalHandler, err := api.NewRelationalHandler(logger)
	if err != nil {
		logger.Fatalf("Failed to create Relational handler: %v", err)
	}

	// Create Murex handler
	murexHandler, err := api.NewMurexHandler(logger)
	if err != nil {
		logger.Fatalf("Failed to create Murex handler: %v", err)
	}

	// Setup HTTP routes
	mux := http.NewServeMux()

	// Perplexity API endpoints
	mux.HandleFunc("/api/perplexity/process", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleProcessDocuments(w, r)
	})

	mux.HandleFunc("/api/perplexity/status/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleGetStatus(w, r)
	})

	mux.HandleFunc("/api/perplexity/results/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Check if it's intelligence endpoint
		if r.URL.Path[len("/api/perplexity/results/"):] != "" && r.URL.Path[len("/api/perplexity/results/"):] != "intelligence" {
			// Check if path ends with /intelligence
			path := r.URL.Path
			if len(path) > len("/api/perplexity/results/") && path[len(path)-len("/intelligence"):] == "/intelligence" {
				perplexityHandler.HandleGetIntelligence(w, r)
				return
			}
		}
		perplexityHandler.HandleGetResults(w, r)
	})

	mux.HandleFunc("/api/perplexity/results/", func(w http.ResponseWriter, r *http.Request) {
		// Handle intelligence endpoint
		path := r.URL.Path
		if len(path) > len("/api/perplexity/results/") && path[len(path)-len("/intelligence"):] == "/intelligence" {
			perplexityHandler.HandleGetIntelligence(w, r)
			return
		}
		perplexityHandler.HandleGetResults(w, r)
	})

	mux.HandleFunc("/api/perplexity/history", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleGetHistory(w, r)
	})

	mux.HandleFunc("/api/perplexity/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleSearchQuery(w, r)
	})

	mux.HandleFunc("/api/perplexity/learning/report", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleGetLearningReport(w, r)
	})

	mux.HandleFunc("/api/perplexity/jobs/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleCancelJob(w, r)
	})

	mux.HandleFunc("/api/perplexity/batch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		perplexityHandler.HandleBatchProcess(w, r)
	})

	// DMS API endpoints
	mux.HandleFunc("/api/dms/process", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleProcessDocuments(w, r)
	})

	mux.HandleFunc("/api/dms/status/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleGetStatus(w, r)
	})

	mux.HandleFunc("/api/dms/results/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Check if it's intelligence endpoint
		path := r.URL.Path
		if len(path) > len("/api/dms/results/") && path[len(path)-len("/intelligence"):] == "/intelligence" {
			dmsHandler.HandleGetIntelligence(w, r)
			return
		}
		// Check if it's export endpoint
		if len(path) > len("/api/dms/results/") && path[len(path)-len("/export"):] == "/export" {
			dmsHandler.HandleExportResults(w, r)
			return
		}
		dmsHandler.HandleGetResults(w, r)
	})

	mux.HandleFunc("/api/dms/history", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleGetHistory(w, r)
	})

	mux.HandleFunc("/api/dms/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleSearchQuery(w, r)
	})

	mux.HandleFunc("/api/dms/jobs/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleCancelJob(w, r)
	})

	mux.HandleFunc("/api/dms/batch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleBatchProcess(w, r)
	})

	mux.HandleFunc("/api/dms/graph/", func(w http.ResponseWriter, r *http.Request) {
		// Handle knowledge graph query
		path := r.URL.Path
		if len(path) > len("/api/dms/graph/") && path[len(path)-len("/query"):] == "/query" {
			if r.Method != http.MethodPost {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}
			dmsHandler.HandleKnowledgeGraphQuery(w, r)
			return
		}
		// Handle relationships endpoint
		if len(path) > len("/api/dms/graph/") && path[len(path)-len("/relationships"):] == "/relationships" {
			if r.Method != http.MethodGet {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}
			// For now, return relationships from results/intelligence
			// This could be enhanced later
			http.Error(w, "Not implemented", http.StatusNotImplemented)
			return
		}
		http.Error(w, "Not found", http.StatusNotFound)
	})

	mux.HandleFunc("/api/dms/domains/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleDomainQuery(w, r)
	})

	mux.HandleFunc("/api/dms/catalog/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleCatalogSearch(w, r)
	})

	mux.HandleFunc("/api/dms/documents/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		dmsHandler.HandleGetDocument(w, r)
	})

	// Relational API endpoints
	mux.HandleFunc("/api/relational/process", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleProcessTables(w, r)
	})

	mux.HandleFunc("/api/relational/status/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleGetStatus(w, r)
	})

	mux.HandleFunc("/api/relational/results/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Check if it's intelligence endpoint
		path := r.URL.Path
		if len(path) > len("/api/relational/results/") && path[len(path)-len("/intelligence"):] == "/intelligence" {
			relationalHandler.HandleGetIntelligence(w, r)
			return
		}
		// Check if it's export endpoint
		if len(path) > len("/api/relational/results/") && path[len(path)-len("/export"):] == "/export" {
			relationalHandler.HandleExportResults(w, r)
			return
		}
		relationalHandler.HandleGetResults(w, r)
	})

	mux.HandleFunc("/api/relational/history", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleGetHistory(w, r)
	})

	mux.HandleFunc("/api/relational/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleSearchQuery(w, r)
	})

	mux.HandleFunc("/api/relational/jobs/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleCancelJob(w, r)
	})

	mux.HandleFunc("/api/relational/batch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleBatchProcess(w, r)
	})

	mux.HandleFunc("/api/relational/graph/", func(w http.ResponseWriter, r *http.Request) {
		// Handle knowledge graph query
		path := r.URL.Path
		if len(path) > len("/api/relational/graph/") && path[len(path)-len("/query"):] == "/query" {
			if r.Method != http.MethodPost {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}
			relationalHandler.HandleKnowledgeGraphQuery(w, r)
			return
		}
		http.Error(w, "Not found", http.StatusNotFound)
	})

	mux.HandleFunc("/api/relational/domains/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleDomainQuery(w, r)
	})

	mux.HandleFunc("/api/relational/catalog/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		relationalHandler.HandleCatalogSearch(w, r)
	})

	// Murex API endpoints
	mux.HandleFunc("/api/murex/process", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		murexHandler.HandleProcessTrades(w, r)
	})

	mux.HandleFunc("/api/murex/status/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		murexHandler.HandleGetStatus(w, r)
	})

	mux.HandleFunc("/api/murex/results/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Check if it's intelligence endpoint
		path := r.URL.Path
		if len(path) > len("/api/murex/results/") && path[len(path)-len("/intelligence"):] == "/intelligence" {
			murexHandler.HandleGetIntelligence(w, r)
			return
		}
		murexHandler.HandleGetResults(w, r)
	})

	mux.HandleFunc("/api/murex/history", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		murexHandler.HandleGetHistory(w, r)
	})

	// Health check
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// CORS middleware
	corsHandler := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	}

	// Get port from environment
	port := os.Getenv("ORCHESTRATION_PORT")
	if port == "" {
		port = "8080"
	}

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      corsHandler(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.Printf("Starting orchestration server on port %s", port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("Server failed to start: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Println("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Fatalf("Server forced to shutdown: %v", err)
	}

	logger.Println("Server exited")
}

