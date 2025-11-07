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

