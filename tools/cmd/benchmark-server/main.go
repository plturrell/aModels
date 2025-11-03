package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"ai_benchmarks/internal/catalog/flightcatalog"
	"ai_benchmarks/internal/localai"
	catalogprompt "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightcatalog/prompt"
)

var (
	port               = flag.String("port", "8082", "Port to run the benchmark server on")
	localAIURL         = flag.String("localai-url", "http://localhost:8080", "LocalAI server URL")
	apiKey             = flag.String("api-key", "", "LocalAI API key")
	enableUI           = flag.Bool("ui", true, "Enable web UI")
	uiPort             = flag.String("ui-port", "8081", "Port for web UI")
	agentSDKFlightAddr = flag.String("agent-sdk-flight", os.Getenv("AGENTSDK_FLIGHT_ADDR"), "Agent SDK Flight address for catalog enrichment")
)

func main() {
	flag.Parse()

	// Create LocalAI client
	client := localai.NewClient(*localAIURL, *apiKey)

	// Optionally fetch Agent SDK catalog
	var agentCatalog *flightcatalog.Catalog
	if addr := strings.TrimSpace(*agentSDKFlightAddr); addr != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		cat, err := flightcatalog.Fetch(ctx, addr)
		cancel()
		if err != nil {
			log.Printf("Warning: failed to fetch Agent SDK catalog from %s: %v", addr, err)
		} else {
			agentCatalog = &cat
		}
	}

	// Create enhanced inference engine
	var inferenceEngine *localai.EnhancedInferenceEngine
	if agentCatalog != nil {
		inferenceEngine = localai.NewEnhancedInferenceEngine(client, localai.WithAgentCatalog(agentCatalog))
	} else {
		inferenceEngine = localai.NewEnhancedInferenceEngine(client)
	}

	// Create HTTP server
	server := &http.Server{
		Addr:    ":" + *port,
		Handler: createHandler(inferenceEngine, strings.TrimSpace(*agentSDKFlightAddr)),
	}

	// Start server in goroutine
	go func() {
		log.Printf("Starting benchmark server on port %s", *port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Start web UI if enabled
	if *enableUI {
		go startWebUI(*uiPort)
	}

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited")
}

func createHandler(engine *localai.EnhancedInferenceEngine, flightAddr string) http.Handler {
	mux := http.NewServeMux()

	// API endpoints
	mux.HandleFunc("/api/v1/infer", handleInfer(engine))
	mux.HandleFunc("/api/v1/benchmark", handleBenchmark(engine))
	mux.HandleFunc("/api/v1/models", handleModels(engine))
	mux.HandleFunc("/api/v1/capabilities", handleCapabilities(engine))
	mux.HandleFunc("/api/v1/health", handleHealth())
	mux.HandleFunc("/api/v1/agent-catalog", handleAgentCatalog(flightAddr))

	// WebSocket endpoint for real-time updates
	mux.HandleFunc("/ws", handleWebSocket())

	// Static file serving for web UI
	mux.Handle("/", http.FileServer(http.Dir("./web/")))

	return mux
}

func handleInfer(engine *localai.EnhancedInferenceEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req localai.InferenceRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if engine.AgentCatalog != nil {
			if req.Metadata == nil {
				req.Metadata = map[string]interface{}{}
			}
			req.Metadata["agent_catalog"] = engine.AgentCatalog.Suites
			req.Metadata["agent_tools"] = engine.AgentCatalog.Tools
			view := engine.CatalogEnrichment()
			if view.Summary != "" {
				req.Metadata["agent_catalog_summary"] = view.Summary
			}
			if view.Prompt != "" {
				req.Metadata["agent_catalog_context"] = view.Prompt
			}
			if view.Stats.SuiteCount > 0 || view.Stats.UniqueToolCount > 0 {
				req.Metadata["agent_catalog_stats"] = view.Stats
			}
			if len(view.Implementations) > 0 {
				req.Metadata["agent_catalog_matrix"] = view.Implementations
			}
			if len(view.UniqueTools) > 0 {
				req.Metadata["agent_catalog_unique_tools"] = view.UniqueTools
			}
			if len(view.StandaloneTools) > 0 {
				req.Metadata["agent_catalog_tool_details"] = view.StandaloneTools
			}
		}

		ctx := r.Context()
		resp, err := engine.Infer(ctx, &req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func handleBenchmark(engine *localai.EnhancedInferenceEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			InferenceRequest localai.InferenceRequest `json:"inference_request"`
			CorrectAnswer    string                   `json:"correct_answer"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		ctx := r.Context()
		result, err := engine.RunBenchmark(ctx, &req.InferenceRequest, req.CorrectAnswer)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

func handleModels(engine *localai.EnhancedInferenceEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		models := engine.ListAvailableModels()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models)
	}
}

func handleCapabilities(engine *localai.EnhancedInferenceEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		modelName := r.URL.Query().Get("model")
		if modelName == "" {
			http.Error(w, "Model name required", http.StatusBadRequest)
			return
		}

		capabilities, err := engine.GetModelCapabilities(modelName)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(capabilities)
	}
}

func handleHealth() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status": "healthy",
			"time":   time.Now().Format(time.RFC3339),
		})
	}
}

func handleAgentCatalog(flightAddr string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if strings.TrimSpace(flightAddr) == "" {
			http.Error(w, "agent catalog source not configured", http.StatusServiceUnavailable)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		catalog, err := flightcatalog.Fetch(ctx, flightAddr)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to fetch agent catalog: %v", err), http.StatusBadGateway)
			return
		}
		view := catalogprompt.Enrich(catalogprompt.Catalog{
			Suites: catalog.Suites,
			Tools:  catalog.Tools,
		})
		w.Header().Set("Content-Type", "application/json")
		payload := map[string]any{
			"Suites":                     catalog.Suites,
			"Tools":                      catalog.Tools,
			"agent_catalog_summary":      view.Summary,
			"agent_catalog_stats":        view.Stats,
			"agent_catalog_matrix":       view.Implementations,
			"agent_catalog_unique_tools": view.UniqueTools,
			"agent_catalog_tool_details": view.StandaloneTools,
			"agent_catalog_context":      view.Prompt,
		}
		json.NewEncoder(w).Encode(payload)
	}
}

func handleWebSocket() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// WebSocket implementation would go here
		// For now, return a simple response
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "WebSocket endpoint"})
	}
}

func startWebUI(port string) {
	log.Printf("Starting web UI on port %s", port)

	// Simple HTTP server for web UI
	uiServer := &http.Server{
		Addr:    ":" + port,
		Handler: http.FileServer(http.Dir("./web/")),
	}

	if err := uiServer.ListenAndServe(); err != nil {
		log.Printf("Web UI server error: %v", err)
	}
}
