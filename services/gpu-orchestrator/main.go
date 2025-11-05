package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/api"
	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_allocator"
	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_monitor"
	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_orchestrator"
	"github.com/plturrell/aModels/services/gpu-orchestrator/scheduler"
	"github.com/plturrell/aModels/services/gpu-orchestrator/workload_analyzer"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	// Initialize logger
	logger := log.New(os.Stdout, "[gpu-orchestrator] ", log.LstdFlags|log.Lmsgprefix)
	logger.Println("Starting GPU orchestrator service")

	// Load configuration
	port := os.Getenv("PORT")
	if port == "" {
		port = "8086"
	}

	deepAgentsURL := os.Getenv("DEEPAGENTS_URL")
	if deepAgentsURL == "" {
		deepAgentsURL = "http://localhost:9004"
	}

	graphServiceURL := os.Getenv("GRAPH_SERVICE_URL")
	if graphServiceURL == "" {
		graphServiceURL = "http://localhost:8081"
	}

	// Initialize GPU monitor
	monitor, err := gpu_monitor.NewGPUMonitor(logger)
	if err != nil {
		logger.Printf("Warning: Failed to initialize GPU monitor: %v", err)
		logger.Println("Continuing without GPU monitoring (may be running on non-GPU system)")
	}

	// Initialize GPU allocator
	allocator := gpu_allocator.NewGPUAllocator(monitor, logger)

	// Initialize workload analyzer
	workloadAnalyzer := workload_analyzer.NewWorkloadAnalyzer(graphServiceURL, logger)

	// Initialize scheduler
	gpuScheduler := scheduler.NewScheduler(allocator, workloadAnalyzer, logger)

	// Initialize orchestrator
	orchestrator := gpu_orchestrator.NewGPUOrchestrator(
		allocator,
		gpuScheduler,
		workloadAnalyzer,
		monitor,
		deepAgentsURL,
		graphServiceURL,
		logger,
	)

	// Initialize API handlers
	handlers := api.NewHandlers(orchestrator, logger)

	// Setup HTTP routes
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("/healthz", handlers.HandleHealthz)

	// GPU management endpoints
	mux.HandleFunc("/gpu/allocate", handlers.HandleAllocateGPU)
	mux.HandleFunc("/gpu/release", handlers.HandleReleaseGPU)
	mux.HandleFunc("/gpu/status", handlers.HandleGPUStatus)
	mux.HandleFunc("/gpu/list", handlers.HandleListGPUs)
	mux.HandleFunc("/gpu/workload", handlers.HandleWorkloadAnalysis)

	// Prometheus metrics
	mux.Handle("/metrics", promhttp.Handler())

	// Apply metrics middleware
	handler := api.MetricsMiddleware(mux)

	logger.Printf("GPU orchestrator service starting on port %s", port)

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start background monitoring
	if monitor != nil {
		go func() {
			ticker := time.NewTicker(5 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				if err := monitor.Refresh(); err != nil {
					logger.Printf("Error refreshing GPU monitor: %v", err)
				}
			}
		}()
	}

	if err := server.ListenAndServe(); err != nil {
		logger.Fatalf("Server failed to start: %v", err)
	}
}

