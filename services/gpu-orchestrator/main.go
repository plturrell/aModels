package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/api"
	"github.com/plturrell/aModels/services/gpu-orchestrator/config"
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
	configPath := os.Getenv("CONFIG_PATH")
	if configPath == "" {
		configPath = "config.yaml"
	}
	
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		logger.Fatalf("Failed to load configuration: %v", err)
	}
	logger.Printf("Loaded configuration from %s", configPath)

	// Override with environment variables if set
	if port := os.Getenv("PORT"); port != "" {
		cfg.Server.Port = port
	}
	if url := os.Getenv("DEEPAGENTS_URL"); url != "" {
		cfg.Services.DeepAgentsURL = url
	}
	if url := os.Getenv("GRAPH_SERVICE_URL"); url != "" {
		cfg.Services.GraphServiceURL = url
	}

	// Initialize GPU monitor
	monitor, err2 := gpu_monitor.NewGPUMonitor(logger)
	if err2 != nil {
		logger.Printf("Warning: Failed to initialize GPU monitor: %v", err2)
		logger.Println("Continuing without GPU monitoring (may be running on non-GPU system)")
	}

	// Initialize GPU allocator
	allocator := gpu_allocator.NewGPUAllocator(monitor, logger)

	// Initialize workload analyzer
	workloadAnalyzer := workload_analyzer.NewWorkloadAnalyzer(cfg.Services.GraphServiceURL, logger)

	// Initialize scheduler
	gpuScheduler := scheduler.NewScheduler(allocator, workloadAnalyzer, logger)

	// Initialize orchestrator
	orchestrator := gpu_orchestrator.NewGPUOrchestrator(
		allocator,
		gpuScheduler,
		workloadAnalyzer,
		monitor,
		cfg.Services.DeepAgentsURL,
		cfg.Services.GraphServiceURL,
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

	// Apply middleware (authentication first, then metrics)
	handler := mux
	
	// Apply authentication middleware if enabled
	if cfg.Auth.Enabled {
		authConfig := &api.AuthConfig{
			Enabled:    cfg.Auth.Enabled,
			APIKeys:    cfg.Auth.APIKeys,
			HeaderName: cfg.Auth.HeaderName,
			Logger:     logger,
		}
		handler = api.AuthMiddleware(authConfig)(handler)
		logger.Printf("API key authentication enabled with %d keys", len(cfg.Auth.APIKeys))
	}
	
	// Apply metrics middleware
	handler = api.MetricsMiddleware(handler)

	logger.Printf("GPU orchestrator service starting on port %s", cfg.Server.Port)

	server := &http.Server{
		Addr:         ":" + cfg.Server.Port,
		Handler:      handler,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	// Start background monitoring and metrics collection
	if monitor != nil && cfg.Monitoring.MetricsEnabled {
		go func() {
			ticker := time.NewTicker(cfg.Monitoring.RefreshInterval)
			defer ticker.Stop()
			for range ticker.C {
				if err := monitor.Refresh(); err != nil {
					logger.Printf("Error refreshing GPU monitor: %v", err)
				}
				
				// Update GPU metrics
				updateGPUMetrics(monitor, orchestrator)
			}
		}()
	}

	if err := server.ListenAndServe(); err != nil {
		logger.Fatalf("Server failed to start: %v", err)
	}
}

// updateGPUMetrics updates Prometheus metrics with current GPU stats
func updateGPUMetrics(monitor *gpu_monitor.GPUMonitor, orchestrator *gpu_orchestrator.GPUOrchestrator) {
	gpus := monitor.ListGPUs()
	available := monitor.GetAvailableGPUs()
	
	// Update GPU count metrics
	api.GPUTotal.Set(float64(len(gpus)))
	api.GPUAvailable.Set(float64(len(available)))
	
	// Update per-GPU metrics
	for _, gpu := range gpus {
		gpuID := fmt.Sprintf("%d", gpu.ID)
		
		api.GPUUtilization.WithLabelValues(gpuID, gpu.Name).Set(gpu.Utilization)
		api.GPUMemoryUsed.WithLabelValues(gpuID, gpu.Name).Set(float64(gpu.MemoryUsed))
		api.GPUMemoryTotal.WithLabelValues(gpuID, gpu.Name).Set(float64(gpu.MemoryTotal))
		api.GPUTemperature.WithLabelValues(gpuID, gpu.Name).Set(gpu.Temperature)
		api.GPUPowerDraw.WithLabelValues(gpuID, gpu.Name).Set(gpu.PowerDraw)
	}
	
	// Update allocation metrics
	allocations := orchestrator.ListAllocations()
	api.GPUAllocationsActive.Set(float64(len(allocations)))
	
	// Update queue metrics
	queue := orchestrator.GetQueueStatus()
	api.GPUQueueDepth.Set(float64(len(queue)))
}

