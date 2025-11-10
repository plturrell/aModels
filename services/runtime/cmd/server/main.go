package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/runtime/orchestrator"
	"github.com/plturrell/aModels/services/runtime/rest"
)

func main() {
	addr := getEnv("RUNTIME_ADDR", ":8098")
	catalogURL := getEnv("CATALOG_URL", "http://localhost:8084")

	logger := log.New(os.Stdout, "runtime: ", log.LstdFlags|log.Lshortfile)

	analyticsClient, err := analytics.NewClient(catalogURL)
	if err != nil {
		logger.Fatalf("failed to create analytics client: %v", err)
	}

	orch := orchestrator.New(analyticsClient, logger)
	handler := rest.NewHandler(orch)
	
	// Configure service URLs for unified analytics
	trainingURL := getEnv("TRAINING_SERVICE_URL", "http://localhost:8001")
	searchURL := getEnv("SEARCH_SERVICE_URL", "http://localhost:8000")
	handler.SetTrainingServiceURL(trainingURL)
	handler.SetSearchServiceURL(searchURL)

	// Create metrics collector for observability
	metrics := rest.NewMetricsCollector(logger)

	// Create WebSocket hub
	hub := rest.NewHub()
	go hub.Run()

	// Start periodic dashboard updates
	go func() {
		ticker := time.NewTicker(5 * time.Second) // Update every 5 seconds
		defer ticker.Stop()
		for range ticker.C {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			stats, templates, err := orch.FetchDashboardData(ctx)
			cancel()
			if err != nil {
				logger.Printf("dashboard refresh failed: %v", err)
				continue
			}
			hub.Broadcast(map[string]interface{}{
				"type":      "dashboard_update",
				"timestamp": time.Now().Unix(),
				"stats":     stats,
				"templates": templates,
			})
		}
	}()

	mux := http.NewServeMux()
	mux.Handle("/analytics/dashboard", handler)
	mux.HandleFunc("/analytics/ws", func(w http.ResponseWriter, r *http.Request) {
		handler.ServeWebSocket(w, r, hub)
	})
	
	// Unified Analytics API (v1)
	mux.HandleFunc("/api/v1/analytics", func(w http.ResponseWriter, r *http.Request) {
		handler.unifiedAnalytics.ServeUnifiedAnalytics(w, r)
	})
	mux.HandleFunc("/api/v1/analytics/system", func(w http.ResponseWriter, r *http.Request) {
		handler.unifiedAnalytics.ServeSystemWideAnalytics(w, r)
	})
	mux.HandleFunc("/api/v1/analytics/docs", func(w http.ResponseWriter, r *http.Request) {
		handler.ServeAPIDocumentation(w, r)
	})
	
	// Dashboard management endpoints
	mux.HandleFunc("/dashboard/create", handler.CreateDashboard)
	mux.HandleFunc("/dashboard/get", handler.GetDashboard)
	mux.HandleFunc("/dashboard/list", handler.ListDashboards)
	mux.HandleFunc("/dashboard/update", handler.UpdateDashboard)
	mux.HandleFunc("/dashboard/delete", handler.DeleteDashboard)
	mux.HandleFunc("/dashboard/share", handler.ShareDashboard)
	mux.HandleFunc("/dashboard/versions", handler.GetDashboardVersions)
	mux.HandleFunc("/dashboard/export", handler.ExportDashboard)
	mux.HandleFunc("/dashboard/import", handler.ImportDashboard)
	
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	// Apply observability middleware
	observableMux := rest.ObservabilityMiddleware(metrics, mux)
	observableMux = rest.LoggingMiddleware(logger, observableMux)
	
	// Add metrics endpoint
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(metrics.GetMetrics())
	})
	
	srv := &http.Server{
		Addr:              addr,
		Handler:           observableMux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	logger.Printf("runtime analytics server listening on %s (catalog=%s)", addr, catalogURL)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Fatalf("server error: %v", err)
	}
}

func getEnv(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}

func loggingMiddleware(logger *log.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		logger.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
