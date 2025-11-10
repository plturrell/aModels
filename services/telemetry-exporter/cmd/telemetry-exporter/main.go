package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/internal/api"
	"github.com/plturrell/aModels/services/telemetry-exporter/internal/config"
	"github.com/plturrell/aModels/services/telemetry-exporter/internal/exporter"
	"github.com/plturrell/aModels/services/telemetry-exporter/internal/sources"
)

func main() {
	var (
		port = flag.String("port", "", "HTTP server port (overrides TELEMETRY_EXPORTER_PORT)")
	)
	flag.Parse()

	logger := log.New(os.Stdout, "[telemetry-exporter] ", log.LstdFlags)

	// Load configuration
	cfg := config.LoadConfig()
	if *port != "" {
		cfg.Port = *port
	}
	if cfg.Port == "" {
		cfg.Port = "8085"
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		logger.Fatalf("Configuration error: %v", err)
	}

	logger.Printf("Starting telemetry exporter service on port %s", cfg.Port)
	logger.Printf("Signavio API URL: %s", cfg.SignavioAPIURL)
	logger.Printf("Extract Service URL: %s", cfg.ExtractServiceURL)
	if cfg.AgentMetricsBaseURL != "" {
		logger.Printf("Agent Metrics Base URL: %s", cfg.AgentMetricsBaseURL)
	}

	// Create Signavio exporter
	signavioExporter := exporter.NewSignavioExporter(
		cfg.SignavioAPIURL,
		cfg.SignavioAPIKey,
		cfg.SignavioTenantID,
		cfg.SignavioDataset,
		true, // enabled
		cfg.SignavioTimeout,
		cfg.SignavioMaxRetries,
		logger,
	)

	// Validate Signavio connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	if err := signavioExporter.ValidateConnection(ctx); err != nil {
		logger.Printf("Warning: Signavio connection validation failed: %v", err)
	} else {
		logger.Printf("Signavio connection validated successfully")
	}
	cancel()

	// Create Extract service client
	extractClient := sources.NewExtractServiceClient(
		cfg.ExtractServiceURL,
		30*time.Second,
	)

	// Create agent telemetry client if configured
	var agentTelemetryClient *sources.AgentTelemetryClient
	if cfg.AgentMetricsBaseURL != "" {
		var err error
		agentTelemetryClient, err = sources.NewAgentTelemetryClient(
			cfg.AgentMetricsBaseURL,
			30*time.Second,
		)
		if err != nil {
			logger.Printf("Warning: Failed to create agent telemetry client: %v", err)
		} else {
			logger.Printf("Agent telemetry client initialized")
		}
	}

	// Create unified discovery
	discovery := sources.NewUnifiedDiscovery(
		extractClient,
		agentTelemetryClient,
		logger,
	)

	// Create API server
	apiServer := api.NewServer(
		signavioExporter,
		discovery,
		cfg.AgentName,
		logger,
	)

	// Register routes
	mux := http.NewServeMux()
	apiServer.RegisterRoutes(mux)

	// Create HTTP server
	server := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.Printf("Telemetry exporter service listening on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt signal for graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Println("Shutting down telemetry exporter service...")

	// Graceful shutdown with timeout
	ctx, cancel = context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Printf("Error during server shutdown: %v", err)
	} else {
		logger.Println("Server shutdown complete")
	}
}

