package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/api"
	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/exporter"
	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/file"
	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/signavio"
)

func main() {
	logger := log.New(os.Stdout, "[telemetry-exporter] ", log.LstdFlags)

	// Load configuration from environment
	cfg := loadConfig()

	// Initialize export manager
	exportMgr, err := exporter.NewExportManager(exporter.ExportManagerConfig{
		Mode:          exporter.ExportMode(cfg.ExportMode),
		FlushInterval: cfg.FlushInterval,
		FileEnabled:   cfg.FileEnabled,
		FileConfig: file.FileExporterConfig{
			BasePath:    cfg.FilePath,
			MaxFileSize: cfg.MaxFileSize,
			MaxFiles:    cfg.MaxFiles,
			Logger:      logger.Printf,
		},
		SignavioEnabled: cfg.SignavioEnabled,
		SignavioConfig: signavio.SignavioExporterConfig{
			BaseURL:    cfg.SignavioBaseURL,
			APIKey:     cfg.SignavioAPIKey,
			TenantID:   cfg.SignavioTenantID,
			Dataset:    cfg.SignavioDataset,
			BatchSize:  cfg.SignavioBatchSize,
			Timeout:    cfg.SignavioTimeout,
			MaxRetries: cfg.SignavioMaxRetries,
			Logger:     logger,
		},
		Logger: logger,
	})
	if err != nil {
		logger.Fatalf("Failed to create export manager: %v", err)
	}

	// Create export handler
	exportHandler := api.NewExportHandler(exportMgr, logger)

	// Setup HTTP server
	mux := http.NewServeMux()
	exportHandler.RegisterRoutes(mux)

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	// Start server
	go func() {
		logger.Printf("Telemetry exporter service starting on :8080")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	logger.Println("Shutting down...")

	// Shutdown server
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		logger.Printf("Server shutdown error: %v", err)
	}

	// Shutdown export manager
	if err := exportMgr.Shutdown(ctx); err != nil {
		logger.Printf("Export manager shutdown error: %v", err)
	}

	logger.Println("Shutdown complete")
}

// Config holds service configuration.
type Config struct {
	ExportMode         string
	FlushInterval      time.Duration
	FileEnabled        bool
	FilePath           string
	MaxFileSize        int64
	MaxFiles           int
	SignavioEnabled    bool
	SignavioBaseURL    string
	SignavioAPIKey     string
	SignavioTenantID   string
	SignavioDataset    string
	SignavioBatchSize  int
	SignavioTimeout    time.Duration
	SignavioMaxRetries int
}

func loadConfig() Config {
	cfg := Config{
		ExportMode:      getEnvOrDefault("OTEL_EXPORT_MODE", "both"),
		FlushInterval:   getDurationEnvOrDefault("OTEL_EXPORT_FLUSH_INTERVAL", 30*time.Second),
		FileEnabled:     getBoolEnvOrDefault("OTEL_EXPORT_FILE_ENABLED", true),
		FilePath:        getEnvOrDefault("OTEL_EXPORT_FILE_PATH", "/app/data/traces"),
		MaxFileSize:     getInt64EnvOrDefault("OTEL_EXPORT_FILE_MAX_SIZE", 100*1024*1024),
		MaxFiles:        getIntEnvOrDefault("OTEL_EXPORT_FILE_MAX_FILES", 10),
		SignavioEnabled: getBoolEnvOrDefault("OTEL_EXPORT_SIGNAVIO_ENABLED", false),
		SignavioBaseURL: getEnvOrDefault("SIGNAVIO_API_URL", ""),
		SignavioAPIKey:  getEnvOrDefault("SIGNAVIO_API_KEY", ""),
		SignavioTenantID: getEnvOrDefault("SIGNAVIO_TENANT_ID", ""),
		SignavioDataset:  getEnvOrDefault("SIGNAVIO_DATASET", ""),
		SignavioBatchSize: getIntEnvOrDefault("SIGNAVIO_BATCH_SIZE", 100),
		SignavioTimeout:   getDurationEnvOrDefault("SIGNAVIO_TIMEOUT", 30*time.Second),
		SignavioMaxRetries: getIntEnvOrDefault("SIGNAVIO_MAX_RETRIES", 3),
	}
	return cfg
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getBoolEnvOrDefault(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return value == "true"
	}
	return defaultValue
}

func getIntEnvOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if result, err := strconv.Atoi(value); err == nil {
			return result
		}
	}
	return defaultValue
}

func getInt64EnvOrDefault(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if result, err := strconv.ParseInt(value, 10, 64); err == nil {
			return result
		}
	}
	return defaultValue
}

func getDurationEnvOrDefault(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if d, err := time.ParseDuration(value); err == nil {
			return d
		}
	}
	return defaultValue
}

