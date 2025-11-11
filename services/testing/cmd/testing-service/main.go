package main

import (
	"context"
	"database/sql"
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	_ "github.com/lib/pq"
	
	"github.com/plturrell/aModels/services/testing"
)

func main() {
	var (
		port           = flag.String("port", "8082", "HTTP server port")
		dbDSN          = flag.String("db", os.Getenv("TEST_DB_DSN"), "Database DSN")
		extractURL     = flag.String("extract-url", os.Getenv("EXTRACT_SERVICE_URL"), "Extract service URL")
	)
	flag.Parse()

	logger := log.New(os.Stdout, "[testing] ", log.LstdFlags)

	if *dbDSN == "" {
		logger.Fatalf("Database DSN required (set TEST_DB_DSN or use -db flag)")
	}

	if *extractURL == "" {
		*extractURL = "http://localhost:8081"
	}

	// Connect to database with connection pooling
	db, err := sql.Open("postgres", *dbDSN)
	if err != nil {
		logger.Fatalf("failed to connect to database: %v", err)
	}
	defer db.Close()
	
	// Configure connection pool
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Verify database connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		logger.Fatalf("failed to ping database: %v", err)
	}

	// Load configuration
	cfg := testing.LoadConfig()
	if cfg.DatabaseDSN == "" {
		cfg.DatabaseDSN = *dbDSN
	}
	if cfg.ExtractServiceURL == "" {
		cfg.ExtractServiceURL = *extractURL
	}
	if err := cfg.Validate(); err != nil {
		logger.Fatalf("Configuration error: %v", err)
	}

	// Create Extract client
	extractClient := testing.NewHTTPExtractClient(cfg.ExtractServiceURL)

	// Create LocalAI client
	localaiClient := testing.NewLocalAIClient(
		cfg.LocalAIURL,
		cfg.LocalAIModel,
		cfg.EnableLocalAI && cfg.LocalAIEnabled,
		cfg.LocalAITimeout,
		cfg.LocalAIRetryAttempts,
		logger,
	)

	// Create search client
	searchClient := testing.NewSearchClient(
		cfg.SearchServiceURL,
		cfg.SearchServiceTimeout,
		cfg.EnableSearch,
		logger,
	)

	// Create sample generator
	generator := testing.NewSampleGenerator(db, extractClient, logger)

	// Create Signavio client if enabled
	if cfg.SignavioEnabled {
		signavioClient := testing.NewSignavioClient(
			cfg.SignavioAPIURL,
			cfg.SignavioAPIKey,
			cfg.SignavioTenantID,
			cfg.SignavioEnabled,
			cfg.SignavioTimeout,
			cfg.SignavioMaxRetries,
			logger,
		)
		generator.SetSignavioClient(signavioClient, cfg.SignavioDataset)
		
		// Test connection if auto-export is enabled
		if cfg.SignavioAutoExport {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			if err := signavioClient.HealthCheck(ctx); err != nil {
				logger.Printf("Warning: Signavio health check failed: %v", err)
			} else {
				logger.Printf("Signavio client initialized and connected (auto-export enabled)")
			}
			cancel()
		} else {
			logger.Printf("Signavio client initialized (auto-export disabled)")
		}
	}

	// Create test service
	testService := testing.NewTestService(generator, searchClient, logger)

	// Register routes
	mux := http.NewServeMux()
	testService.RegisterRoutes(mux)

	// Health check
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	addr := ":" + *port
	logger.Printf("Testing service listening on %s", addr)
	logger.Printf("Extract service: %s", *extractURL)
	
	if err := http.ListenAndServe(addr, mux); err != nil {
		logger.Fatalf("server exited with error: %v", err)
	}
}

