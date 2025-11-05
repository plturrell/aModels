package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/plturrell/aModels/services/catalog/api"
	"github.com/plturrell/aModels/services/catalog/cache"
	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/migrations"
	"github.com/plturrell/aModels/services/catalog/observability"
	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/security"
	"github.com/plturrell/aModels/services/catalog/triplestore"
	"github.com/plturrell/aModels/services/catalog/workflows"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	// Initialize structured logger
	structLogger := observability.DefaultLogger()
	structLogger.Info("Starting catalog service", map[string]interface{}{
		"version": "1.0.0",
	})

	// Initialize Prometheus metrics
	observability.RegisterMetrics()
	structLogger.Info("Prometheus metrics initialized", nil)

	// Load configuration
	neo4jURI := os.Getenv("NEO4J_URI")
	if neo4jURI == "" {
		neo4jURI = "bolt://localhost:7687"
	}
	neo4jUsername := os.Getenv("NEO4J_USERNAME")
	if neo4jUsername == "" {
		neo4jUsername = "neo4j"
	}
	neo4jPassword := os.Getenv("NEO4J_PASSWORD")
	if neo4jPassword == "" {
		neo4jPassword = "password"
	}

	baseURI := os.Getenv("CATALOG_BASE_URI")
	if baseURI == "" {
		baseURI = "http://amodels.org/catalog"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8084"
	}

	extractServiceURL := os.Getenv("EXTRACT_SERVICE_URL")
	if extractServiceURL == "" {
		extractServiceURL = "http://localhost:9002"
	}
	
	graphServiceURL := os.Getenv("GRAPH_SERVICE_URL")
	if graphServiceURL == "" {
		graphServiceURL = "http://localhost:8081"
	}
	
	localaiURL := os.Getenv("LOCALAI_URL")
	if localaiURL == "" {
		localaiURL = "http://localhost:8081"
	}
	
	deepResearchURL := os.Getenv("DEEP_RESEARCH_URL")
	if deepResearchURL == "" {
		deepResearchURL = "http://localhost:8085"
	}

	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis://localhost:6379/0"
	}

	// Initialize Redis cache
	var cacheClient *cache.Cache
	if redisURL != "" {
		var err error
		cacheClient, err = cache.NewCache(redisURL, structLogger)
		if err != nil {
			structLogger.Warn("Failed to initialize Redis cache, continuing without cache", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			defer cacheClient.Close()
			structLogger.Info("Redis cache initialized", nil)
		}
	}

	// Initialize connection pool (if using performance package)
	// For now, we'll use the existing triplestore client
	
	// Initialize ISO 11179 registry
	registry := iso11179.NewMetadataRegistry("catalog", "aModels Catalog", baseURI)
	structLogger.Info("ISO 11179 metadata registry initialized", nil)

	// Run migrations if enabled
	runMigrations := os.Getenv("RUN_MIGRATIONS") == "true"
	if runMigrations {
		// Create legacy logger for migration runner (takes *log.Logger)
		legacyLogger := log.New(os.Stdout, "[catalog:migrations] ", log.LstdFlags|log.Lmsgprefix)
		migrationRunner := migrations.NewMigrationRunner(neo4jURI, neo4jUsername, neo4jPassword, legacyLogger)
		if err := migrationRunner.RunMigrations(context.Background()); err != nil {
			structLogger.Warn("Failed to run migrations, continuing", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			structLogger.Info("Migrations completed successfully", nil)
		}
	}

	// Initialize triplestore client
	legacyLogger := log.New(os.Stdout, "[catalog] ", log.LstdFlags|log.Lmsgprefix)
	triplestoreClient, err := triplestore.NewTriplestoreClient(neo4jURI, neo4jUsername, neo4jPassword, legacyLogger)
	if err != nil {
		structLogger.Error("Failed to create triplestore client", err, nil)
		os.Exit(1)
	}
	defer triplestoreClient.Close()
	structLogger.Info("Triplestore client initialized", nil)

	// Initialize SPARQL client and endpoint
	sparqlClient := triplestore.NewSPARQLClient(triplestoreClient, legacyLogger)
	sparqlEndpoint := triplestore.NewSPARQLEndpoint(sparqlClient, legacyLogger)
	structLogger.Info("SPARQL endpoint initialized", nil)

	// Initialize quality monitor (connects to Extract service)
	qualityMonitor := quality.NewQualityMonitor(extractServiceURL, legacyLogger)
	structLogger.Info("Quality monitor initialized (connected to Extract service)", nil)

	// Initialize unified workflow integration
	unifiedWorkflow := workflows.NewUnifiedWorkflowIntegration(
		graphServiceURL,
		graphServiceURL, // Orchestration via graph service
		"http://localhost:9001", // AgentFlow
		localaiURL,
		deepResearchURL,
		registry,
		qualityMonitor,
		legacyLogger,
	)
	structLogger.Info("Unified workflow integration initialized", nil)

	// Initialize API handlers
	catalogHandlers := api.NewCatalogHandlers(registry, legacyLogger)
	sparqlHandler := api.NewSPARQLHandler(sparqlEndpoint, legacyLogger)
	dataProductHandler := api.NewDataProductHandler(unifiedWorkflow, legacyLogger)
	
	// Initialize auth middleware
	authMiddleware := security.NewAuthMiddleware(legacyLogger)
	// Register default token for testing (in production, load from config)
	authMiddleware.RegisterToken("test-token", "test-user")

	// Initialize health checkers
	healthCheckers := []api.HealthChecker{
		api.NewBasicHealthChecker("neo4j", func(ctx context.Context) api.HealthStatus {
			// Check Neo4j connection by verifying driver is accessible
			// We'll use a simple test query through the triplestore client
			testTriple := triplestore.Triple{
				Subject:   "http://test/health",
				Predicate: "http://test/check",
				Object:    "health",
			}
			// Try to store a test triple (will be cleaned up)
			err := triplestoreClient.StoreTriple(ctx, testTriple)
			if err != nil {
				return api.HealthStatus{
					Status:    "down",
					Message:   err.Error(),
					Timestamp: time.Now(),
				}
			}
			return api.HealthStatus{
				Status:    "ok",
				Timestamp: time.Now(),
			}
		}),
	}
	
	if cacheClient != nil {
		healthCheckers = append(healthCheckers, api.NewBasicHealthChecker("redis", func(ctx context.Context) api.HealthStatus {
			// Check Redis connection
			exists, err := cacheClient.Exists(ctx, "healthcheck")
			if err != nil {
				return api.HealthStatus{
					Status:    "down",
					Message:   err.Error(),
					Timestamp: time.Now(),
				}
			}
			return api.HealthStatus{
				Status:    "ok",
				Details:   map[string]any{"exists_check": exists},
				Timestamp: time.Now(),
			}
		}))
	}
	
	healthHandler := api.NewHealthHandler(healthCheckers, structLogger)

	// Setup HTTP routes
	mux := http.NewServeMux()

	// Enhanced health checks
	mux.HandleFunc("/healthz", healthHandler.HandleHealthz)
	mux.HandleFunc("/ready", healthHandler.HandleReadiness)
	mux.HandleFunc("/live", healthHandler.HandleLiveness)
	
	// Prometheus metrics endpoint
	mux.Handle("/metrics", promhttp.Handler())

	// Catalog API endpoints
	mux.HandleFunc("/catalog/data-elements", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			catalogHandlers.HandleListDataElements(w, r)
		case http.MethodPost:
			catalogHandlers.HandleCreateDataElement(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/catalog/data-elements/", catalogHandlers.HandleGetDataElement)
	mux.HandleFunc("/catalog/ontology", catalogHandlers.HandleGetOntology)
	mux.HandleFunc("/catalog/semantic-search", catalogHandlers.HandleSemanticSearch)

	// SPARQL endpoint
	mux.HandleFunc("/catalog/sparql", sparqlHandler.HandleSPARQL)

	// Complete data product endpoints (thin slice approach)
	mux.HandleFunc("/catalog/data-products/build", dataProductHandler.HandleBuildDataProduct)
	mux.HandleFunc("/catalog/data-products/", dataProductHandler.HandleGetDataProduct)

	// Apply auth middleware to protected endpoints (optional - can be enabled via env var)
	useAuth := os.Getenv("ENABLE_AUTH") == "true"
	if useAuth {
		protectedMux := http.NewServeMux()
		protectedMux.HandleFunc("/catalog/data-elements", catalogHandlers.HandleCreateDataElement)
		protectedMux.HandleFunc("/catalog/data-products/build", dataProductHandler.HandleBuildDataProduct)
		mux.Handle("/catalog/", authMiddleware.Middleware(protectedMux))
		structLogger.Info("Authentication enabled", nil)
	}

	// Apply metrics middleware to all routes
	handler := api.MetricsMiddleware(mux)

	structLogger.Info("Catalog service starting", map[string]interface{}{
		"port":    port,
		"metrics": "/metrics",
		"health":  "/healthz",
	})
	structLogger.Info("Complete data product endpoints available", map[string]interface{}{
		"endpoint": "/catalog/data-products/build",
	})

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	if err := server.ListenAndServe(); err != nil {
		structLogger.Error("Server failed to start", err, nil)
		os.Exit(1)
	}
}

