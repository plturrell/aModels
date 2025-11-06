package main

import (
	"context"
	"database/sql"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	_ "github.com/lib/pq"
	"github.com/plturrell/aModels/services/catalog/ai"
	"github.com/plturrell/aModels/services/catalog/analytics"
	"github.com/plturrell/aModels/services/catalog/api"
	"github.com/plturrell/aModels/services/catalog/autonomous"
	"github.com/plturrell/aModels/services/catalog/breakdetection"
	"github.com/plturrell/aModels/services/catalog/cache"
	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/vectorstore"
	"github.com/plturrell/aModels/services/catalog/migrations"
	"github.com/plturrell/aModels/services/catalog/multimodal"
	"github.com/plturrell/aModels/services/catalog/observability"
	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/research"
	"github.com/plturrell/aModels/services/catalog/security"
	"github.com/plturrell/aModels/services/catalog/streaming"
	"github.com/plturrell/aModels/services/catalog/triplestore"
	"github.com/plturrell/aModels/services/catalog/workflows"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

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
	catalogDBURL := os.Getenv("CATALOG_DATABASE_URL")

	legacyLogger := log.New(os.Stdout, "[catalog] ", log.LstdFlags|log.Lmsgprefix)
	migrationLogger := log.New(os.Stdout, "[catalog:migrations] ", log.LstdFlags|log.Lmsgprefix)
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
		migrationRunner := migrations.NewMigrationRunner(neo4jURI, neo4jUsername, neo4jPassword, migrationLogger)
		if err := migrationRunner.RunMigrations(ctx); err != nil {
			structLogger.Warn("Failed to run Neo4j migrations, continuing", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			structLogger.Info("Neo4j migrations completed successfully", nil)
		}

		sqlDriver := os.Getenv("SQL_MIGRATIONS_DRIVER")
		sqlDSN := os.Getenv("SQL_MIGRATIONS_DSN")
		if sqlDSN == "" && catalogDBURL != "" {
			sqlDSN = catalogDBURL
		}
		if sqlDriver == "" && sqlDSN != "" {
			sqlDriver = "postgres"
		}
		if sqlDriver != "" && sqlDSN != "" {
			if err := migrations.RunGooseMigrations(sqlDriver, sqlDSN, "", migrationLogger); err != nil {
				structLogger.Warn("Failed to run SQL migrations, continuing", map[string]interface{}{
					"error": err.Error(),
				})
			} else {
				structLogger.Info("SQL migrations completed successfully", map[string]interface{}{
					"driver": sqlDriver,
				})
			}
		}
	}

	// Initialize triplestore client
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

	reportStore, err := research.NewReportStore(catalogDBURL, legacyLogger)
	if err != nil {
		structLogger.Warn("Failed to initialize research report store", map[string]interface{}{
			"error": err.Error(),
		})
	} else if reportStore != nil {
		defer reportStore.Close(ctx)
		structLogger.Info("Research report store initialized", nil)
		reportStore.StartRetentionJob(ctx, 24*time.Hour)
	}

	// Initialize Deep Research client for autonomous intelligence
	deepResearchClient := research.NewDeepResearchClient(deepResearchURL, legacyLogger)

	// Initialize version manager if database is available
	var versionManager *workflows.VersionManager
	if catalogDBURL != "" {
		if db, err := sql.Open("postgres", catalogDBURL); err == nil {
			if err := db.Ping(); err == nil {
				versionManager = workflows.NewVersionManager(db, legacyLogger)
				structLogger.Info("Version manager initialized", nil)
			} else {
				db.Close()
				structLogger.Warn("Failed to ping database for version manager, continuing without versioning", map[string]interface{}{
					"error": err.Error(),
				})
			}
		} else {
			structLogger.Warn("Failed to open database for version manager, continuing without versioning", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}

	// Initialize unified workflow integration
	unifiedWorkflow := workflows.NewUnifiedWorkflowIntegration(
		graphServiceURL,
		graphServiceURL,         // Orchestration via graph service
		"http://localhost:9001", // AgentFlow
		localaiURL,
		deepResearchURL,
		registry,
		qualityMonitor,
		reportStore,
		versionManager,
		legacyLogger,
	)
	structLogger.Info("Unified workflow integration initialized", nil)

	// Initialize AI capabilities
	metadataDiscoverer := ai.NewMetadataDiscoverer(deepResearchURL, extractServiceURL, legacyLogger)
	qualityPredictor := ai.NewQualityPredictor(extractServiceURL, legacyLogger)
	recommender := ai.NewRecommender(registry, legacyLogger)

	// Initialize event streaming (if Redis available)
	var eventStream *streaming.EventStream
	if redisURL != "" && cacheClient != nil {
		var err error
		eventStream, err = streaming.NewEventStream(redisURL, legacyLogger)
		if err != nil {
			structLogger.Warn("Failed to initialize event stream, continuing without streaming", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			defer eventStream.Close()
			structLogger.Info("Event streaming initialized", nil)
		}
	}

	// Initialize multi-modal extractor
	deepSeekOCRURL := os.Getenv("DEEPSEEK_OCR_URL")
	if deepSeekOCRURL == "" {
		deepSeekOCRURL = "http://localhost:8086"
	}
	multiModalExtractor := multimodal.NewMultiModalExtractor(deepSeekOCRURL, legacyLogger)
	structLogger.Info("Multi-modal extractor initialized", nil)

	// Initialize analytics dashboard
	analyticsDashboard := analytics.NewAnalyticsDashboard(registry, recommender, legacyLogger)
	structLogger.Info("Analytics dashboard initialized", nil)

	// Initialize break detection service if database is available
	var breakDetectionService *breakdetection.BreakDetectionService
	var baselineManager *breakdetection.BaselineManager
	var breakDetectionHandler *api.BreakDetectionHandler
	if catalogDBURL != "" {
		if db, err := sql.Open("postgres", catalogDBURL); err == nil {
			if err := db.Ping(); err == nil {
				// Initialize baseline manager
				baselineManager = breakdetection.NewBaselineManager(db, legacyLogger)
				
				// Initialize SAP Fioneer URL
				sapFioneerURL := os.Getenv("SAP_FIONEER_URL")
				if sapFioneerURL == "" {
					sapFioneerURL = "http://localhost:8080"
				}
				
				// Initialize finance detector
				financeDetector := breakdetection.NewFinanceDetector(sapFioneerURL, legacyLogger)
				
				// Initialize capital detector
				bcrsURL := os.Getenv("BCRS_URL")
				if bcrsURL == "" {
					bcrsURL = "http://localhost:8080"
				}
				capitalDetector := breakdetection.NewCapitalDetector(bcrsURL, legacyLogger)
				
				// Initialize liquidity detector
				rcoURL := os.Getenv("RCO_URL")
				if rcoURL == "" {
					rcoURL = "http://localhost:8080"
				}
				liquidityDetector := breakdetection.NewLiquidityDetector(rcoURL, legacyLogger)
				
				// Initialize regulatory detector
				axiomSLURL := os.Getenv("AXIOMSL_URL")
				if axiomSLURL == "" {
					axiomSLURL = "http://localhost:8080"
				}
				regulatoryDetector := breakdetection.NewRegulatoryDetector(axiomSLURL, legacyLogger)
				
				// Initialize Deep Research integration services
				var analysisService *breakdetection.BreakAnalysisService
				var enrichmentService *breakdetection.EnrichmentService
				var ruleGenerator *breakdetection.RuleGeneratorService
				
				if deepResearchClient != nil {
					analysisService = breakdetection.NewBreakAnalysisService(deepResearchClient, legacyLogger)
					enrichmentService = breakdetection.NewEnrichmentService(deepResearchClient, legacyLogger)
					ruleGenerator = breakdetection.NewRuleGeneratorService(deepResearchClient, db, legacyLogger)
					structLogger.Info("Deep Research integration for break detection initialized", nil)
				}
				
				// Initialize Search service
				searchService := breakdetection.NewBreakSearchService(extractServiceURL, legacyLogger)
				
				// Initialize LocalAI service
				var aiAnalysisService *breakdetection.AIAnalysisService
				if localaiURL != "" {
					aiAnalysisService = breakdetection.NewAIAnalysisService(localaiURL, legacyLogger)
					structLogger.Info("LocalAI integration for break detection initialized", nil)
				}
				
				// Initialize break detection service
				breakDetectionService = breakdetection.NewBreakDetectionService(
					db,
					baselineManager,
					financeDetector,
					capitalDetector,
					liquidityDetector,
					regulatoryDetector,
					analysisService,
					enrichmentService,
					ruleGenerator,
					searchService,
					aiAnalysisService,
					legacyLogger,
				)
				
				// Initialize break detection handler
				breakDetectionHandler = api.NewBreakDetectionHandler(
					breakDetectionService,
					baselineManager,
					legacyLogger,
				)
				
				structLogger.Info("Break detection service initialized", nil)
			} else {
				db.Close()
				structLogger.Warn("Failed to ping database for break detection service, continuing without break detection", map[string]interface{}{
					"error": err.Error(),
				})
			}
		} else {
			structLogger.Warn("Failed to open database for break detection service, continuing without break detection", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}

	// Initialize discoverability system if database is available
	var discoverSystem *discoverability.DiscoverabilitySystem
	if catalogDB != nil {
		discoverSystem = discoverability.NewDiscoverabilitySystem(catalogDB, legacyLogger)
		structLogger.Info("Discoverability system initialized", nil)
	}

	// Initialize API handlers
	catalogHandlers := api.NewCatalogHandlers(registry, legacyLogger)
	sparqlHandler := api.NewSPARQLHandler(sparqlEndpoint, legacyLogger)
	dataProductHandler := api.NewDataProductHandler(unifiedWorkflow, registry, versionManager, legacyLogger)
	aiHandlers := api.NewAIHandlers(metadataDiscoverer, qualityPredictor, recommender, legacyLogger)
	
	// Initialize discoverability handler if system is available
	var discoverabilityHandler *api.DiscoverabilityHandler
	if discoverSystem != nil {
		discoverabilityHandler = api.NewDiscoverabilityHandler(discoverSystem, legacyLogger)
		structLogger.Info("Discoverability handler initialized", nil)
	}

	// Initialize advanced handlers (Phase 3)
	var advancedHandlers *api.AdvancedHandlers
	var wsHandler *api.WebSocketHandler
	if eventStream != nil {
		wsHandler = api.NewWebSocketHandler(eventStream, legacyLogger)
		advancedHandlers = api.NewAdvancedHandlers(eventStream, multiModalExtractor, analyticsDashboard, legacyLogger)
		structLogger.Info("Advanced handlers initialized", nil)
	}

	// Initialize auth middleware
	authMiddleware := security.NewAuthMiddleware(legacyLogger)
	// Register default token for testing (in production, load from config)
	authMiddleware.RegisterToken("test-token", "test-user")

	// Initialize production readiness middleware
	errorHandler := api.NewErrorHandler(legacyLogger)
	rateLimiter := api.DefaultRateLimiter()
	monitoringMiddleware := api.NewMonitoringMiddleware(legacyLogger)
	structLogger.Info("Production readiness middleware initialized", nil)

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

	mux.HandleFunc("/catalog/ontology", catalogHandlers.HandleGetOntology)
	mux.HandleFunc("/catalog/semantic-search", catalogHandlers.HandleSemanticSearch)

	// SPARQL endpoint
	mux.HandleFunc("/catalog/sparql", sparqlHandler.HandleSPARQL)

	// Complete data product endpoints (thin slice approach)
	mux.HandleFunc("/catalog/data-products/build", dataProductHandler.HandleBuildDataProduct)
	
	// Data products endpoint (handles both get and sample)
	mux.HandleFunc("/catalog/data-products/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/sample") {
			dataProductHandler.HandleGetSampleData(w, r)
		} else {
			dataProductHandler.HandleGetDataProduct(w, r)
		}
	})
	
	// Data elements endpoint (handles both get and sample)
	mux.HandleFunc("/catalog/data-elements/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/sample") {
			dataProductHandler.HandleGetSampleData(w, r)
		} else {
			catalogHandlers.HandleGetDataElement(w, r)
		}
	})

	// AI capabilities endpoints
	mux.HandleFunc("/catalog/ai/discover", aiHandlers.HandleDiscoverMetadata)
	mux.HandleFunc("/catalog/ai/predict-quality", aiHandlers.HandlePredictQuality)
	mux.HandleFunc("/catalog/ai/recommendations", aiHandlers.HandleGetRecommendations)
	mux.HandleFunc("/catalog/ai/usage", aiHandlers.HandleRecordUsage)

		// Discoverability endpoints
		if discoverabilityHandler != nil {
			mux.HandleFunc("/api/discover/search", discoverabilityHandler.HandleSearch)
			mux.HandleFunc("/api/discover/marketplace", discoverabilityHandler.HandleMarketplace)
			mux.HandleFunc("/api/discover/tags", discoverabilityHandler.HandleCreateTag)
			mux.HandleFunc("/api/discover/access-request", discoverabilityHandler.HandleRequestAccess)
		}

		// Autonomous Intelligence Layer endpoints
		deepAgentsURL := os.Getenv("DEEPAGENTS_URL")
		if deepAgentsURL == "" {
			deepAgentsURL = "http://deepagents-service:9004"
		}
		unifiedWorkflowURL := os.Getenv("GRAPH_SERVICE_URL")
		if unifiedWorkflowURL == "" {
			unifiedWorkflowURL = "http://graph-service:8081"
		}
		autonomousHandler := autonomous.NewAutonomousHandler(
			deepResearchClient,
			deepAgentsURL,
			unifiedWorkflowURL,
			legacyLogger,
		)
		mux.HandleFunc("/api/autonomous/execute", autonomousHandler.HandleExecuteTask)
		mux.HandleFunc("/api/autonomous/metrics", autonomousHandler.HandleGetMetrics)
		mux.HandleFunc("/api/autonomous/agents", autonomousHandler.HandleGetAgents)
		mux.HandleFunc("/api/autonomous/knowledge", autonomousHandler.HandleGetKnowledgeBase)
		structLogger.Info("Autonomous Intelligence Layer endpoints registered", nil)

	// Break detection endpoints
	if breakDetectionHandler != nil {
		mux.HandleFunc("/catalog/break-detection/detect", breakDetectionHandler.HandleDetectBreaks)
		mux.HandleFunc("/catalog/break-detection/baselines", func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				breakDetectionHandler.HandleCreateBaseline(w, r)
			} else if r.Method == http.MethodGet {
				breakDetectionHandler.HandleListBaselines(w, r)
			} else {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			}
		})
		mux.HandleFunc("/catalog/break-detection/baselines/", breakDetectionHandler.HandleGetBaseline)
		mux.HandleFunc("/catalog/break-detection/breaks", breakDetectionHandler.HandleListBreaks)
		mux.HandleFunc("/catalog/break-detection/breaks/", breakDetectionHandler.HandleGetBreak)
		structLogger.Info("Break detection endpoints registered", nil)
	}

	// HANA Cloud Vector Store endpoints (public information)
	hanaConnectionString := os.Getenv("HANA_CLOUD_CONNECTION_STRING")
	if hanaConnectionString != "" {
		hanaConfig := &vectorstore.HANAConfig{
			ConnectionString: hanaConnectionString,
			Schema:           getEnvOrDefault("HANA_CLOUD_SCHEMA", "PUBLIC"),
			TableName:        getEnvOrDefault("HANA_CLOUD_TABLE_NAME", "PUBLIC_VECTORS"),
			VectorDimension: getEnvIntOrDefault("HANA_CLOUD_VECTOR_DIMENSION", 1536),
			EnableIndexing:   os.Getenv("HANA_CLOUD_ENABLE_INDEXING") != "false",
		}

		hanaStore, err := vectorstore.NewHANACloudVectorStore(hanaConnectionString, hanaConfig, legacyLogger)
		if err != nil {
			structLogger.Warn("HANA Cloud vector store not available", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			embeddingService := vectorstore.NewEmbeddingService(
				os.Getenv("LOCALAI_URL"),
				legacyLogger,
			)

			vectorStoreHandler := vectorstore.NewHANAVectorStoreHandler(
				hanaStore,
				embeddingService,
				legacyLogger,
			)

			// Register vector store endpoints
			mux.HandleFunc("/vectorstore/store", vectorStoreHandler.HandleStoreInformation)
			mux.HandleFunc("/vectorstore/search", vectorStoreHandler.HandleSearchInformation)
			mux.HandleFunc("/vectorstore", func(w http.ResponseWriter, r *http.Request) {
				// Handle both list (GET with query params) and get by ID (GET /vectorstore/{id})
				if r.Method == http.MethodGet {
					// Check if path has ID (not just /vectorstore)
					path := r.URL.Path
					if path == "/vectorstore" || path == "/vectorstore/" {
						// List public information
						vectorStoreHandler.HandleListPublicInformation(w, r)
					} else {
						// Get by ID
						vectorStoreHandler.HandleGetInformation(w, r)
					}
				} else {
					http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				}
			})
			mux.HandleFunc("/vectorstore/", vectorStoreHandler.HandleGetInformation)

			structLogger.Info("HANA Cloud vector store endpoints registered", map[string]interface{}{
				"endpoints": []string{
					"POST /vectorstore/store",
					"POST /vectorstore/search",
					"GET /vectorstore (list)",
					"GET /vectorstore/{id}",
				},
			})
		}
	}

	// Advanced features endpoints (Phase 3)
	if advancedHandlers != nil {
		mux.HandleFunc("/catalog/multimodal/extract", advancedHandlers.HandleExtractMultimodal)
		mux.HandleFunc("/catalog/analytics/dashboard", advancedHandlers.HandleGetDashboardStats)
		mux.HandleFunc("/catalog/analytics/elements/", advancedHandlers.HandleGetElementAnalytics)
		mux.HandleFunc("/catalog/analytics/top", advancedHandlers.HandleGetTopElements)
		structLogger.Info("Advanced features endpoints registered", nil)
	}

	// WebSocket endpoints (Phase 3)
	if wsHandler != nil {
		mux.HandleFunc("/catalog/ws", wsHandler.HandleWebSocket)
		mux.HandleFunc("/catalog/ws/subscribe", wsHandler.HandleWebSocketSubscribe)
		structLogger.Info("WebSocket endpoints registered", nil)
	}

	// Apply auth middleware to protected endpoints (optional - can be enabled via env var)
	useAuth := os.Getenv("ENABLE_AUTH") == "true"
	if useAuth {
		protectedMux := http.NewServeMux()
		protectedMux.HandleFunc("/catalog/data-elements", catalogHandlers.HandleCreateDataElement)
		protectedMux.HandleFunc("/catalog/data-products/build", dataProductHandler.HandleBuildDataProduct)
		mux.Handle("/catalog/", authMiddleware.Middleware(protectedMux))
		structLogger.Info("Authentication enabled", nil)
	}

	// Apply production readiness middleware (order matters: recovery -> rate limit -> monitoring -> metrics)
	handler := errorHandler.RecoveryMiddleware(
		api.RateLimitMiddleware(rateLimiter, 100.0/60.0, 10)( // 100 req/min per IP, burst of 10
			monitoringMiddleware.Middleware(
				api.MetricsMiddleware(mux),
			),
		),
	)

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

// Helper functions for environment variables
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
