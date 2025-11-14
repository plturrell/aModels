package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	_ "github.com/Chahine-tech/sql-parser-go/pkg/parser"
	_ "github.com/SAP/go-hdb/driver"
	"github.com/lib/pq"
	extractpb "github.com/plturrell/aModels/services/extract/gen/extractpb"

	telemetryclient "github.com/plturrell/aModels/services/extract/internal/agents/telemetry"
	"github.com/plturrell/aModels/services/extract/internal/config"
	handlers "github.com/plturrell/aModels/services/extract/internal/handlers"
	"github.com/plturrell/aModels/services/extract/internal/processing"

	"github.com/plturrell/aModels/services/extract/internal/middleware"
	"github.com/plturrell/aModels/services/extract/internal/observability"
	"github.com/plturrell/aModels/services/extract/pkg/ai"
	"github.com/plturrell/aModels/services/extract/pkg/catalog"
	"github.com/plturrell/aModels/services/extract/pkg/clients"
	"github.com/plturrell/aModels/services/extract/pkg/embeddings"
	"github.com/plturrell/aModels/services/extract/pkg/extraction"
	"github.com/plturrell/aModels/services/extract/pkg/git"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
	"github.com/plturrell/aModels/services/extract/pkg/integrations"
	"github.com/plturrell/aModels/services/extract/pkg/monitoring"
	"github.com/plturrell/aModels/services/extract/pkg/persistence"
	"github.com/plturrell/aModels/services/extract/pkg/pipeline"
	"github.com/plturrell/aModels/services/extract/pkg/schema"
	"github.com/plturrell/aModels/services/extract/pkg/storage"
	"github.com/plturrell/aModels/services/extract/pkg/terminology"
	"github.com/plturrell/aModels/services/extract/pkg/utils"
	"github.com/plturrell/aModels/services/extract/pkg/workflow"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

const (
	// Server configuration
	defaultPort       = "8081"
	defaultGRPCPort   = "9090"
	defaultFlightAddr = ":8815"

	// External service URLs
	defaultLangextractURL = "http://langextract-api:5000"

	// Extraction defaults
	defaultPromptDescription = "Extract the key entities (people, projects, dates, locations) from the document text."
	defaultModelID           = "gemini-2.5-flash"

	// Training output
	defaultTrainingDir = "../../data/training/extracts"

	// Telemetry
	defaultTelemetryLibrary   = "layer4_extract"
	defaultTelemetryOperation = "run_extract"

	// HTTP client timeouts
	defaultHTTPClientTimeout = 45 * time.Second
	defaultDialTimeout       = 5 * time.Second
	defaultCallTimeout       = 3 * time.Second

	// Preview and display limits
	previewMaxLength      = 200
	documentPreviewLength = 120
	maxExamplePreviews    = 3
	maxDocumentPreviews   = 3

	// Data type distribution defaults
	defaultStringRatio  = 0.4
	defaultNumberRatio  = 0.4
	defaultBooleanRatio = 0.1
	defaultDateRatio    = 0.05
	defaultArrayRatio   = 0.03
	defaultObjectRatio  = 0.02
)

func main() {
	explorer := flag.Bool("explorer", false, "start the catalog explorer")
	flag.Parse()

	logger := log.New(os.Stdout, "[extract-service] ", log.LstdFlags|log.Lmsgprefix)

	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		logger.Fatalf("failed to load configuration: %v", err)
	}

	server := &extractServer{
		client:         &http.Client{Timeout: config.DefaultHTTPClientTimeout},
		langextractURL: cfg.Langextract.URL,
		apiKey:         cfg.Langextract.APIKey,
		trainingDir:    cfg.Training.OutputDir,
		logger:         logger,
		ocrCommand:     handlers.DeriveOCRCommand(),

		// Persistence config
		sqlitePath:    cfg.Persistence.SQLitePath,
		redisAddr:     cfg.Persistence.RedisAddr,
		redisPassword: cfg.Persistence.RedisPassword,
		redisDB:       cfg.Persistence.RedisDB,
		neo4jURI:      cfg.Persistence.Neo4jURI,
		neo4jUsername: cfg.Persistence.Neo4jUsername,
		neo4jPassword: cfg.Persistence.Neo4jPassword,

		// Document store
		docStorePath: cfg.Persistence.DocStorePath,

		// DeepAgents client (enabled by default, 10/10 integration)
		deepAgentsClient: clients.NewDeepAgentsClient(logger),

		// Domain detector for associating extracted data with domains
		domainDetector: extraction.NewDomainDetector(os.Getenv("LOCALAI_URL"), logger),

		// AgentFlow client for direct integration
		agentFlowClient: clients.NewAgentFlowClient(logger),

		// Metrics collector for all improvements
		metricsCollector: monitoring.GetMetricsCollector(logger),
	}

	if cfg.AgentTelemetry.BaseURL != "" {
		agentTelemetryClient, err := telemetryclient.NewClient(cfg.AgentTelemetry.BaseURL, server.client)
		if err != nil {
			logger.Printf("agent telemetry disabled: %v", err)
		} else {
			server.agentTelemetry = agentTelemetryClient
			logger.Printf("agent telemetry enabled (base=%s)", cfg.AgentTelemetry.BaseURL)
		}
	} else {
		logger.Println("agent telemetry disabled: AGENT_METRICS_BASE_URL not set")
	}

	// Create persistence layer
	var graphPersistences []persistence.GraphPersistence
	var neo4jPersistence *storage.Neo4jPersistence
	var realTimeGleanExporter *persistence.RealTimeGleanExporter

	if server.neo4jURI != "" {
		var err error
		neo4jPersistence, err = storage.NewNeo4jPersistence(
			server.neo4jURI,
			server.neo4jUsername,
			server.neo4jPassword,
			cfg.Persistence.EnableCatalogSchemaIntegration,
			cfg.Persistence.CatalogResourceBaseURI,
		)
		if err != nil {
			logger.Fatalf("failed to create neo4j persistence: %v", err)
		}
		// Verify connectivity
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := neo4jPersistence.Driver().VerifyConnectivity(ctx); err != nil {
			logger.Fatalf("failed to connect to neo4j: %v", err)
		}
		graphPersistences = append(graphPersistences, neo4jPersistence)
		server.neo4jPersistence = neo4jPersistence
		logger.Println("connected to neo4j")
	}

	var gleanPersistence *persistence.GleanPersistence
	if exportDir := strings.TrimSpace(os.Getenv("GLEAN_EXPORT_DIR")); exportDir != "" {
		predicatePrefix := strings.TrimSpace(os.Getenv("GLEAN_PREDICATE_PREFIX"))
		var err error
		gleanPersistence, err = persistence.NewGleanPersistence(exportDir, predicatePrefix, logger)
		if err != nil {
			logger.Fatalf("failed to create glean persistence: %v", err)
		}
		graphPersistences = append(graphPersistences, gleanPersistence)
		logger.Printf("glean export enabled (dir=%s, prefix=%s)", gleanPersistence.ExportDir(), gleanPersistence.PredicatePrefix())

		// Initialize real-time Glean exporter if enabled
		dbName := strings.TrimSpace(os.Getenv("GLEAN_DB_NAME"))
		schemaPath := strings.TrimSpace(os.Getenv("GLEAN_SCHEMA_PATH"))
		if schemaPath == "" {
			schemaRoot := strings.TrimSpace(os.Getenv("GLEAN_SCHEMA_ROOT"))
			if schemaRoot != "" {
				schemaPath = filepath.Join(schemaRoot, "source", "etl.angle")
			} else {
				schemaPath = filepath.Join("glean", "schema", "source", "etl.angle")
			}
		}
		realTimeGleanExporter = persistence.NewRealTimeGleanExporter(gleanPersistence, dbName, schemaPath, logger)
		server.realTimeGleanExporter = realTimeGleanExporter
	}

	server.graphPersistence = persistence.NewCompositeGraphPersistence(graphPersistences...)

	if hr := schema.NewHANASchemaReplication(logger); hr != nil {
		server.hanaReplication = hr
		logger.Println("hana schema replication configured")
		defer hr.Close()
	}

	if pr := schema.NewPostgresSchemaReplication(logger); pr != nil {
		server.postgresReplication = pr
		logger.Println("postgres schema replication configured")
		defer pr.Close()
	}

	// Create document persistence layer
	if server.docStorePath != "" {
		docPersistence, err := persistence.NewFilePersistence(server.docStorePath)
		if err != nil {
			logger.Fatalf("failed to create file persistence: %v", err)
		}
		server.docPersistence = docPersistence
		logger.Println("document persistence enabled")
	}

	// Create table persistence layer
	if server.sqlitePath != "" {
		sqlitePersistence, err := storage.NewSQLitePersistence(server.sqlitePath)
		if err != nil {
			logger.Fatalf("failed to create sqlite persistence: %v", err)
		}
		server.tablePersistence = sqlitePersistence
		logger.Println("sqlite persistence enabled")
	}

	// Create vector persistence layers (Phase 2 & 3: pgvector and OpenSearch integration)
	var vectorStores []persistence.VectorPersistence
	var primaryStore persistence.VectorPersistence

	// Initialize pgvector (primary store for structured queries)
	if pgVectorDSN := os.Getenv("POSTGRES_VECTOR_DSN"); pgVectorDSN != "" {
		pgPersistence, err := persistence.NewPgVectorPersistence(pgVectorDSN, logger)
		if err != nil {
			logger.Printf("failed to initialize pgvector persistence: %v", err)
		} else {
			primaryStore = pgPersistence
			vectorStores = append(vectorStores, pgPersistence)
			logger.Println("pgvector persistence enabled")
		}
	}

	// Initialize OpenSearch (secondary store for semantic/hybrid search)
	if opensearchURL := os.Getenv("OPENSEARCH_URL"); opensearchURL != "" {
		opensearchPersistence, err := persistence.NewOpenSearchPersistence(opensearchURL, logger)
		if err != nil {
			logger.Printf("failed to initialize OpenSearch persistence: %v", err)
		} else {
			vectorStores = append(vectorStores, opensearchPersistence)
			logger.Println("OpenSearch persistence enabled")
		}
	}

	// Initialize Redis (fallback/cache store)
	if server.redisAddr != "" {
		redisPersistence, err := storage.NewRedisPersistence(server.redisAddr, server.redisPassword, server.redisDB)
		if err != nil {
			logger.Printf("failed to create redis persistence: %v", err)
		} else {
			vectorStores = append(vectorStores, redisPersistence)
			logger.Println("redis persistence enabled (as cache/fallback)")
		}
	}

	// Create composite persistence if multiple stores are available
	if len(vectorStores) > 1 {
		// Use pgvector as primary, others as secondary
		var secondary []persistence.VectorPersistence
		for _, store := range vectorStores {
			if store != primaryStore {
				secondary = append(secondary, store)
			}
		}
		server.vectorPersistence = persistence.NewCompositeVectorPersistence(primaryStore, secondary, logger)
		logger.Println("composite vector persistence enabled")
	} else if len(vectorStores) == 1 {
		// Single store
		server.vectorPersistence = vectorStores[0]
		logger.Println("single vector persistence enabled")
	} else {
		logger.Println("no vector persistence configured")
	}

	// Initialize Orchestration chain matcher (Phase 2 integration)
	chainMatcher := integrations.NewOrchestrationChainMatcher(logger)
	baseURL := fmt.Sprintf("http://localhost:%s", os.Getenv("PORT"))
	if baseURL == "http://localhost:" {
		baseURL = "http://localhost:8081"
	}
	chainMatcher.SetExtractServiceURL(baseURL)
	server.chainMatcher = chainMatcher

	// Initialize embedding cache and batch generator (Phase 3 optimization)
	cacheMaxSize := parseIntEnv(os.Getenv("EMBEDDING_CACHE_SIZE"), 1000)
	cacheTTL := 24 * time.Hour // Default 24 hour TTL
	if ttlStr := os.Getenv("EMBEDDING_CACHE_TTL"); ttlStr != "" {
		if ttl, err := time.ParseDuration(ttlStr); err == nil {
			cacheTTL = ttl
		}
	}
	server.embeddingCache = embeddings.NewEmbeddingCache(cacheMaxSize, cacheTTL, logger)

	batchSize := parseIntEnv(os.Getenv("EMBEDDING_BATCH_SIZE"), 10)
	server.batchEmbeddingGen = embeddings.NewBatchEmbeddingGenerator(logger, server.embeddingCache, batchSize)
	logger.Printf("Embedding cache initialized (max_size=%d, ttl=%v, batch_size=%d)", cacheMaxSize, cacheTTL, batchSize)

	// Initialize training data collector (Phase 4 full model utilization)
	trainingDataPath := os.Getenv("SAP_RPT_TRAINING_DATA_PATH")
	if trainingDataPath == "" {
		trainingDataPath = "./training_data/sap_rpt_classifications.json"
	}
	server.trainingDataCollector = terminology.NewTrainingDataCollector(trainingDataPath, logger)
	if os.Getenv("COLLECT_TRAINING_DATA") == "true" {
		logger.Printf("Training data collection enabled (path=%s)", trainingDataPath)
	}

	// Initialize model monitor (Phase 5 advanced capabilities)
	metricsPath := os.Getenv("MODEL_METRICS_PATH")
	if metricsPath == "" {
		metricsPath = "./training_data/model_metrics.json"
	}
	server.modelMonitor = terminology.NewModelMonitor(metricsPath, logger)
	if os.Getenv("MODEL_MONITORING_ENABLED") == "true" {
		logger.Printf("Model monitoring enabled (path=%s)", metricsPath)
	}

	// Initialize multi-modal extractor (Phase 6 unified integration)
	server.multiModalExtractor = extraction.NewMultiModalExtractor(logger)
	if os.Getenv("USE_MULTIMODAL_EXTRACTION") == "true" {
		logger.Printf("Multi-modal extraction enabled (OCR: %v)", os.Getenv("USE_DEEPSEEK_OCR") == "true")
	}

	// Initialize MarkItDown integration
	markitdownMetricsCollector := func(service, endpoint string, statusCode int, latency time.Duration, correlationID string) {
		if logger != nil {
			logger.Printf("[%s] MarkItDown integration: %s %s -> %d (latency: %v)",
				correlationID, service, endpoint, statusCode, latency)
		}
	}
	markitdownClient := clients.NewMarkItDownClient("", logger, markitdownMetricsCollector)
	markitdownAdapter := &markitdownClientAdapter{client: markitdownClient}
	server.markitdownIntegration = integrations.NewMarkItDownIntegration(markitdownAdapter, logger)
	// MarkItDown integration is initialized above
	if server.markitdownIntegration != nil {
		logger.Printf("MarkItDown integration initialized")
	}

	// Phase 8.1: Initialize semantic schema analyzer
	server.semanticSchemaAnalyzer = extraction.NewSemanticSchemaAnalyzer(logger)
	logger.Println("Semantic schema analyzer initialized (Phase 8.1)")

	// Phase 10: Initialize terminology learner with LNN
	terminologyStore := terminology.NewNeo4jTerminologyStore(server.neo4jPersistence, logger)
	terminologyLearner := terminology.NewTerminologyLearner(terminologyStore, logger)

	// Load existing terminology from store
	if err := terminologyLearner.LoadTerminology(context.Background()); err != nil {
		logger.Printf("Warning: Failed to load existing terminology: %v", err)
	}

	// Set global terminology learner for embedding enhancement
	embeddings.SetGlobalTerminologyLearner(terminologyLearner)

	// Wire terminology learner to components
	server.semanticSchemaAnalyzer.SetTerminologyLearner(terminologyLearner)
	logger.Println("Terminology learner initialized (Phase 10)")

	// Initialize SAP BDC integration
	server.sapBDCIntegration = integrations.NewSAPBDCIntegration(logger)
	logger.Println("SAP Business Data Cloud integration initialized")

	// Phase 9.2: Initialize self-healing system
	server.selfHealingSystem = monitoring.NewSelfHealingSystem(logger)
	logger.Println("Self-healing system initialized (Phase 9.2)")

	// Register health monitors for critical services
	if server.neo4jPersistence != nil {
		server.selfHealingSystem.RegisterHealthMonitor(
			"neo4j",
			30*time.Second,
			func() error {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				return server.neo4jPersistence.Driver().VerifyConnectivity(ctx)
			},
		)
	}

	server.telemetryOperation = cfg.Telemetry.Operation

	if cfg.Telemetry.Enabled && cfg.Telemetry.Address != "" {
		telemetryClient, err := monitoring.NewTelemetryClient(context.Background(), monitoring.TelemetryConfig{
			Address:          cfg.Telemetry.Address,
			LibraryType:      cfg.Telemetry.LibraryType,
			DefaultOperation: cfg.Telemetry.Operation,
			PrivacyLevel:     cfg.Telemetry.PrivacyLevel,
			UserIDHash:       cfg.Telemetry.UserIDHash,
			DialTimeout:      cfg.Telemetry.DialTimeout,
			CallTimeout:      cfg.Telemetry.CallTimeout,
		})
		if err != nil {
			logger.Printf("telemetry disabled: %v", err)
		} else {
			server.telemetry = telemetryClient
			logger.Printf("telemetry enabled (addr=%s, library=%s)", cfg.Telemetry.Address, cfg.Telemetry.LibraryType)
			defer telemetryClient.Close()
		}
	} else if cfg.Telemetry.Enabled && cfg.Telemetry.Address == "" {
		logger.Printf("telemetry disabled: POSTGRES_LANG_SERVICE_ADDR not set")
	}

	if err := os.MkdirAll(cfg.Training.OutputDir, 0o755); err != nil {
		logger.Fatalf("failed to prepare training directory: %v", err)
	}

	grpcAddr := ":" + cfg.Server.GRPCPort
	flightAddr := cfg.Server.FlightAddr

	flightServer := newExtractFlightServer(logger)
	server.flight = flightServer

	catalog, err := catalog.NewCatalog("catalog.json")
	if err != nil {
		logger.Fatalf("failed to create catalog: %v", err)
	}
	server.catalog = catalog

	// Initialize catalog service client (optional - won't fail if not configured)
	catalogServiceURL := os.Getenv("CATALOG_SERVICE_URL")
	if catalogServiceURL == "" {
		catalogServiceURL = "http://localhost:8084" // Default catalog service URL
	}
	// Create metrics collector for catalog integration
	var metricsCollector clients.MetricsCollector
	// In production, this would integrate with Prometheus or similar
	metricsCollector = func(service, endpoint string, statusCode int, latency time.Duration, correlationID string) {
		if logger != nil {
			logger.Printf("[%s] Catalog integration: %s %s -> %d (latency: %v)",
				correlationID, service, endpoint, statusCode, latency)
		}
	}
	server.catalogClient = clients.NewCatalogClient(catalogServiceURL, logger, metricsCollector)
	if catalogServiceURL != "" {
		logger.Printf("Catalog service client initialized (url=%s)", catalogServiceURL)
	}

	go func() {
		if err := server.startGRPCServer(grpcAddr); err != nil {
			logger.Fatalf("gRPC server exited: %v", err)
		}
	}()

	go func() {
		if err := flightServer.Start(flightAddr); err != nil {
			logger.Fatalf("Flight server exited: %v", err)
		}
	}()

	// Initialize OpenTelemetry tracing
	tracerProvider, err := observability.InitTracing("extract-service", logger)
	if err != nil {
		logger.Printf("failed to initialize tracing: %v", err)
	} else if tracerProvider != nil {
		defer tracerProvider.Shutdown(context.Background())
	}

	// Initialize structured logging
	structuredLogger := middleware.NewStructuredLogger(logger)

	// Initialize health checker
	healthChecker := middleware.NewHealthChecker(logger)
	// Register health checks
	if server.neo4jPersistence != nil && server.neo4jPersistence.Driver() != nil {
		// Note: Neo4j driver doesn't expose *sql.DB, so we'll check it differently
		healthChecker.RegisterCheck("neo4j", func(ctx context.Context) error {
			return server.neo4jPersistence.Driver().VerifyConnectivity(ctx)
		}, 5*time.Second)
	}

	// Initialize authentication middleware
	authConfig := middleware.LoadAuthConfig()
	authMiddleware := middleware.NewAuthMiddleware(authConfig, logger)
	if authConfig.Enabled {
		logger.Printf("authentication enabled (type=%s)", authConfig.AuthType)
	} else {
		logger.Println("authentication disabled")
	}

	mux := http.NewServeMux()

	// Health check endpoints (registered before middleware)
	mux.HandleFunc("/health", healthChecker.HandleHealth)
	mux.HandleFunc("/ready", healthChecker.HandleReady)
	mux.HandleFunc("/healthz", server.handleHealthz)
	mux.HandleFunc("/extract", server.handleExtract)
	mux.HandleFunc("/generate/training", server.handleGenerateTraining)
	mux.HandleFunc("/knowledge-graph", server.handleGraph)                                            // Main knowledge graph processing endpoint
	mux.HandleFunc("/graph", server.handleGraph)                                                      // Legacy alias for backward compatibility
	mux.HandleFunc("/knowledge-graph/query", server.handleNeo4jQuery)                                 // Neo4j Cypher query endpoint
	mux.HandleFunc("/documents/upload", server.handleDocumentUpload)                                 // Document upload endpoint (replaces DMS)
	mux.HandleFunc("/documents", server.handleDocumentsRouter)                                        // Document router (list and get)
	mux.HandleFunc("/workflow/petri-to-langgraph", server.handlePetriNetToLangGraph)                  // Convert Petri net to LangGraph
	mux.HandleFunc("/workflow/petri-to-langgraph-advanced", server.handlePetriNetToAdvancedLangGraph) // Convert Petri net to advanced LangGraph (Phase 7.3)
	mux.HandleFunc("/workflow/petri-to-agentflow", server.handlePetriNetToAgentFlow)                  // Convert Petri net to AgentFlow
	mux.HandleFunc("/agentflow/run", server.handleAgentFlowRun)                                       // Direct AgentFlow execution
	// Phase 10: Terminology learning endpoints
	mux.HandleFunc("/terminology/domains", server.handleTerminologyDomains)     // List learned domains
	mux.HandleFunc("/terminology/roles", server.handleTerminologyRoles)         // List learned roles
	mux.HandleFunc("/terminology/patterns", server.handleTerminologyPatterns)   // List learned naming patterns
	mux.HandleFunc("/terminology/learn", server.handleTerminologyLearn)         // Trigger manual learning
	mux.HandleFunc("/terminology/evolution", server.handleTerminologyEvolution) // Terminology evolution over time
	// SAP Business Data Cloud integration endpoints
	mux.HandleFunc("/sap-bdc/extract", server.handleSAPBDCExtract)                  // Extract from SAP BDC
	mux.HandleFunc("/semantic/analyze-column", server.handleAnalyzeColumnSemantics) // Semantic column analysis (Phase 8.1)
	mux.HandleFunc("/semantic/analyze-lineage", server.handleAnalyzeDataLineage)    // Semantic data lineage analysis (Phase 8.1)
	mux.HandleFunc("/health/status", server.handleHealthStatus)                     // Health status for all services (Phase 9.2)
	mux.HandleFunc("/knowledge-graph/queries", server.handleGraphQueryHelpers)      // Get common graph query helpers
	mux.HandleFunc("/knowledge-graph/search", server.handleVectorSearch)            // Vector similarity search (RAG)
	mux.HandleFunc("/knowledge-graph/embed", server.handleGenerateEmbedding)        // Generate embedding for text
	mux.HandleFunc("/knowledge-graph/embed/", server.handleGetEmbedding)            // Get embedding by key
	mux.HandleFunc("/training-data/stats", server.handleTrainingDataStats)          // Get training data statistics (Phase 4)
	mux.HandleFunc("/training-data/export", server.handleExportTrainingData)        // Export training data (Phase 4)
	mux.HandleFunc("/model/metrics", server.handleModelMetrics)                     // Get model performance metrics (Phase 5)
	mux.HandleFunc("/model/uncertain", server.handleUncertainPredictions)           // Get uncertain predictions for review (Phase 5)
	mux.HandleFunc("/signavio/agent-metrics/", server.handleSignavioAgentMetrics)
	mux.HandleFunc("/catalog/projects", server.handleGetProjects)
	mux.HandleFunc("/catalog/projects/add", server.handleAddProject)
	mux.HandleFunc("/catalog/systems", server.handleGetSystems)
	mux.HandleFunc("/catalog/systems/add", server.handleAddSystem)
	mux.HandleFunc("/catalog/information-systems", server.handleGetInformationSystems)
	mux.HandleFunc("/catalog/information-systems/add", server.handleAddInformationSystem)
	mux.HandleFunc("/webhooks/gitea", server.handleGiteaWebhook) // Gitea webhook endpoint for automatic pipeline triggering
	mux.HandleFunc("/ui", server.handleWebUI)
	mux.HandleFunc("/metrics/improvements", server.handleImprovementsMetrics) // Metrics for all 6 improvements
	// Gitea repository management endpoints
	mux.HandleFunc("/gitea/repositories", server.handleGiteaRepositories)        // List and create repositories
	mux.HandleFunc("/gitea/repositories/", server.handleGiteaRepositoryRouter)    // Route to specific repository handlers

	// Apply middleware chain
	var handler http.Handler = mux
	handler = middleware.LoggingMiddleware(structuredLogger)(handler)
	if authConfig.Enabled {
		handler = authMiddleware.Middleware(handler)
	}

	if *explorer {
		server.startExplorer()
	} else {
		addr := ":" + cfg.Server.Port
		logger.Printf("extract service listening on %s (proxying %s)", addr, server.langextractURL)
		if err := http.ListenAndServe(addr, handler); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Fatalf("server exited with error: %v", err)
		}
	}
}

type extractServer struct {
	extractpb.UnimplementedExtractServiceServer
	client         *http.Client
	langextractURL string
	apiKey         string
	trainingDir    string
	logger         *log.Logger
	ocrCommand     []string

	// Persistence config
	sqlitePath    string
	redisAddr     string
	redisPassword string
	redisDB       int
	neo4jURI      string
	neo4jUsername string
	neo4jPassword string

	// Document store
	docStorePath   string
	docPersistence persistence.DocumentPersistence

	// DeepAgents client
	deepAgentsClient *clients.DeepAgentsClient

	// Domain detector for associating extracted data with domains
	domainDetector *extraction.DomainDetector

	// AgentFlow client for direct integration
	agentFlowClient *clients.AgentFlowClient

	tablePersistence       persistence.TablePersistence
	vectorPersistence      persistence.VectorPersistence
	graphPersistence       persistence.GraphPersistence
	neo4jPersistence       *storage.Neo4jPersistence          // Direct Neo4j access for queries
	realTimeGleanExporter  *persistence.RealTimeGleanExporter // Real-time Glean synchronization
	flight                 *extractFlightServer
	semanticSchemaAnalyzer *extraction.SemanticSchemaAnalyzer // Phase 8.1: Semantic schema understanding
	selfHealingSystem      *monitoring.SelfHealingSystem      // Phase 9.2: Self-healing system
	sapBDCIntegration      *integrations.SAPBDCIntegration    // SAP Business Data Cloud integration
	hanaReplication        *schema.HANAReplication
	postgresReplication    *schema.PostgresReplication
	metricsCollector       *monitoring.MetricsCollector // Metrics for all improvements

	telemetry          interface{}
	telemetryOperation string
	catalog            *catalog.Catalog
	catalogClient      *clients.CatalogClient // HTTP client for catalog service integration

	// Orchestration chain matcher (for Phase 2 integration)
	chainMatcher *integrations.OrchestrationChainMatcher

	// Embedding cache and batch generator (for Phase 3 optimization)
	embeddingCache    *embeddings.EmbeddingCache
	batchEmbeddingGen *embeddings.BatchEmbeddingGenerator

	// Training data collector (for Phase 4 full model utilization)
	trainingDataCollector *terminology.TrainingDataCollector

	// Model monitor (for Phase 5 advanced capabilities)
	modelMonitor *terminology.ModelMonitor

	// Multi-modal extractor (for Phase 6 unified integration)
	multiModalExtractor *extraction.MultiModalExtractor

	// MarkItDown integration for document conversion
	markitdownIntegration *integrations.MarkItDownIntegration

	// Agent telemetry client for Signavio exposure
	agentTelemetry *telemetryclient.Client
}

func parseIntEnv(envVar string, defaultValue int) int {
	if envVar == "" {
		return defaultValue
	}
	val, err := strconv.Atoi(envVar)
	if err != nil {
		return defaultValue
	}
	return val
}

func parseBoolEnv(value string, defaultValue bool) bool {
	if value == "" {
		return defaultValue
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "t", "yes", "y", "on":
		return true
	case "0", "false", "f", "no", "n", "off":
		return false
	default:
		return defaultValue
	}
}

// --- /healthz ---
func (s *extractServer) handleHealthz(w http.ResponseWriter, r *http.Request) {
	status := map[string]any{
		"status":        "ok",
		"service":       "extract",
		"langextract":   s.langextractURL,
		"training_dir":  s.trainingDir,
		"ocr_available": len(s.ocrCommand) > 0,
	}

	// Check DeepAgents health if enabled (10/10 integration)
	if s.deepAgentsClient != nil && s.deepAgentsClient.IsEnabled() {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		deepAgentsHealthy := s.deepAgentsClient.CheckHealth(ctx)
		status["deepagents"] = map[string]any{
			"enabled": true,
			"healthy": deepAgentsHealthy,
		}
	} else {
		status["deepagents"] = map[string]any{
			"enabled": false,
		}
	}

	handlers.WriteJSON(w, http.StatusOK, status)
}

// --- /extract ---

func (s *extractServer) handleExtract(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer r.Body.Close()

	ctx := r.Context()
	started := time.Now()
	sessionID := strings.TrimSpace(r.Header.Get("X-Session-ID"))

	var req extractRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	inputSummary := telemetryInputFromRequest(req)

	response, err := s.runExtract(ctx, req)
	latency := time.Since(started)

	var outputSummary map[string]any
	if err == nil {
		outputSummary = telemetryOutputFromResponse(&response)
	}
	if logErr := s.recordTelemetry(ctx, sessionID, inputSummary, outputSummary, err, started, latency); logErr != nil {
		s.logger.Printf("telemetry warning: %v", logErr)
	}

	if err != nil {
		var extractErr *extractError
		if errors.As(err, &extractErr) {
			http.Error(w, extractErr.Error(), extractErr.status)
			return
		}
		http.Error(w, fmt.Sprintf("extract failed: %v", err), http.StatusInternalServerError)
		return
	}

	handlers.WriteJSON(w, http.StatusOK, response)
}

func (s *extractServer) recordTelemetry(ctx context.Context, sessionID string, input map[string]any, output map[string]any, runErr error, started time.Time, latency time.Duration) error {
	if s.telemetry == nil {
		if s.logger != nil {
			s.logger.Printf(
				"telemetry disabled, event dropped: op=%s session=%s latency=%s err=%v",
				s.telemetryOperation,
				sessionID,
				latency,
				runErr,
			)
		}
		return nil
	}

	if latency < 0 {
		latency = 0
	}

	completed := started.Add(latency)

	record := monitoring.TelemetryRecord{
		Operation:    s.telemetryOperation,
		Input:        input,
		Output:       output,
		Error:        runErr,
		StartedAt:    started,
		CompletedAt:  completed,
		Latency:      latency,
		SessionID:    sessionID,
		PrivacyLevel: "",
		UserIDHash:   "",
	}

	if runErr != nil {
		var extractErr *extractError
		if errors.As(runErr, &extractErr) && extractErr != nil && extractErr.err != nil {
			record.ErrorMessage = extractErr.err.Error()
		} else {
			record.ErrorMessage = runErr.Error()
		}
	}

	if telemetryClient, ok := s.telemetry.(*monitoring.TelemetryClient); ok {
		return telemetryClient.Log(ctx, record)
	}
	return nil
}

// --- /generate/training ---

func (s *extractServer) handleGenerateTraining(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req trainingGenerationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	mode := strings.ToLower(req.Mode)
	if mode == "" {
		mode = "document"
	}

	ctx := r.Context()
	switch mode {
	case "table", "tables":
		result, err := s.generateTableExtract(ctx, req.TableOptions)
		if err != nil {
			http.Error(w, fmt.Sprintf("table extract failed: %v", err), http.StatusBadGateway)
			return
		}
		handlers.WriteJSON(w, http.StatusOK, map[string]any{
			"success":  true,
			"mode":     "table",
			"manifest": result.ManifestPath,
			"files":    result.FilePaths,
		})
	case "document", "documents":
		result, err := s.generateDocumentExtract(ctx, req.DocumentOptions)
		if err != nil {
			http.Error(w, fmt.Sprintf("document extract failed: %v", err), http.StatusBadGateway)
			return
		}
		handlers.WriteJSON(w, http.StatusOK, map[string]any{
			"success":  true,
			"mode":     "document",
			"manifest": result.ManifestPath,
			"files":    result.FilePaths,
		})
	default:
		http.Error(w, fmt.Sprintf("unsupported mode %q", mode), http.StatusBadRequest)
	}
}

// --- /knowledge-graph ---
// Processes knowledge graphs: extracts schema, relationships, and metadata from
// JSON tables, Hive DDLs, SQL queries, and Control-M files.
// Returns normalized graph with nodes (tables, columns, jobs) and edges (relationships).
// This is different from LangGraph workflow graphs which execute agent workflows.

// Node and Edge types are now in pkg/graph/types.go

type signavioAgentMetricsResponse struct {
	SessionID string                      `json:"session_id"`
	Metrics   signavioAgentMetricsSummary `json:"metrics"`
	Events    []signavioAgentMetricEvent  `json:"events"`
}

type signavioAgentMetricsSummary struct {
	UserPromptCount      int     `json:"user_prompt_count"`
	ToolCallStartedCount int     `json:"tool_call_started_count"`
	ToolCallCount        int     `json:"tool_call_count"`
	ToolSuccessCount     int     `json:"tool_success_count"`
	ToolErrorCount       int     `json:"tool_error_count"`
	ToolSuccessRate      float64 `json:"tool_success_rate"`
	AverageToolLatencyMs float64 `json:"avg_tool_latency_ms"`
	ModelChangeCount     int     `json:"model_change_count"`
	LastUserPrompt       string  `json:"last_user_prompt,omitempty"`
}

type signavioAgentMetricEvent struct {
	Timestamp time.Time                            `json:"timestamp"`
	SessionID string                               `json:"session_id"`
	Type      telemetryclient.AgentMetricEventKind `json:"type"`
	Payload   map[string]any                       `json:"payload"`
}

func (s *extractServer) handleSignavioAgentMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.agentTelemetry == nil {
		http.Error(w, "agent telemetry unavailable", http.StatusServiceUnavailable)
		return
	}

	const prefix = "/signavio/agent-metrics/"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		http.NotFound(w, r)
		return
	}

	sessionID := strings.Trim(strings.TrimPrefix(r.URL.Path, prefix), "/")
	if sessionID == "" {
		http.Error(w, "session id is required", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	telemetryResp, err := s.agentTelemetry.GetEvents(ctx, sessionID)
	if err != nil {
		s.logger.Printf("agent telemetry fetch failed for session %s: %v", sessionID, err)
		http.Error(w, "failed to fetch agent telemetry", http.StatusBadGateway)
		return
	}

	summary := signavioAgentMetricsSummary{}
	var latencySum float64
	var latencyCount int

	events := make([]signavioAgentMetricEvent, 0, len(telemetryResp.Events))
	for _, event := range telemetryResp.Events {
		payloadCopy := make(map[string]any, len(event.Payload))
		for k, v := range event.Payload {
			payloadCopy[k] = v
		}

		events = append(events, signavioAgentMetricEvent{
			Timestamp: event.Timestamp,
			SessionID: event.SessionID,
			Type:      event.Type,
			Payload:   payloadCopy,
		})

		switch event.Type {
		case telemetryclient.EventUserPrompt:
			summary.UserPromptCount++
			if prompt, ok := payloadCopy["prompt"].(string); ok && prompt != "" {
				summary.LastUserPrompt = prompt
			}
		case telemetryclient.EventToolCallStarted:
			summary.ToolCallStartedCount++
		case telemetryclient.EventToolCallCompleted:
			summary.ToolCallCount++
			if success, ok := payloadCopy["success"].(bool); ok {
				if success {
					summary.ToolSuccessCount++
				} else {
					summary.ToolErrorCount++
				}
			}
			if latency, ok := payloadCopy["latency_ms"].(float64); ok {
				latencySum += latency
				latencyCount++
			}
		case telemetryclient.EventModelChange:
			summary.ModelChangeCount++
		}
	}

	if summary.ToolCallCount > 0 {
		summary.ToolSuccessRate = float64(summary.ToolSuccessCount) / float64(summary.ToolCallCount)
	}
	if latencyCount > 0 {
		summary.AverageToolLatencyMs = latencySum / float64(latencyCount)
	}

	response := signavioAgentMetricsResponse{
		SessionID: telemetryResp.SessionID,
		Metrics:   summary,
		Events:    events,
	}

	handlers.WriteJSON(w, http.StatusOK, response)
}

type graphRequest struct {
	JSONTables          []string           `json:"json_tables"`
	HiveDDLs            []string           `json:"hive_ddls"`
	SqlQueries          []string           `json:"sql_queries"`
	ControlMFiles       []string           `json:"control_m_files"`
	SignavioFiles       []string           `json:"signavio_files"`
	DocumentFiles       []string           `json:"document_files,omitempty"` // Files to process through markitdown/OCR
	GitRepositories     []gitRepositoryReq `json:"git_repositories,omitempty"`
	GiteaStorage        *giteaStorageReq   `json:"gitea_storage,omitempty"`
	ViewLineage         []map[string]any   `json:"view_lineage,omitempty"`
	IdealDistribution   map[string]float64 `json:"ideal_distribution"`
	ProjectID           string             `json:"project_id"`
	SystemID            string             `json:"system_id"`
	InformationSystemID string             `json:"information_system_id"`
	AIEnabled           bool               `json:"ai_enabled,omitempty"`
	AIModel             string             `json:"ai_model,omitempty"`
}

type gitRepositoryReq struct {
	URL         string   `json:"url"`
	Type        string   `json:"type,omitempty"`
	Branch      string   `json:"branch,omitempty"`
	Tag         string   `json:"tag,omitempty"`
	Commit      string   `json:"commit,omitempty"`
	FilePatterns []string `json:"file_patterns,omitempty"`
	Auth        struct {
		Type     string `json:"type,omitempty"`
		Token    string `json:"token,omitempty"`
		KeyPath  string `json:"key_path,omitempty"`
		Username string `json:"username,omitempty"`
		Password string `json:"password,omitempty"`
	} `json:"auth,omitempty"`
}

type giteaStorageReq struct {
	Enabled     bool   `json:"enabled,omitempty"`
	GiteaURL    string `json:"gitea_url,omitempty"`
	GiteaToken  string `json:"gitea_token,omitempty"`
	Owner       string `json:"owner,omitempty"`
	RepoName    string `json:"repo_name,omitempty"`
	Branch      string `json:"branch,omitempty"`
	BasePath    string `json:"base_path,omitempty"`
	AutoCreate  bool   `json:"auto_create,omitempty"`
	Description string `json:"description,omitempty"`
}

func (s *extractServer) handleGraph(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req graphRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	started := time.Now()

	var nodes []graph.Node
	var edges []graph.Edge
	var rootID string
	signavioMetadata := integrations.SignavioMetadata{}

	if len(req.SignavioFiles) > 0 {
		signavioNodes, signavioEdges, metadata := integrations.LoadSignavioArtifacts(req.SignavioFiles, s.logger)
		if len(signavioNodes) > 0 {
			nodes = append(nodes, signavioNodes...)
		}
		if len(signavioEdges) > 0 {
			edges = append(edges, signavioEdges...)
		}
		signavioMetadata = metadata

		if metadata.ProcessCount > 0 {
			s.logger.Printf("✅ Loaded %d Signavio processes from %d files", metadata.ProcessCount, metadata.SourceFiles)
			if s.catalog != nil {
				var signavioCatalogDirty bool
				for _, proc := range metadata.Processes {
					// Convert integrations.SignavioProcessSummary to catalog.SignavioProcessSummary
					catalogProc := catalog.SignavioProcessSummary{
						ID:           proc.ID,
						Name:         proc.Name,
						SourceFile:   proc.SourceFile,
						ElementCount: proc.ElementCount,
						ElementTypes: proc.ElementTypes,
						Elements:     make([]catalog.SignavioElementSummary, len(proc.Elements)),
						Labels:       proc.Labels,
						Properties:   proc.Properties,
					}
					for i, elem := range proc.Elements {
						catalogProc.Elements[i] = catalog.SignavioElementSummary{
							ID:   elem.ID,
							Type: elem.Type,
							Name: elem.Name,
						}
					}
					s.catalog.UpsertSignavioProcess(catalogProc)
					signavioCatalogDirty = true
				}
				if signavioCatalogDirty {
					if err := s.catalog.Save(); err != nil {
						s.logger.Printf("failed to persist Signavio processes to catalog: %v", err)
					}
				}
			}
		} else if len(metadata.Errors) > 0 {
			for _, errMsg := range metadata.Errors {
				s.logger.Printf("Signavio ingestion warning: %s", errMsg)
			}
		}
	}

	for _, path := range req.JSONTables {
		if strings.TrimSpace(path) == "" {
			continue
		}
		if s.docPersistence != nil {
			if err := s.docPersistence.SaveDocument(path); err != nil {
				s.logger.Printf("failed to save document %s: %v", path, err)
			}
		}

		fileNodes, fileEdges, data, err := s.extractSchemaFromJSON(path)
		if err != nil {
			s.logger.Printf("failed to extract schema from %s: %v", path, err)
			continue
		}

		if s.tablePersistence != nil && data != nil {
			tableName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
			if err := s.tablePersistence.SaveTable(tableName, data); err != nil {
				s.logger.Printf("failed to save table %s: %v", tableName, err)
			}
		}

		nodes = append(nodes, fileNodes...)
		edges = append(edges, fileEdges...)
	}

	// Collect file system file paths for Gitea-first upload
	var allFileSystemPaths []string
	var documentFiles []string // Files that need document processing

	// Collect paths from file system sources
	if len(req.JSONTables) > 0 {
		allFileSystemPaths = append(allFileSystemPaths, req.JSONTables...)
		// Check for document files that need processing
		for _, path := range req.JSONTables {
			if git.IsDocumentFile(path) {
				documentFiles = append(documentFiles, path)
			}
		}
	}
	// Add explicit document files
	if len(req.DocumentFiles) > 0 {
		allFileSystemPaths = append(allFileSystemPaths, req.DocumentFiles...)
		documentFiles = append(documentFiles, req.DocumentFiles...)
	}
	if len(req.HiveDDLs) > 0 {
		// HiveDDLs can be file paths or DDL content strings
		for _, ddl := range req.HiveDDLs {
			ddl = strings.TrimSpace(ddl)
			if ddl == "" {
				continue
			}
			// Check if it's a file path (exists as file)
			if _, err := os.Stat(ddl); err == nil {
				allFileSystemPaths = append(allFileSystemPaths, ddl)
			}
		}
	}
	if len(req.ControlMFiles) > 0 {
		allFileSystemPaths = append(allFileSystemPaths, req.ControlMFiles...)
	}
	if len(req.SignavioFiles) > 0 {
		allFileSystemPaths = append(allFileSystemPaths, req.SignavioFiles...)
	}

	// Process documents with Gitea-first flow if configured
	if len(documentFiles) > 0 && req.GiteaStorage != nil && req.GiteaStorage.Enabled {
		giteaURL := req.GiteaStorage.GiteaURL
		if giteaURL == "" {
			giteaURL = os.Getenv("GITEA_URL")
		}
		giteaToken := req.GiteaStorage.GiteaToken
		if giteaToken == "" {
			giteaToken = os.Getenv("GITEA_TOKEN")
		}

		if giteaURL != "" && giteaToken != "" {
			giteaStorage := git.NewGiteaStorage(giteaURL, giteaToken, s.logger)
			storageConfig := git.StorageConfig{
				Owner:       req.GiteaStorage.Owner,
				RepoName:    req.GiteaStorage.RepoName,
				Branch:      req.GiteaStorage.Branch,
				BasePath:    "documents/processed/",
				ProjectID:   req.ProjectID,
				SystemID:    req.SystemID,
				AutoCreate:  req.GiteaStorage.AutoCreate,
				Description: fmt.Sprintf("Processed documents for project %s", req.ProjectID),
			}

			if storageConfig.RepoName == "" {
				storageConfig.RepoName = fmt.Sprintf("%s-documents", req.ProjectID)
			}
			if storageConfig.Owner == "" {
				storageConfig.Owner = "extract-service"
			}

			// Step 1: Upload raw documents to Gitea
			rawBasePath := "raw/documents/"
			giteaRepo, cloneURL, err := giteaStorage.UploadRawFiles(ctx, storageConfig, documentFiles, rawBasePath)
			if err != nil {
				s.logger.Printf("Warning: failed to upload raw documents to Gitea: %v", err)
			} else {
				s.logger.Printf("Uploaded %d raw documents to Gitea: %s", len(documentFiles), giteaRepo.HTMLURL)

				// Step 2: Clone from Gitea
				tempDir := filepath.Join(os.TempDir(), "extract-gitea-docs")
				if err := os.MkdirAll(tempDir, 0755); err != nil {
					s.logger.Printf("failed to create temp dir for Gitea clone: %v", err)
				} else {
					branch := storageConfig.Branch
					if branch == "" {
						branch = "main"
					}
					clonePath, err := giteaStorage.CloneFromGitea(ctx, cloneURL, branch, tempDir)
					if err != nil {
						s.logger.Printf("Warning: failed to clone from Gitea: %v", err)
					} else {
						// Cleanup cloned repo at end
						cleanupClone := func() {
							if err := os.RemoveAll(clonePath); err != nil {
								s.logger.Printf("Warning: failed to cleanup Gitea clone: %v", err)
							}
						}
						defer cleanupClone()

						// Step 3: Get document paths from Gitea clone
						rawPathInClone := filepath.Join(clonePath, rawBasePath)
						var documentPathsFromClone []string
						if entries, err := os.ReadDir(rawPathInClone); err == nil {
							for _, entry := range entries {
								if !entry.IsDir() {
									documentPathsFromClone = append(documentPathsFromClone, filepath.Join(rawPathInClone, entry.Name()))
								}
							}
						}

						// Step 4: Process documents from Gitea clone
						if len(documentPathsFromClone) > 0 {
							// Get markitdown client from integration
							var markitdownClient *clients.MarkItDownClient
							if adapter, ok := s.markitdownIntegration.(*markitdownClientAdapter); ok {
								markitdownClient = adapter.client
							}

							// Create document pipeline
							docPipeline := git.NewDocumentPipeline(
								markitdownClient,
								s.multiModalExtractor,
								giteaStorage,
								s.logger,
							)

							// Process documents from clone
							docNodes, docEdges, err := docPipeline.ProcessDocuments(ctx, documentPathsFromClone, storageConfig, req.ProjectID, req.SystemID)
							if err == nil {
								nodes = append(nodes, docNodes...)
								edges = append(edges, docEdges...)
								s.logger.Printf("Processed %d documents through markitdown/OCR from Gitea", len(documentPathsFromClone))
							} else {
								s.logger.Printf("Warning: document processing failed: %v", err)
							}

							// Create standardized structure
							if err := docPipeline.StandardizeDocumentStructure(ctx, storageConfig); err != nil {
								s.logger.Printf("Warning: failed to create standardized structure: %v", err)
							}
						}
					}
				}
			}
		}
	}

	// Process file system files with Gitea-first flow if configured
	if len(allFileSystemPaths) > 0 && req.GiteaStorage != nil && req.GiteaStorage.Enabled {
		giteaURL := req.GiteaStorage.GiteaURL
		if giteaURL == "" {
			giteaURL = os.Getenv("GITEA_URL")
		}
		giteaToken := req.GiteaStorage.GiteaToken
		if giteaToken == "" {
			giteaToken = os.Getenv("GITEA_TOKEN")
		}

		if giteaURL != "" && giteaToken != "" {
			giteaStorage := git.NewGiteaStorage(giteaURL, giteaToken, s.logger)
			storageConfig := git.StorageConfig{
				Owner:       req.GiteaStorage.Owner,
				RepoName:    req.GiteaStorage.RepoName,
				Branch:      req.GiteaStorage.Branch,
				BasePath:    req.GiteaStorage.BasePath,
				ProjectID:   req.ProjectID,
				SystemID:    req.SystemID,
				AutoCreate:  req.GiteaStorage.AutoCreate,
				Description: req.GiteaStorage.Description,
			}

			// Default repo name if not provided
			if storageConfig.RepoName == "" {
				storageConfig.RepoName = fmt.Sprintf("%s-extracted-code", req.ProjectID)
			}
			if storageConfig.Owner == "" {
				storageConfig.Owner = "extract-service"
			}

			// Step 1: Upload raw files to Gitea
			rawBasePath := "raw/filesystem/"
			giteaRepo, cloneURL, err := giteaStorage.UploadRawFiles(ctx, storageConfig, allFileSystemPaths, rawBasePath)
			if err != nil {
				s.logger.Printf("Warning: failed to upload raw files to Gitea: %v", err)
			} else {
				s.logger.Printf("Uploaded %d raw files to Gitea: %s", len(allFileSystemPaths), giteaRepo.HTMLURL)

				// Step 2: Clone from Gitea
				tempDir := filepath.Join(os.TempDir(), "extract-gitea-clone")
				if err := os.MkdirAll(tempDir, 0755); err != nil {
					s.logger.Printf("failed to create temp dir for Gitea clone: %v", err)
				} else {
					branch := storageConfig.Branch
					if branch == "" {
						branch = "main"
					}
					clonePath, err := giteaStorage.CloneFromGitea(ctx, cloneURL, branch, tempDir)
					if err != nil {
						s.logger.Printf("Warning: failed to clone from Gitea: %v", err)
					} else {
						defer func() {
							// Cleanup cloned repo
							if err := os.RemoveAll(clonePath); err != nil {
								s.logger.Printf("Warning: failed to cleanup Gitea clone: %v", err)
							}
						}()

						// Step 3: Extract files from Gitea clone
						filesystemExtractor := git.NewFileSystemExtractor()
						// Extract from the raw path in the cloned repo
						rawPathInClone := filepath.Join(clonePath, rawBasePath)
						var extractedFiles []git.ExtractedFile
						
						// Get list of files in the raw path
						if entries, err := os.ReadDir(rawPathInClone); err == nil {
							var filePathsInClone []string
							for _, entry := range entries {
								if !entry.IsDir() {
									filePathsInClone = append(filePathsInClone, filepath.Join(rawPathInClone, entry.Name()))
								}
							}
							if len(filePathsInClone) > 0 {
								extractedFiles, err = filesystemExtractor.ExtractFilesFromFileList(filePathsInClone)
								if err != nil {
									s.logger.Printf("Warning: failed to extract files from Gitea clone: %v", err)
								}
							}
						}

						// Step 4: Process files (schema extraction, etc.)
						// Process JSON tables for schema extraction
						for _, path := range req.JSONTables {
							// Find corresponding file in clone
							fileName := filepath.Base(path)
							cloneFilePath := filepath.Join(rawPathInClone, fileName)
							if _, err := os.Stat(cloneFilePath); err == nil {
								fileNodes, fileEdges, data, err := s.extractSchemaFromJSON(cloneFilePath)
								if err == nil {
									nodes = append(nodes, fileNodes...)
									edges = append(edges, fileEdges...)
									if s.tablePersistence != nil && data != nil {
										tableName := strings.TrimSuffix(filepath.Base(cloneFilePath), filepath.Ext(cloneFilePath))
										if err := s.tablePersistence.SaveTable(tableName, data); err != nil {
											s.logger.Printf("failed to save table %s: %v", tableName, err)
										}
									}
								}
							}
						}

						// Step 5: Update Gitea with processed results
						if len(extractedFiles) > 0 {
							processedBasePath := "filesystem/"
							if storageConfig.BasePath != "" {
								processedBasePath = storageConfig.BasePath
							}
							fsRepoMeta := &git.RepositoryMetadata{
								URL:    "filesystem",
								Branch: branch,
								Commit: "processed",
							}
							_, err := giteaStorage.StoreCode(ctx, storageConfig, extractedFiles, fsRepoMeta)
							if err != nil {
								s.logger.Printf("Warning: failed to store processed files in Gitea: %v", err)
							} else {
								s.logger.Printf("Successfully stored %d processed files in Gitea", len(extractedFiles))
								
								// Create Gitea repository node
								giteaRepoInfo, err := giteaStorage.GiteaClient().GetRepository(ctx, storageConfig.Owner, storageConfig.RepoName)
								if err == nil {
									giteaNode := giteaStorage.CreateRepositoryNode(giteaRepoInfo, storageConfig, fsRepoMeta)
									nodes = append(nodes, giteaNode)
									
									// Create file nodes for processed files
									fileStorage := git.NewFileStorage(s.logger)
									fsRepoID := fmt.Sprintf("filesystem:%s", req.ProjectID)
									fileNodes, fileEdges := fileStorage.CreateFileNodes(extractedFiles, fsRepoID, "filesystem", "processed", req.ProjectID, req.SystemID)
									nodes = append(nodes, fileNodes...)
									edges = append(edges, fileEdges...)
									
									// Link files to Gitea repository
									for _, fileNode := range fileNodes {
										edges = append(edges, graph.Edge{
											SourceID: fileNode.ID,
											TargetID: giteaNode.ID,
											Label:    "STORED_IN",
											Props: map[string]interface{}{
												"storage_type": "gitea",
												"source":       "filesystem",
											},
										})
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// Process Git repositories
	if len(req.GitRepositories) > 0 {
		tempDir := filepath.Join(os.TempDir(), "extract-git-repos")
		if err := os.MkdirAll(tempDir, 0755); err != nil {
			s.logger.Printf("failed to create temp dir for Git repos: %v", err)
		} else {
			gitPipeline := git.NewPipeline(tempDir, s.logger)
			
			for _, repoReq := range req.GitRepositories {
				if strings.TrimSpace(repoReq.URL) == "" {
					continue
				}

				// Convert request to pipeline GitRepository
				repo := pipeline.GitRepository{
					URL:         repoReq.URL,
					Type:        repoReq.Type,
					Branch:      repoReq.Branch,
					Tag:         repoReq.Tag,
					Commit:      repoReq.Commit,
					FilePatterns: repoReq.FilePatterns,
					Auth: pipeline.GitAuth{
						Type:     repoReq.Auth.Type,
						Token:    repoReq.Auth.Token,
						KeyPath:  repoReq.Auth.KeyPath,
						Username: repoReq.Auth.Username,
						Password: repoReq.Auth.Password,
					},
				}

				// Step 1: Clone source repository
				extractedFiles, repoMeta, err := gitPipeline.ProcessRepository(ctx, repo)
				if err != nil {
					s.logger.Printf("failed to process Git repository %s: %v", repoReq.URL, err)
					continue
				}

				// Create repository nodes for source repo
				repoNodes, repoEdges := gitPipeline.CreateRepositoryNodes(repoMeta, req.ProjectID, req.SystemID)
				nodes = append(nodes, repoNodes...)
				edges = append(edges, repoEdges...)
				repoID := repoNodes[0].ID

				// Step 2: Upload to Gitea if configured, then process from Gitea
				if req.GiteaStorage != nil && req.GiteaStorage.Enabled {
					giteaURL := req.GiteaStorage.GiteaURL
					if giteaURL == "" {
						giteaURL = os.Getenv("GITEA_URL")
					}
					giteaToken := req.GiteaStorage.GiteaToken
					if giteaToken == "" {
						giteaToken = os.Getenv("GITEA_TOKEN")
					}

					if giteaURL != "" && giteaToken != "" {
						giteaStorage := git.NewGiteaStorage(giteaURL, giteaToken, s.logger)
						storageConfig := git.StorageConfig{
							Owner:       req.GiteaStorage.Owner,
							RepoName:    req.GiteaStorage.RepoName,
							Branch:      req.GiteaStorage.Branch,
							BasePath:    req.GiteaStorage.BasePath,
							ProjectID:   req.ProjectID,
							SystemID:    req.SystemID,
							AutoCreate:  req.GiteaStorage.AutoCreate,
							Description: req.GiteaStorage.Description,
						}

						// Default repo name if not provided
						if storageConfig.RepoName == "" {
							storageConfig.RepoName = fmt.Sprintf("%s-extracted-code", req.ProjectID)
						}
						if storageConfig.Owner == "" {
							storageConfig.Owner = "extract-service"
						}

						// Upload raw files from source clone to Gitea
						// Store extracted files as raw files in Gitea
						rawBasePath := fmt.Sprintf("raw/git-repos/%s/", filepath.Base(repoMeta.URL))
						rawRepoMeta := &git.RepositoryMetadata{
							URL:    repoMeta.URL,
							Branch: repoMeta.Branch,
							Commit: repoMeta.Commit,
						}
						
						// Upload raw files to Gitea using StoreCode with raw base path
						rawStorageConfig := storageConfig
						rawStorageConfig.BasePath = rawBasePath
						_, err = giteaStorage.StoreCode(ctx, rawStorageConfig, extractedFiles, rawRepoMeta)
						if err != nil {
							s.logger.Printf("Warning: failed to upload raw files to Gitea: %v", err)
							// Fallback: continue with source clone
							fileStorage := git.NewFileStorage(s.logger)
							fileNodes, fileEdges := fileStorage.CreateFileNodes(extractedFiles, repoID, repoMeta.URL, repoMeta.Commit, req.ProjectID, req.SystemID)
							nodes = append(nodes, fileNodes...)
							edges = append(edges, fileEdges...)
						} else {
							// Get repository to get clone URL
							giteaRepoInfo, err := giteaStorage.GiteaClient().GetRepository(ctx, storageConfig.Owner, storageConfig.RepoName)
							if err != nil {
								s.logger.Printf("Warning: failed to get Gitea repository info: %v", err)
								// Fallback: continue with source clone
								fileStorage := git.NewFileStorage(s.logger)
								fileNodes, fileEdges := fileStorage.CreateFileNodes(extractedFiles, repoID, repoMeta.URL, repoMeta.Commit, req.ProjectID, req.SystemID)
								nodes = append(nodes, fileNodes...)
								edges = append(edges, fileEdges...)
							} else {
								cloneURL := giteaRepoInfo.CloneURL
								s.logger.Printf("Uploaded raw files to Gitea, cloning for processing...")
								
								// Step 3: Clone from Gitea
								giteaTempDir := filepath.Join(os.TempDir(), "extract-gitea-git-repos")
								if err := os.MkdirAll(giteaTempDir, 0755); err != nil {
									s.logger.Printf("failed to create temp dir for Gitea clone: %v", err)
								} else {
									branch := storageConfig.Branch
									if branch == "" {
										branch = "main"
									}
									giteaClonePath, err := giteaStorage.CloneFromGitea(ctx, cloneURL, branch, giteaTempDir)
									if err != nil {
										s.logger.Printf("Warning: failed to clone from Gitea: %v", err)
									} else {
										// Cleanup Gitea clone at end
										cleanupGiteaClone := func() {
											if err := os.RemoveAll(giteaClonePath); err != nil {
												s.logger.Printf("Warning: failed to cleanup Gitea clone: %v", err)
											}
										}
										defer cleanupGiteaClone()

										// Step 4: Extract from Gitea clone
										// Use CodeExtractor directly on the cloned path
										codeExtractor := git.NewCodeExtractor()
										rawPathInClone := filepath.Join(giteaClonePath, rawBasePath)
										giteaExtractedFiles, err := codeExtractor.ExtractFiles(rawPathInClone, repo.FilePatterns)
										if err != nil {
											s.logger.Printf("Warning: failed to extract from Gitea clone: %v", err)
											// Fallback to source extracted files
											giteaExtractedFiles = extractedFiles
										}

										// Step 5: Update Gitea with processed results
										processedBasePath := "extracted-code/"
										if storageConfig.BasePath != "" {
											processedBasePath = storageConfig.BasePath
										}
										processedStorageConfig := storageConfig
										processedStorageConfig.BasePath = processedBasePath
										_, err = giteaStorage.StoreCode(ctx, processedStorageConfig, giteaExtractedFiles, repoMeta)
										if err != nil {
											s.logger.Printf("Warning: failed to store processed code in Gitea: %v", err)
										} else {
											s.logger.Printf("Successfully stored processed code in Gitea")
										}

										// Create file nodes from Gitea extracted files
										fileStorage := git.NewFileStorage(s.logger)
										fileNodes, fileEdges := fileStorage.CreateFileNodes(giteaExtractedFiles, repoID, repoMeta.URL, repoMeta.Commit, req.ProjectID, req.SystemID)
										nodes = append(nodes, fileNodes...)
										edges = append(edges, fileEdges...)

										// Create Gitea repository node
										giteaNode := giteaStorage.CreateRepositoryNode(giteaRepoInfo, storageConfig, repoMeta)
										nodes = append(nodes, giteaNode)
										// Link source repo to Gitea repo
										edges = append(edges, graph.Edge{
											SourceID: repoID,
											TargetID: giteaNode.ID,
											Label:    "STORED_IN",
											Props: map[string]interface{}{
												"storage_type": "gitea",
											},
										})
										// Link files to Gitea repo
										for _, fileNode := range fileNodes {
											edges = append(edges, graph.Edge{
												SourceID: fileNode.ID,
												TargetID: giteaNode.ID,
												Label:    "STORED_IN",
												Props: map[string]interface{}{
													"storage_type": "gitea",
												},
											})
										}
									}
								}
							}
						}
					} else {
						// No Gitea storage - use source clone directly
						fileStorage := git.NewFileStorage(s.logger)
						fileNodes, fileEdges := fileStorage.CreateFileNodes(extractedFiles, repoID, repoMeta.URL, repoMeta.Commit, req.ProjectID, req.SystemID)
						nodes = append(nodes, fileNodes...)
						edges = append(edges, fileEdges...)
					}
				} else {
					// No Gitea storage - use source clone directly
					fileStorage := git.NewFileStorage(s.logger)
					fileNodes, fileEdges := fileStorage.CreateFileNodes(extractedFiles, repoID, repoMeta.URL, repoMeta.Commit, req.ProjectID, req.SystemID)
					nodes = append(nodes, fileNodes...)
					edges = append(edges, fileEdges...)
				}

				// Process extracted files based on their extensions
				for _, file := range extractedFiles {
					ext := strings.ToLower(filepath.Ext(file.Path))
					
					// Route to appropriate parser based on file extension
					switch ext {
					case ".hql", ".sql":
						// Process as DDL/SQL
						parsed, err := storage.ParseHiveDDL(ctx, file.Content)
						if err != nil {
							s.logger.Printf("failed to parse SQL from %s: %v", file.Path, err)
							continue
						}
						ddlNodes, ddlEdges := storage.DDLToGraph(parsed)
						nodes = append(nodes, ddlNodes...)
						edges = append(edges, ddlEdges...)
						
						// Link to repository
						if len(repoNodes) > 0 {
							edges = append(edges, graph.Edge{
								SourceID: repoNodes[0].ID,
								TargetID: ddlNodes[0].ID,
								Label:    "CONTAINS",
								Props: map[string]interface{}{
									"file_path": file.Path,
									"source":    "git",
								},
							})
						}
					case ".json":
						// Process as JSON table
						tmpFile, err := os.CreateTemp("", "git-json-*.json")
						if err != nil {
							s.logger.Printf("failed to create temp file: %v", err)
							continue
						}
						if _, err := tmpFile.WriteString(file.Content); err != nil {
							tmpFile.Close()
							os.Remove(tmpFile.Name())
							s.logger.Printf("failed to write temp file: %v", err)
							continue
						}
						tmpFile.Close()
						defer os.Remove(tmpFile.Name())

						fileNodes, fileEdges, _, err := s.extractSchemaFromJSON(tmpFile.Name())
						if err != nil {
							s.logger.Printf("failed to extract schema from %s: %v", file.Path, err)
							continue
						}
						nodes = append(nodes, fileNodes...)
						edges = append(edges, fileEdges...)
						
						// Link to repository
						if len(repoNodes) > 0 && len(fileNodes) > 0 {
							edges = append(edges, graph.Edge{
								SourceID: repoNodes[0].ID,
								TargetID: fileNodes[0].ID,
								Label:    "CONTAINS",
								Props: map[string]interface{}{
									"file_path": file.Path,
									"source":    "git",
								},
							})
						}
					}
				}

				s.logger.Printf("Processed Git repository %s: %d files extracted", repoReq.URL, len(extractedFiles))
			}
		}
	}

	for i, ddl := range req.HiveDDLs {
		ddl = strings.TrimSpace(ddl)
		if ddl == "" {
			continue
		}

		parsed, err := storage.ParseHiveDDL(ctx, ddl)
		if err != nil {
			s.logger.Printf("failed to parse hive ddl #%d: %v", i+1, err)
			continue
		}

		ddlNodes, ddlEdges := storage.DDLToGraph(parsed)
		nodes = append(nodes, ddlNodes...)
		edges = append(edges, ddlEdges...)
	}

	// Collect all Control-M jobs for Petri net conversion
	allControlMJobs := []integrations.ControlMJob{}
	controlMJobMap := make(map[string][]integrations.ControlMJob) // path -> jobs

	for _, path := range req.ControlMFiles {
		if strings.TrimSpace(path) == "" {
			continue
		}

		jobs, err := integrations.ParseControlMXML(path)
		if err != nil {
			s.logger.Printf("failed to parse control-m xml file %q: %v", path, err)
			continue
		}

		allControlMJobs = append(allControlMJobs, jobs...)
		controlMJobMap[path] = jobs

		for _, job := range jobs {
			jobID := fmt.Sprintf("control-m:%s", job.JobName)
			nodes = append(nodes, graph.Node{
				ID:    jobID,
				Type:  graph.NodeTypeControlMJob,
				Label: job.JobName,
				Props: job.Properties(),
			})

			if calendar := strings.TrimSpace(job.CalendarName); calendar != "" {
				calendarID := fmt.Sprintf("control-m-calendar:%s", calendar)
				nodes = append(nodes, graph.Node{
					ID:    calendarID,
					Type:  graph.NodeTypeControlMCalendar,
					Label: calendar,
				})
				edges = append(edges, graph.Edge{
					SourceID: calendarID,
					TargetID: jobID,
					Label:    "SCHEDULES",
				})
			}

			for _, inCond := range job.InConds {
				condID := fmt.Sprintf("control-m-cond:%s", inCond.Name)
				nodes = append(nodes, graph.Node{
					ID:    condID,
					Type:  graph.NodeTypeControlMCondition,
					Label: inCond.Name,
					Props: inCond.Properties(),
				})
				edges = append(edges, graph.Edge{
					SourceID: condID,
					TargetID: jobID,
					Label:    "BLOCKS",
					Props:    inCond.Properties(),
				})
			}

			for _, outCond := range job.OutConds {
				condID := fmt.Sprintf("control-m-cond:%s", outCond.Name)
				nodes = append(nodes, graph.Node{
					ID:    condID,
					Type:  graph.NodeTypeControlMCondition,
					Label: outCond.Name,
					Props: outCond.Properties(),
				})
				edges = append(edges, graph.Edge{
					SourceID: jobID,
					TargetID: condID,
					Label:    "RELEASES",
					Props:    outCond.Properties(),
				})
			}
		}
	}

	// Convert Control-M jobs to Petri net and add to knowledge graph
	// Also link Control-M jobs to SQL queries and tables
	if len(allControlMJobs) > 0 {
		// Map SQL queries to job names (if we can infer from context)
		// For now, we'll create a simple mapping
		sqlQueriesByJob := make(map[string][]string)
		sqlQueryIDsByJob := make(map[string][]string) // job name -> sql query IDs

		// Try to match SQL queries to jobs based on table names or patterns
		// This is a simplified approach - in practice, you'd have better job-to-SQL mapping
		for i, sql := range req.SqlQueries {
			// Generate SQL query ID for linking
			h := sha256.New()
			h.Write([]byte(sql))
			sqlQueryID := fmt.Sprintf("sql:%x", h.Sum(nil))

			// Use a simple heuristic: assign SQL to jobs in order
			if i < len(allControlMJobs) {
				jobName := allControlMJobs[i].JobName
				sqlQueriesByJob[jobName] = append(sqlQueriesByJob[jobName], sql)
				sqlQueryIDsByJob[jobName] = append(sqlQueryIDsByJob[jobName], sqlQueryID)
			}
		}

		// Link Control-M jobs to SQL queries and tables
		for _, job := range allControlMJobs {
			jobID := fmt.Sprintf("control-m:%s", job.JobName)

			// Link to SQL queries if available
			if sqlQueryIDs, ok := sqlQueryIDsByJob[job.JobName]; ok {
				for _, sqlQueryID := range sqlQueryIDs {
					edges = append(edges, graph.Edge{
						SourceID: jobID,
						TargetID: sqlQueryID,
						Label:    "EXECUTES",
					})
				}
			}

			// Try to extract table names from job command and link to tables
			// Look for table references in the command (e.g., table names in SQL commands)
			if job.Command != "" {
				// Simple pattern matching for table names in commands
				// This could be enhanced with better SQL parsing
				tablePattern := regexp.MustCompile(`(?i)\b(?:from|into|update|table)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)\b`)
				matches := tablePattern.FindAllStringSubmatch(job.Command, -1)
				for _, match := range matches {
					if len(match) > 1 {
						tableName := strings.TrimSpace(match[1])
						// Try to find the table node and create relationship
						edges = append(edges, graph.Edge{
							SourceID: jobID,
							TargetID: tableName,
							Label:    "PRODUCES",
						})
					}
				}
			}
		}

		// Convert Control-M to Petri net
		petriNetConverter := workflow.NewPetriNetConverter(s.logger)
		petriNet := petriNetConverter.ConvertControlMToPetriNet(allControlMJobs, sqlQueriesByJob)

		// Convert Petri net to graph nodes/edges
		petriNodes, petriEdges, petriRootID := petriNetConverter.PetriNetToGraphNodes(petriNet)
		nodes = append(nodes, petriNodes...)
		edges = append(edges, petriEdges...)

		// Link Petri net to root node
		if rootID != "" {
			edges = append(edges, graph.Edge{
				SourceID: rootID,
				TargetID: petriRootID,
				Label:    "HAS_PETRI_NET",
				Props: map[string]any{
					"petri_net_id": petriNet.ID,
					"name":         petriNet.Name,
				},
			})
		}

		s.logger.Printf("Converted Control-M to Petri net: %d places, %d transitions, %d arcs",
			len(petriNet.Places), len(petriNet.Transitions), len(petriNet.Arcs))
	}

	// Process view lineage data if available
	viewLineageMap := make(map[string]map[string]any)
	viewMetricsMap := make(map[string]map[string]any) // view name -> metrics
	if len(req.ViewLineage) > 0 {
		for _, viewEntry := range req.ViewLineage {
			if viewName, ok := viewEntry["view"].(string); ok {
				viewLineageMap[viewName] = viewEntry
				// Extract metrics if available
				if metrics, ok := viewEntry["metrics"].(map[string]interface{}); ok {
					viewMetricsMap[viewName] = make(map[string]any)
					if infoLoss, ok := metrics["information_loss"].(float64); ok {
						viewMetricsMap[viewName]["information_loss"] = infoLoss
					}
					if typeCoverage, ok := metrics["type_coverage"].(float64); ok {
						viewMetricsMap[viewName]["type_coverage"] = typeCoverage
					}
					if sourceSimilarity, ok := metrics["source_similarity"].(float64); ok {
						viewMetricsMap[viewName]["source_similarity"] = sourceSimilarity
					}
					if columnCount, ok := metrics["column_count"].(float64); ok {
						viewMetricsMap[viewName]["column_count"] = int(columnCount)
					}
					if columnsWithSources, ok := metrics["columns_with_sources"].(float64); ok {
						viewMetricsMap[viewName]["columns_with_sources"] = int(columnsWithSources)
					}
					if joinCount, ok := metrics["join_count"].(float64); ok {
						viewMetricsMap[viewName]["join_count"] = int(joinCount)
					}
				}
			}
		}
	}

	for _, sql := range req.SqlQueries {
		sql = strings.TrimSpace(sql)
		if sql == "" {
			continue
		}

		lineage, err := storage.ParseSQL(sql)
		if err != nil {
			s.logger.Printf("failed to parse sql query %q: %v", sql, err)
			continue
		}

		if len(lineage.SourceTables) > 0 && len(lineage.TargetTables) > 0 {
			for _, targetTable := range lineage.TargetTables {
				targetTable = strings.TrimSpace(targetTable)
				if targetTable == "" {
					continue
				}

				for _, sourceTable := range lineage.SourceTables {
					sourceTable = strings.TrimSpace(sourceTable)
					if sourceTable == "" {
						continue
					}

					edges = append(edges, graph.Edge{
						SourceID: sourceTable,
						TargetID: targetTable,
						Label:    "DATA_FLOW",
					})
				}
			}
		}

		if len(lineage.ColumnLineage) > 0 {
			defaultSource := ""
			if len(lineage.SourceTables) == 1 {
				defaultSource = strings.TrimSpace(lineage.SourceTables[0])
			}
			defaultTarget := ""
			if len(lineage.TargetTables) == 1 {
				defaultTarget = strings.TrimSpace(lineage.TargetTables[0])
			}

			// Generate SQL query ID for linking
			h := sha256.New()
			h.Write([]byte(sql))
			sqlQueryID := fmt.Sprintf("sql:%x", h.Sum(nil))

			// Track step order for multi-step pipelines
			stepOrder := 0
			seenTargetTables := make(map[string]int)
			for _, targetTable := range lineage.TargetTables {
				targetTable = strings.TrimSpace(targetTable)
				if targetTable != "" {
					if order, exists := seenTargetTables[targetTable]; exists {
						stepOrder = order
					} else {
						stepOrder = len(seenTargetTables)
						seenTargetTables[targetTable] = stepOrder
					}
				}
			}

			for _, cl := range lineage.ColumnLineage {
				sourceTable := strings.TrimSpace(cl.SourceTable)
				targetTable := strings.TrimSpace(cl.TargetTable)

				if sourceTable == "" {
					sourceTable = defaultSource
				}
				if targetTable == "" {
					targetTable = defaultTarget
				}

				sourceCol := strings.TrimSpace(cl.SourceColumn)
				targetCol := strings.TrimSpace(cl.TargetColumn)
				if sourceCol == "" || targetCol == "" {
					continue
				}

				var sourceColumnID string
				if sourceTable != "" {
					sourceColumnID = fmt.Sprintf("%s.%s", sourceTable, sourceCol)
				} else {
					sourceColumnID = sourceCol
				}

				var targetColumnID string
				if targetTable != "" {
					targetColumnID = fmt.Sprintf("%s.%s", targetTable, targetCol)
				} else {
					targetColumnID = targetCol
				}

				// Build source columns list
				sourceColumns := []string{sourceColumnID}
				if sourceTable != "" && sourceCol != "" {
					sourceColumns = []string{fmt.Sprintf("%s.%s", sourceTable, sourceCol)}
				}

				// Create source column node with properties
				sourceNodeProps := make(map[string]any)
				if sourceTable != "" {
					sourceNodeProps["table_name"] = sourceTable
				}
				nodes = append(nodes, graph.Node{
					ID:    sourceColumnID,
					Type:  graph.NodeTypeColumn,
					Label: sourceCol,
					Props: sourceNodeProps,
				})

				// Create target column node with transformation properties
				targetNodeProps := make(map[string]any)
				if targetTable != "" {
					targetNodeProps["table_name"] = targetTable
				}
				if cl.TransformationType != "" {
					targetNodeProps["transformation_type"] = cl.TransformationType
				}
				if cl.Function != "" {
					targetNodeProps["function"] = cl.Function
				}
				if cl.SQLExpression != "" {
					targetNodeProps["expression"] = cl.SQLExpression
				}
				if len(sourceColumns) > 0 {
					targetNodeProps["source_columns"] = sourceColumns
				}
				if sqlQueryID != "" {
					targetNodeProps["sql_query_id"] = sqlQueryID
				}
				if len(cl.AggregationKeys) > 0 {
					targetNodeProps["aggregation_keys"] = cl.AggregationKeys
				}

				// Enhance with view lineage data if available
				if viewEntry, ok := viewLineageMap[targetTable]; ok {
					if columns, ok := viewEntry["columns"].([]interface{}); ok {
						for _, colInterface := range columns {
							if colMap, ok := colInterface.(map[string]interface{}); ok {
								if colName, ok := colMap["name"].(string); ok && colName == targetCol {
									// Add view lineage transformation details
									if transType, ok := colMap["transformation_type"].(string); ok && transType != "" {
										targetNodeProps["transformation_type"] = transType
									}
									if expr, ok := colMap["expression"].(string); ok && expr != "" {
										targetNodeProps["expression"] = expr
									}
									if fn, ok := colMap["function"].(string); ok && fn != "" {
										targetNodeProps["function"] = fn
									}
									if inferredType, ok := colMap["inferred_type"].(string); ok && inferredType != "" {
										targetNodeProps["inferred_type"] = inferredType
									}
									// Link to view lineage entry
									if selectHash, ok := viewEntry["select_hash"].(string); ok {
										targetNodeProps["view_lineage_hash"] = selectHash
									}
									break
								}
							}
						}
					}
					// Add join information from view lineage
					if joins, ok := viewEntry["joins"].([]interface{}); ok && len(joins) > 0 {
						joinInfo := []map[string]any{}
						for _, joinInterface := range joins {
							if joinMap, ok := joinInterface.(map[string]interface{}); ok {
								joinInfo = append(joinInfo, joinMap)
							}
						}
						if len(joinInfo) > 0 {
							targetNodeProps["view_joins"] = joinInfo
						}
					}
					// Store view lineage metrics on the view/table node (will be added after column processing)
					// Metrics are stored via viewMetricsMap below
				}

				nodes = append(nodes, graph.Node{
					ID:    targetColumnID,
					Type:  graph.NodeTypeColumn,
					Label: targetCol,
					Props: targetNodeProps,
				})

				// Create DATA_FLOW edge with transformation properties
				edgeProps := make(map[string]any)
				if cl.TransformationType != "" {
					edgeProps["transformation_type"] = cl.TransformationType
				}
				if cl.SQLExpression != "" {
					edgeProps["sql_expression"] = cl.SQLExpression
				}
				if cl.Function != "" {
					edgeProps["function"] = cl.Function
				}
				if cl.JoinType != "" {
					edgeProps["join_type"] = cl.JoinType
				}
				if cl.JoinCondition != "" {
					edgeProps["join_condition"] = cl.JoinCondition
				}
				if cl.FilterCondition != "" {
					edgeProps["filter_condition"] = cl.FilterCondition
				}
				if sqlQueryID != "" {
					edgeProps["sql_query_id"] = sqlQueryID
				}
				if stepOrder > 0 {
					edgeProps["step_order"] = stepOrder
				}
				if targetTable != "" && targetTable != sourceTable {
					edgeProps["intermediate_table"] = targetTable
				}

				edges = append(edges, graph.Edge{
					SourceID: sourceColumnID,
					TargetID: targetColumnID,
					Label:    "DATA_FLOW",
					Props:    edgeProps,
				})
			}
		}

		embedding, err := embeddings.GenerateEmbedding(ctx, sql)
		if err != nil {
			s.logger.Printf("failed to generate embedding for sql query %q: %v", sql, err)
			continue
		}

		if s.vectorPersistence != nil {
			h := sha256.New()
			h.Write([]byte(sql))
			key := fmt.Sprintf("sql:%x", h.Sum(nil))

			// Build column lineage relationships for metadata
			columnRelationships := []map[string]any{}
			if len(lineage.ColumnLineage) > 0 {
				for _, cl := range lineage.ColumnLineage {
					sourceTable := strings.TrimSpace(cl.SourceTable)
					targetTable := strings.TrimSpace(cl.TargetTable)
					if sourceTable == "" && len(lineage.SourceTables) == 1 {
						sourceTable = strings.TrimSpace(lineage.SourceTables[0])
					}
					if targetTable == "" && len(lineage.TargetTables) == 1 {
						targetTable = strings.TrimSpace(lineage.TargetTables[0])
					}

					rel := map[string]any{
						"source_column": cl.SourceColumn,
						"target_column": cl.TargetColumn,
					}
					if sourceTable != "" {
						rel["source_table"] = sourceTable
					}
					if targetTable != "" {
						rel["target_table"] = targetTable
					}
					if cl.TransformationType != "" {
						rel["transformation_type"] = cl.TransformationType
					}
					if cl.Function != "" {
						rel["function"] = cl.Function
					}
					columnRelationships = append(columnRelationships, rel)
				}
			}

			// Create rich metadata for SQL query
			metadata := map[string]any{
				"artifact_type":        "sql-query",
				"artifact_id":          key,
				"label":                sql,
				"project_id":           req.ProjectID,
				"system_id":            req.SystemID,
				"created_at":           time.Now().UTC().Format(time.RFC3339Nano),
				"sql":                  sql,
				"source_tables":        lineage.SourceTables,
				"target_tables":        lineage.TargetTables,
				"column_lineage_count": len(lineage.ColumnLineage),
			}
			if len(columnRelationships) > 0 {
				metadata["column_relationships"] = columnRelationships
			}

			if err := s.vectorPersistence.SaveVector(key, embedding, metadata); err != nil {
				s.logger.Printf("failed to save vector for sql query %q: %v", sql, err)
			}
		}
	}

	// Apply view lineage metrics to table/view nodes
	for i := range nodes {
		if nodes[i].Type == "table" || nodes[i].Type == "view" {
			if metrics, ok := viewMetricsMap[nodes[i].ID]; ok {
				if nodes[i].Props == nil {
					nodes[i].Props = make(map[string]any)
				}
				for k, v := range metrics {
					nodes[i].Props[k] = v
				}
			}
			// Also check by label (view name might be in label)
			if metrics, ok := viewMetricsMap[nodes[i].Label]; ok {
				if nodes[i].Props == nil {
					nodes[i].Props = make(map[string]any)
				}
				for k, v := range metrics {
					nodes[i].Props[k] = v
				}
			}
		}
	}

	normResult := schema.NormalizeGraph(schema.NormalizationInput{
		Nodes:               nodes,
		Edges:               edges,
		ProjectID:           req.ProjectID,
		SystemID:            req.SystemID,
		InformationSystemID: req.InformationSystemID,
		Catalog:             s.catalog,
	})
	nodes = normResult.Nodes
	edges = normResult.Edges
	rootID = normResult.RootNodeID

	for _, warning := range normResult.Warnings {
		s.logger.Printf("normalization warning: %s", warning)
	}

	// Validate graph before persistence
	validationWarnings := validateGraph(nodes, edges)
	for _, warning := range validationWarnings {
		s.logger.Printf("validation warning: %s", warning)
	}

	// Associate domains with extracted nodes and edges
	if s.domainDetector != nil {
		s.domainDetector.AssociateDomainsWithNodes(nodes)

		// Create node map for edge association
		nodeMap := make(map[string]*graph.Node)
		for i := range nodes {
			nodeMap[nodes[i].ID] = &nodes[i]
		}
		s.domainDetector.AssociateDomainsWithEdges(edges, nodeMap)

		// Associate SQL queries with domains
		sqlToAgentID := s.domainDetector.AssociateDomainsWithSQL(req.SqlQueries)
		if len(sqlToAgentID) > 0 {
			s.logger.Printf("✅ Associated %d SQL queries with domains", len(sqlToAgentID))
		}
	}

	// Extract raw column types
	rawColumnDtypes := make([]string, 0)
	for _, node := range nodes {
		if node.Type != "column" || node.Props == nil {
			continue
		}
		if dtype, ok := node.Props["type"].(string); ok && dtype != "" {
			rawColumnDtypes = append(rawColumnDtypes, dtype)
		}
	}

	// Normalize to canonical types for quality metrics calculation
	// This ensures actual distribution keys match ideal distribution keys
	normalizedDtypes := make([]string, 0, len(rawColumnDtypes))
	typeNormalizationStats := make(map[string]int)
	for _, rawType := range rawColumnDtypes {
		canonicalType := utils.NormalizeToCanonicalType(rawType)
		normalizedDtypes = append(normalizedDtypes, canonicalType)
		typeNormalizationStats[canonicalType]++
	}

	// Calculate entropy on normalized types (improves diversity perception)
	metadataEntropy := utils.CalculateEntropy(normalizedDtypes)

	// Calculate distribution on normalized canonical types
	// This aligns with ideal distribution which uses canonical types
	actualDistribution := make(map[string]float64)
	totalColumns := float64(len(normalizedDtypes))

	// Initialize all canonical types to ensure they're represented
	canonicalTypes := []string{"string", "number", "boolean", "date", "array", "object"}
	for _, ct := range canonicalTypes {
		actualDistribution[ct] = 0.0
	}

	// Count normalized types
	for _, dtype := range normalizedDtypes {
		actualDistribution[dtype]++
	}

	// Normalize to probabilities
	if totalColumns > 0 {
		for dtype := range actualDistribution {
			actualDistribution[dtype] = actualDistribution[dtype] / totalColumns
		}
	}

	// Log type normalization statistics for debugging
	if s.logger != nil && len(rawColumnDtypes) > 0 {
		statsParts := make([]string, 0, len(typeNormalizationStats))
		for canonical, count := range typeNormalizationStats {
			statsParts = append(statsParts, fmt.Sprintf("%d→%s", count, canonical))
		}
		s.logger.Printf("Type normalization: %d columns normalized to canonical types: %s",
			len(rawColumnDtypes), strings.Join(statsParts, ", "))
	}

	idealDistribution := req.IdealDistribution
	if idealDistribution == nil {
		idealDistribution = map[string]float64{
			"string":  defaultStringRatio,
			"number":  defaultNumberRatio,
			"boolean": defaultBooleanRatio,
			"date":    defaultDateRatio,
			"array":   defaultArrayRatio,
			"object":  defaultObjectRatio,
		}
	}
	klDivergence := utils.CalculateKLDivergence(actualDistribution, idealDistribution)

	// Interpret metrics and determine actionable insights
	thresholds := processing.DefaultMetricsThresholds()
	interpretation := processing.InterpretMetrics(
		metadataEntropy,
		klDivergence,
		len(normalizedDtypes),
		actualDistribution,
		idealDistribution,
		thresholds,
		s.logger,
	)

	// Take action based on interpretation
	// Allow bypass for projects that have known data patterns (configurable via project config)
	// Removed hard-coded SGMI check - now configurable per project
	allowBypass := false // Can be set via project config in the future
	
	if interpretation.ShouldReject && !allowBypass {
		s.logger.Printf("ERROR: Rejecting graph processing due to critical data quality issues (quality=%s, score=%.2f)",
			interpretation.QualityLevel, interpretation.QualityScore)
		handlers.WriteJSON(w, http.StatusUnprocessableEntity, map[string]any{
			"error":           "Graph processing rejected due to data quality issues",
			"quality_level":   interpretation.QualityLevel,
			"quality_score":   interpretation.QualityScore,
			"issues":          interpretation.Issues,
			"recommendations": interpretation.Recommendations,
			"metrics": map[string]any{
				"metadata_entropy": metadataEntropy,
				"kl_divergence":    klDivergence,
				"column_count":     len(normalizedDtypes),
			},
		})
		return
	}
	
	if interpretation.ShouldReject && allowBypass {
		s.logger.Printf("WARNING: Bypassing quality check for project %s (quality=%s, score=%.2f) - proceeding with processing",
			req.ProjectID, interpretation.QualityLevel, interpretation.QualityScore)
		// Continue processing despite quality issues
	}

	// Log warnings if needed
	if interpretation.ShouldWarn {
		for _, issue := range interpretation.Issues {
			s.logger.Printf("WARNING: %s", issue)
		}
	}

	// Get processing strategy flags (for future use)
	processingFlags := processing.GetProcessingFlags(interpretation)
	if processingFlags["skip_processing"] {
		s.logger.Printf("WARNING: Data quality suggests simplified processing")
	}

	// Store metrics in root node for graph analysis and Glean export
	// This ensures metrics are available in all persistence layers (Neo4j, Glean, etc.)
	if rootID != "" {
		for i := range nodes {
			if nodes[i].ID == rootID {
				if nodes[i].Props == nil {
					nodes[i].Props = make(map[string]any)
				}
				nodes[i].Props["metadata_entropy"] = metadataEntropy
				nodes[i].Props["kl_divergence"] = klDivergence
				nodes[i].Props["actual_distribution"] = actualDistribution
				nodes[i].Props["ideal_distribution"] = idealDistribution
				nodes[i].Props["column_count"] = len(normalizedDtypes)
				// Store metrics timestamp for tracking over time
				nodes[i].Props["metrics_calculated_at"] = time.Now().UTC().Format(time.RFC3339Nano)
				break
			}
		}
	}

	s.replicateSchema(ctx, nodes, edges)

	// Improvement 3: Add automatic consistency validation
	if projectID := req.ProjectID; projectID != "" {
		consistencyStart := time.Now()
		consistencyResult := schema.ValidateConsistency(ctx, projectID, s.logger)
		consistencyDuration := time.Since(consistencyStart)

		// Record consistency metrics
		if s.metricsCollector != nil {
			s.metricsCollector.RecordConsistency(consistencyResult, consistencyDuration)
		}

		if !consistencyResult.Consistent {
			s.logger.Printf("WARNING: Consistency validation found %d issues after replication", len(consistencyResult.Issues))
			for _, issue := range consistencyResult.Issues {
				s.logger.Printf("  [%s] %s: %s", issue.Severity, issue.Type, issue.Message)
			}
		} else {
			s.logger.Printf("Consistency validation passed: nodes variance=%d, edges variance=%d",
				consistencyResult.Metrics.NodeVariance, consistencyResult.Metrics.EdgeVariance)
		}
	}

	// Export Petri net to catalog if available
	if len(allControlMJobs) > 0 {
		petriNetConverter := workflow.NewPetriNetConverter(s.logger)
		sqlQueriesByJob := make(map[string][]string)
		for i, sql := range req.SqlQueries {
			if i < len(allControlMJobs) {
				jobName := allControlMJobs[i].JobName
				sqlQueriesByJob[jobName] = append(sqlQueriesByJob[jobName], sql)
			}
		}
		petriNet := petriNetConverter.ConvertControlMToPetriNet(allControlMJobs, sqlQueriesByJob)

		if s.catalog != nil {
			catalogEntry := petriNetConverter.ExportPetriNetToCatalog(petriNet)
			if s.catalog.PetriNets == nil {
				s.catalog.PetriNets = make(map[string]any)
			}
			s.catalog.PetriNets[petriNet.ID] = catalogEntry

			// Save catalog to persist Petri net
			if err := s.catalog.Save(); err != nil {
				s.logger.Printf("failed to save Petri net to catalog: %v", err)
			} else {
				s.logger.Printf("Saved Petri net '%s' to catalog", petriNet.ID)
			}
		}
	}

	// Register extracted data elements in catalog service
	if s.catalogClient != nil && len(nodes) > 0 {
		// Convert nodes to data elements and register in bulk
		dataElements := make([]clients.DataElementRequest, 0, len(nodes))
		for _, node := range nodes {
			// Only register meaningful nodes (skip root, project, system nodes)
			if node.Type == "root" || node.Type == "project" || node.Type == "system" || node.Type == "information-system" {
				continue
			}
			// Use AI enrichment if enabled, otherwise use basic conversion
			var element clients.DataElementRequest
			if s.catalogClient != nil {
				element = s.catalogClient.ConvertNodeToDataElementWithAI(ctx, node, req.ProjectID, req.SystemID)
			} else {
				element = clients.ConvertNodeToDataElement(node, req.ProjectID, req.SystemID)
			}
			dataElements = append(dataElements, element)
		}

		if len(dataElements) > 0 {
			// Register in background to avoid blocking extraction
			go func() {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()
				if err := s.catalogClient.RegisterDataElementsBulk(ctx, dataElements); err != nil {
					s.logger.Printf("Warning: Failed to register data elements in catalog service: %v", err)
				} else {
					s.logger.Printf("✅ Registered %d data elements in catalog service", len(dataElements))
				}
			}()
		}
	}

	// AI Enhancement (optional, non-blocking)
	aiEnabled := req.AIEnabled
	if !aiEnabled {
		// Check environment variable as fallback
		aiEnabled = strings.ToLower(os.Getenv("AI_ENABLED")) == "true"
	}
	
	if aiEnabled && len(nodes) > 0 {
		localAIURL := os.Getenv("LOCALAI_URL")
		if localAIURL == "" {
			localAIURL = "http://localai:8080"
		}
		
		aiModel := req.AIModel
		if aiModel == "" {
			aiModel = os.Getenv("AI_MODEL")
			if aiModel == "" {
				aiModel = "auto"
			}
		}
		
		cwmClient := ai.NewCWMClient(localAIURL)
		
		// Enhance nodes with semantic analysis (non-blocking, log errors)
		go func() {
			enhancedCount := 0
			for i := range nodes {
				// Only enhance code-related nodes
				if nodes[i].Type == graph.NodeTypeTable || 
				   nodes[i].Type == graph.NodeTypeView ||
				   nodes[i].Type == graph.NodeTypeFunction ||
				   nodes[i].Type == "Repository" {
					
					// Extract code content if available
					codeContent := ""
					if content, ok := nodes[i].Props["content"].(string); ok {
						codeContent = content
					} else if sql, ok := nodes[i].Props["sql_query"].(string); ok {
						codeContent = sql
					}
					
					if codeContent != "" {
						codeType := "SQL"
						if nodes[i].Type == "Repository" {
							codeType = "Repository"
						}
						
						analysis, err := cwmClient.AnalyzeCodeSemantics(context.Background(), codeContent, codeType)
						if err == nil && analysis != nil {
							// Add semantic metadata to node properties
							if nodes[i].Props == nil {
								nodes[i].Props = make(map[string]interface{})
							}
							nodes[i].Props["ai_intent"] = analysis.Intent
							nodes[i].Props["ai_transformations"] = analysis.Transformations
							nodes[i].Props["ai_dependencies"] = analysis.Dependencies
							enhancedCount++
						}
					}
				}
			}
			
			if enhancedCount > 0 {
				s.logger.Printf("AI enhancement: enhanced %d nodes with semantic analysis", enhancedCount)
			}
		}()
	}

	// Save graph to persistence layers (Neo4j, Glean, etc.)
	// Information theory metrics are included:
	// 1. In root node properties (accessible in Neo4j queries)
	// 2. In Glean export manifest (via glean_persistence.go)
	if s.graphPersistence != nil {
		// Improvement 2: Add retry logic for Neo4j operations
		if err := utils.RetryNeo4jOperation(ctx, func() error {
			return s.graphPersistence.SaveGraph(nodes, edges)
		}, s.logger); err != nil {
			s.logger.Printf("failed to save graph to Neo4j after retries: %v", err)
		}

		// Populate execution tracking, data quality, and performance metrics in Neo4j
		if s.neo4jPersistence != nil {
			s.populateExecutionTracking(ctx, &req, allControlMJobs, nodes, edges)
			s.populateDataQualityMetrics(ctx, interpretation, nodes)
			s.populatePerformanceMetrics(ctx, &req, nodes)
		}

		// Phase 10: Learn terminology from this extraction run (incremental learning)
		if terminologyLearner := embeddings.GetGlobalTerminologyLearner(); terminologyLearner != nil {
			if err := terminologyLearner.LearnFromExtraction(r.Context(), nodes, edges); err != nil {
				s.logger.Printf("Warning: Failed to learn terminology from extraction: %v", err)
				// Non-fatal: continue even if terminology learning fails
			}
		}
	}

	// Real-time Glean synchronization (if enabled)
	// This automatically ingests batches into Glean Catalog as they are created
	if s.realTimeGleanExporter != nil {
		// Extract projectID and systemID from request or nodes
		exportProjectID := req.ProjectID
		exportSystemID := req.SystemID

		// Fallback to extracting from nodes if not in request
		if exportProjectID == "" {
			for _, node := range nodes {
				if node.Type == "project" || node.Type == "information-system" {
					exportProjectID = node.Label
					break
				}
			}
		}
		if exportSystemID == "" {
			for _, node := range nodes {
				if node.Type == "system" {
					exportSystemID = node.Label
					break
				}
			}
		}

		// Export to Glean in real-time (non-blocking async queue)
		if err := s.realTimeGleanExporter.ExportGraph(ctx, nodes, edges, exportProjectID, exportSystemID); err != nil {
			s.logger.Printf("real-time Glean export failed (non-fatal): %v", err)
		}
	}

	if s.flight != nil {
		s.flight.UpdateGraph(nodes, edges)
	}

	// Advanced extraction (Priority 6.1 enhancement)
	// Extract table sequences, parameters, hardcoded lists, table classifications, and testing endpoints
	// Collect Control-M file contents for advanced extraction
	controlMContents := []string{}
	for _, path := range req.ControlMFiles {
		if strings.TrimSpace(path) == "" {
			continue
		}
		content, err := os.ReadFile(path)
		if err != nil {
			s.logger.Printf("failed to read Control-M file %q for advanced extraction: %v", path, err)
			continue
		}
		controlMContents = append(controlMContents, string(content))
	}

	// Collect JSON table data for advanced extraction (read raw JSON)
	jsonTableData := []map[string]any{}
	for _, path := range req.JSONTables {
		if strings.TrimSpace(path) == "" {
			continue
		}
		// Read JSON file content
		content, err := os.ReadFile(path)
		if err != nil {
			s.logger.Printf("failed to read JSON file %q for advanced extraction: %v", path, err)
			continue
		}
		var tableData map[string]any
		if err := json.Unmarshal(content, &tableData); err == nil {
			jsonTableData = append(jsonTableData, tableData)
		}
	}

	// Perform advanced extraction
	advancedExtractor := extraction.NewAdvancedExtractor(s.logger)

	// Phase 10: Wire terminology learner to advanced extractor
	if terminologyLearner := embeddings.GetGlobalTerminologyLearner(); terminologyLearner != nil {
		advancedExtractor.SetTerminologyLearner(terminologyLearner)
	}

	advancedResult := advancedExtractor.ExtractAdvanced(
		req.SqlQueries,
		controlMContents,
		req.HiveDDLs,
		jsonTableData,
	)

	// Add advanced extraction results to graph nodes/edges
	// This enhances the knowledge graph with advanced metadata
	if advancedResult != nil {
		// Collect training data for classifications (Phase 4)
		if s.trainingDataCollector != nil && os.Getenv("COLLECT_TRAINING_DATA") == "true" {
			for _, classification := range advancedResult.TableClassifications {
				// Extract columns for this table from nodes
				var columns []map[string]any
				for _, node := range nodes {
					if node.Type == "column" && node.Props != nil {
						if tableName, ok := node.Props["table_name"].(string); ok && tableName == classification.TableName {
							colDef := map[string]any{
								"name": node.Label,
							}
							if colType, ok := node.Props["type"].(string); ok {
								colDef["type"] = colType
							}
							columns = append(columns, colDef)
						}
					}
				}

				// Collect training data
				if err := s.trainingDataCollector.CollectTableClassification(
					classification.TableName,
					columns,
					classification.Classification,
					classification.Confidence,
					"", // Context would be from DDL if available
				); err != nil {
					s.logger.Printf("failed to collect training data for %s: %v", classification.TableName, err)
				}
			}
		}

		// Add table classifications as node properties
		for _, classification := range advancedResult.TableClassifications {
			for i := range nodes {
				if nodes[i].Type == "table" && nodes[i].Label == classification.TableName {
					if nodes[i].Props == nil {
						nodes[i].Props = make(map[string]any)
					}
					nodes[i].Props["table_classification"] = classification.Classification
					nodes[i].Props["classification_confidence"] = classification.Confidence
					nodes[i].Props["classification_evidence"] = classification.Evidence
					break
				}
			}
		}

		// Add table process sequences as edges or metadata
		for _, sequence := range advancedResult.TableProcessSequences {
			// Create edges between tables in sequence
			for i := 0; i < len(sequence.Tables)-1; i++ {
				sourceTable := sequence.Tables[i]
				targetTable := sequence.Tables[i+1]

				// Find or create edge
				edgeFound := false
				for j := range edges {
					if edges[j].SourceID == sourceTable && edges[j].TargetID == targetTable &&
						edges[j].Label == "PROCESSES_BEFORE" {
						edgeFound = true
						if edges[j].Props == nil {
							edges[j].Props = make(map[string]any)
						}
						edges[j].Props["sequence_id"] = sequence.SequenceID
						edges[j].Props["sequence_type"] = sequence.SequenceType
						edges[j].Props["sequence_order"] = sequence.Order
						break
					}
				}

				if !edgeFound {
					edges = append(edges, graph.Edge{
						SourceID: sourceTable,
						TargetID: targetTable,
						Label:    "PROCESSES_BEFORE",
						Props: map[string]any{
							"sequence_id":    sequence.SequenceID,
							"sequence_type":  sequence.SequenceType,
							"sequence_order": sequence.Order,
						},
					})
				}
			}
		}

		// Store advanced extraction results in root node for reference
		if rootID != "" {
			for i := range nodes {
				if nodes[i].ID == rootID {
					if nodes[i].Props == nil {
						nodes[i].Props = make(map[string]any)
					}
					nodes[i].Props["advanced_extraction"] = map[string]any{
						"table_process_sequences": len(advancedResult.TableProcessSequences),
						"code_parameters":         len(advancedResult.CodeParameters),
						"hardcoded_lists":         len(advancedResult.HardcodedLists),
						"table_classifications":   len(advancedResult.TableClassifications),
						"testing_endpoints":       len(advancedResult.TestingEndpoints),
					}
					break
				}
			}
		}
	}

	// Generate embeddings for all ETL artifacts (Phase 1: Enhanced Embedding Generation)
	// This enhances the knowledge graph with semantic search capabilities
	// Phase 3: Uses batch processing and caching for optimization
	if s.vectorPersistence != nil {
		// Collect all table nodes for batch processing
		tableNodes := []graph.Node{}
		for _, node := range nodes {
			if node.Type == "table" {
				tableNodes = append(tableNodes, node)
			}
		}

		// Use batch processing if available and multiple tables
		if s.batchEmbeddingGen != nil && len(tableNodes) > 1 {
			s.logger.Printf("Using batch processing for %d tables...", len(tableNodes))
			batchResults, err := s.batchEmbeddingGen.GenerateBatchTableEmbeddings(ctx, tableNodes)
			if err != nil {
				s.logger.Printf("batch embedding generation failed: %v, continuing without batch", err)
			}

			// Process batch results
			for _, node := range tableNodes {
				result, exists := batchResults[node.ID]
				if !exists || result.Error != nil {
					if result.Error != nil {
						s.logger.Printf("failed to generate embedding for table %q: %v", node.Label, result.Error)
					}
					continue
				}

				relationalEmbedding := result.Relational
				semanticEmbedding := result.Semantic

				// Store RelationalTransformer embedding (primary)
				metadata := map[string]any{
					"artifact_type":  "table",
					"artifact_id":    node.ID,
					"label":          node.Label,
					"properties":     node.Props,
					"project_id":     req.ProjectID,
					"system_id":      req.SystemID,
					"graph_node_id":  node.ID,
					"created_at":     time.Now().UTC().Format(time.RFC3339Nano),
					"table_name":     node.Label,
					"embedding_type": "relational_transformer",
				}

				key := fmt.Sprintf("table:%s", node.ID)
				if err := s.vectorPersistence.SaveVector(key, relationalEmbedding, metadata); err != nil {
					s.logger.Printf("failed to save table embedding %q: %v", node.Label, err)
				}

				// Store sap-rpt-1-oss semantic embedding (if available)
				if len(semanticEmbedding) > 0 {
					semanticMetadata := map[string]any{
						"artifact_type":  "table",
						"artifact_id":    node.ID,
						"label":          node.Label,
						"properties":     node.Props,
						"project_id":     req.ProjectID,
						"system_id":      req.SystemID,
						"graph_node_id":  node.ID,
						"created_at":     time.Now().UTC().Format(time.RFC3339Nano),
						"table_name":     node.Label,
						"embedding_type": "sap_rpt_semantic",
					}

					semanticKey := fmt.Sprintf("table_semantic:%s", node.ID)
					if err := s.vectorPersistence.SaveVector(semanticKey, semanticEmbedding, semanticMetadata); err != nil {
						s.logger.Printf("failed to save semantic table embedding %q: %v", node.Label, err)
					}
				}
			}
		}

		// Generate embeddings for columns (with batch processing and caching)
		columnNodes := []graph.Node{}
		for _, node := range nodes {
			if node.Type == "column" {
				columnNodes = append(columnNodes, node)
			}
		}

		// Use batch processing if available and multiple columns
		if s.batchEmbeddingGen != nil && len(columnNodes) > 1 {
			s.logger.Printf("Using batch processing for %d columns...", len(columnNodes))
			batchResults, err := s.batchEmbeddingGen.GenerateBatchColumnEmbeddings(ctx, columnNodes)
			if err != nil {
				s.logger.Printf("batch column embedding generation failed: %v, continuing without batch", err)
			}

			// Process batch results
			for _, node := range columnNodes {
				result, exists := batchResults[node.ID]
				if !exists || result.Error != nil {
					if result.Error != nil {
						s.logger.Printf("failed to generate embedding for column %q: %v", node.Label, result.Error)
					}
					continue
				}

				relationalEmbedding := result.Embedding

				// Create rich metadata
				metadata := map[string]any{
					"artifact_type":  "column",
					"artifact_id":    node.ID,
					"label":          node.Label,
					"properties":     node.Props,
					"project_id":     req.ProjectID,
					"system_id":      req.SystemID,
					"graph_node_id":  node.ID,
					"created_at":     time.Now().UTC().Format(time.RFC3339Nano),
					"column_name":    node.Label,
					"embedding_type": "relational_transformer",
				}

				key := fmt.Sprintf("column:%s", node.ID)
				if err := s.vectorPersistence.SaveVector(key, relationalEmbedding, metadata); err != nil {
					s.logger.Printf("failed to save column embedding %q: %v", node.Label, err)
				}

				// Try to generate sap-rpt-1-oss semantic embedding if enabled
				if os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true" {
					columnType := "string"
					if node.Props != nil {
						if t, ok := node.Props["type"].(string); ok && t != "" {
							columnType = t
						}
					}

					cmdSemantic := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed_sap_rpt.py",
						"--artifact-type", "column",
						"--column-name", node.Label,
						"--column-type", columnType,
					)

					outputSemantic, err := cmdSemantic.Output()
					if err == nil {
						var semanticEmbedding []float32
						if err := json.Unmarshal(outputSemantic, &semanticEmbedding); err == nil && len(semanticEmbedding) > 0 {
							semanticMetadata := map[string]any{
								"artifact_type":  "column",
								"artifact_id":    node.ID,
								"label":          node.Label,
								"properties":     node.Props,
								"project_id":     req.ProjectID,
								"system_id":      req.SystemID,
								"graph_node_id":  node.ID,
								"created_at":     time.Now().UTC().Format(time.RFC3339Nano),
								"column_name":    node.Label,
								"embedding_type": "sap_rpt_semantic",
							}

							semanticKey := fmt.Sprintf("column_semantic:%s", node.ID)
							if err := s.vectorPersistence.SaveVector(semanticKey, semanticEmbedding, semanticMetadata); err != nil {
								s.logger.Printf("failed to save semantic column embedding %q: %v", node.Label, err)
							}
						}
					}
				}
			}
		}

		// Generate embeddings for Control-M jobs
		for _, job := range allControlMJobs {
			embedding, err := embeddings.GenerateJobEmbedding(ctx, job)
			if err != nil {
				s.logger.Printf("failed to generate embedding for Control-M job %q: %v", job.JobName, err)
				continue
			}

			// Create rich metadata
			metadata := map[string]any{
				"artifact_type": "control-m-job",
				"artifact_id":   fmt.Sprintf("control-m:%s", job.JobName),
				"label":         job.JobName,
				"properties":    job.Properties(),
				"project_id":    req.ProjectID,
				"system_id":     req.SystemID,
				"graph_node_id": fmt.Sprintf("control-m:%s", job.JobName),
				"created_at":    time.Now().UTC().Format(time.RFC3339Nano),
				"job_name":      job.JobName,
				"command":       job.Command,
			}

			key := fmt.Sprintf("job:%s", job.JobName)
			if err := s.vectorPersistence.SaveVector(key, embedding, metadata); err != nil {
				s.logger.Printf("failed to save job embedding %q: %v", job.JobName, err)
			}
		}

		// Generate embeddings for process sequences
		if advancedResult != nil {
			for _, seq := range advancedResult.TableProcessSequences {
				embedding, err := embeddings.GenerateSequenceEmbedding(ctx, seq)
				if err != nil {
					s.logger.Printf("failed to generate embedding for sequence %q: %v", seq.SequenceID, err)
					continue
				}

				// Create rich metadata
				metadata := map[string]any{
					"artifact_type": "process-sequence",
					"artifact_id":   seq.SequenceID,
					"label":         seq.SequenceID,
					"project_id":    req.ProjectID,
					"system_id":     req.SystemID,
					"created_at":    time.Now().UTC().Format(time.RFC3339Nano),
					"sequence_id":   seq.SequenceID,
					"tables":        seq.Tables,
					"source_type":   seq.SourceType,
					"source_file":   seq.SourceFile,
					"sequence_type": seq.SequenceType,
					"order":         seq.Order,
				}

				key := fmt.Sprintf("sequence:%s", seq.SequenceID)
				if err := s.vectorPersistence.SaveVector(key, embedding, metadata); err != nil {
					s.logger.Printf("failed to save sequence embedding %q: %v", seq.SequenceID, err)
				}
			}
		}

		// Generate embeddings for Signavio processes
		if signavioMetadata.ProcessCount > 0 {
			for _, proc := range signavioMetadata.Processes {
				embedding, err := embeddings.GenerateSignavioProcessEmbedding(ctx, proc)
				if err != nil {
					s.logger.Printf("failed to generate embedding for Signavio process %q: %v", proc.Name, err)
					continue
				}

				metadata := map[string]any{
					"artifact_type":  "signavio-process",
					"artifact_id":    proc.ID,
					"label":          proc.Name,
					"project_id":     req.ProjectID,
					"system_id":      req.SystemID,
					"process_id":     proc.ID,
					"process_name":   proc.Name,
					"element_count":  proc.ElementCount,
					"element_types":  proc.ElementTypes,
					"source_file":    proc.SourceFile,
					"created_at":     time.Now().UTC().Format(time.RFC3339Nano),
					"embedding_type": "semantic_process",
				}

				key := fmt.Sprintf("signavio-process:%s", proc.ID)
				if err := s.vectorPersistence.SaveVector(key, embedding, metadata); err != nil {
					s.logger.Printf("failed to save Signavio process embedding %q: %v", proc.Name, err)
				}
			}
		}

		// Generate embedding for Petri net (if available)
		if len(allControlMJobs) > 0 {
			petriNetConverter := workflow.NewPetriNetConverter(s.logger)
			sqlQueriesByJob := make(map[string][]string)
			for i, sql := range req.SqlQueries {
				if i < len(allControlMJobs) {
					jobName := allControlMJobs[i].JobName
					sqlQueriesByJob[jobName] = append(sqlQueriesByJob[jobName], sql)
				}
			}
			petriNet := petriNetConverter.ConvertControlMToPetriNet(allControlMJobs, sqlQueriesByJob)

			embedding, err := embeddings.GeneratePetriNetEmbedding(ctx, petriNet)
			if err != nil {
				s.logger.Printf("failed to generate embedding for Petri net: %v", err)
			} else {
				// Create rich metadata
				metadata := map[string]any{
					"artifact_type":     "petri-net",
					"artifact_id":       petriNet.ID,
					"label":             petriNet.Name,
					"project_id":        req.ProjectID,
					"system_id":         req.SystemID,
					"created_at":        time.Now().UTC().Format(time.RFC3339Nano),
					"petri_net_id":      petriNet.ID,
					"name":              petriNet.Name,
					"places_count":      len(petriNet.Places),
					"transitions_count": len(petriNet.Transitions),
					"arcs_count":        len(petriNet.Arcs),
					"metadata":          petriNet.Metadata,
				}

				key := fmt.Sprintf("petri_net:%s", petriNet.ID)
				if err := s.vectorPersistence.SaveVector(key, embedding, metadata); err != nil {
					s.logger.Printf("failed to save Petri net embedding: %v", err)
				}
			}
		}
	}

	// DeepAgents analysis (enabled by default, 10/10 integration)
	// Always attempt analysis if client is enabled (non-fatal if service unavailable)
	var deepAgentsAnalysis *clients.AnalyzeGraphResponse
	if s.deepAgentsClient != nil && s.deepAgentsClient.IsEnabled() {
		graphSummary := clients.FormatGraphSummary(nodes, edges, map[string]any{
			"score":  interpretation.QualityScore,
			"level":  interpretation.QualityLevel,
			"issues": interpretation.Issues,
		}, map[string]any{
			"metadata_entropy": metadataEntropy,
			"kl_divergence":    klDivergence,
			"column_count":     float64(len(normalizedDtypes)),
		})

		analysisCtx, analysisCancel := context.WithTimeout(ctx, 90*time.Second) // Increased timeout for retries
		defer analysisCancel()

		// Analysis is non-fatal - failures don't affect graph processing
		analysis, err := s.deepAgentsClient.AnalyzeKnowledgeGraph(analysisCtx, graphSummary, req.ProjectID, req.SystemID)
		if err != nil {
			// Error already logged in AnalyzeKnowledgeGraph with retry details
			// Continue processing without analysis
		} else if analysis != nil {
			deepAgentsAnalysis = analysis
			s.logger.Printf("DeepAgents analysis completed and included in response")
		}
	}

	response := map[string]any{
		"nodes":            nodes,
		"edges":            edges,
		"metadata_entropy": metadataEntropy,
		"kl_divergence":    klDivergence,
		"root_node_id":     rootID,
		"metrics": map[string]any{
			"metadata_entropy":    metadataEntropy,
			"kl_divergence":       klDivergence,
			"actual_distribution": actualDistribution,
			"ideal_distribution":  idealDistribution,
			"column_count":        len(normalizedDtypes),
		},
		"quality": map[string]any{
			"score":               interpretation.QualityScore,
			"level":               interpretation.QualityLevel,
			"issues":              interpretation.Issues,
			"recommendations":     interpretation.Recommendations,
			"processing_strategy": interpretation.ProcessingStrategy,
			"needs_validation":    interpretation.NeedsValidation,
			"needs_review":        interpretation.NeedsReview,
		},
	}

	if signavioMetadata.ProcessCount > 0 || len(req.SignavioFiles) > 0 {
		signavioSummary := map[string]any{
			"process_count": signavioMetadata.ProcessCount,
			"source_files":  signavioMetadata.SourceFiles,
		}
		if len(signavioMetadata.Processes) > 0 {
			processes := make([]map[string]any, 0, len(signavioMetadata.Processes))
			for _, proc := range signavioMetadata.Processes {
				processes = append(processes, map[string]any{
					"id":            proc.ID,
					"name":          proc.Name,
					"source_file":   proc.SourceFile,
					"element_count": proc.ElementCount,
					"element_types": proc.ElementTypes,
				})
			}
			signavioSummary["processes"] = processes
		}
		if len(signavioMetadata.Errors) > 0 {
			signavioSummary["errors"] = signavioMetadata.Errors
		}
		response["signavio"] = signavioSummary
	}

	// Add DeepAgents analysis if available
	if deepAgentsAnalysis != nil {
		response["deepagents_analysis"] = map[string]any{
			"messages": deepAgentsAnalysis.Messages,
			"result":   deepAgentsAnalysis.Result,
		}
	}

	// Add warnings if present
	if interpretation.ShouldWarn {
		response["warnings"] = interpretation.Issues
	}

	if normResult.Stats != nil {
		response["normalization"] = map[string]any{
			"root_node_id": rootID,
			"stats":        normResult.Stats,
			"warnings":     normResult.Warnings,
		}
	}

	// Record telemetry for graph processing with information theory metrics
	if s.telemetry != nil {
		telemetryCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		telemetryRecord := monitoring.TelemetryRecord{
			LibraryType: "layer4_extract",
			Operation:   "graph_processing",
			Input: map[string]any{
				"json_tables_count":     len(req.JSONTables),
				"hive_ddls_count":       len(req.HiveDDLs),
				"sql_queries_count":     len(req.SqlQueries),
				"control_m_files_count": len(req.ControlMFiles),
				"signavio_files_count":  len(req.SignavioFiles),
				"project_id":            req.ProjectID,
				"system_id":             req.SystemID,
				"information_system_id": req.InformationSystemID,
			},
			Output: map[string]any{
				"nodes_count":            len(nodes),
				"edges_count":            len(edges),
				"metadata_entropy":       metadataEntropy,
				"kl_divergence":          klDivergence,
				"actual_distribution":    actualDistribution,
				"ideal_distribution":     idealDistribution,
				"column_count":           len(normalizedDtypes),
				"root_node_id":           rootID,
				"signavio_process_count": signavioMetadata.ProcessCount,
			},
			StartedAt:   started,
			CompletedAt: time.Now(),
			Latency:     time.Since(started),
		}
		if normResult.Stats != nil {
			telemetryRecord.Output["normalization_stats"] = normResult.Stats
		}
		if telemetryClient, ok := s.telemetry.(*monitoring.TelemetryClient); ok {
			if err := telemetryClient.Log(telemetryCtx, telemetryRecord); err != nil {
				s.logger.Printf("telemetry warning: %v", err)
			}
		}
	}

	handlers.WriteJSON(w, http.StatusOK, response)
}

// --- /catalog/projects ---

func (s *extractServer) handleGetProjects(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.catalog.Projects)
}

func (s *extractServer) handleAddProject(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var p catalog.Project
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.catalog.Projects = append(s.catalog.Projects, p)

	if err := s.catalog.Save(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
}

// --- /catalog/systems ---

func (s *extractServer) handleGetSystems(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.catalog.Systems)
}

func (s *extractServer) handleAddSystem(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var sys catalog.System
	if err := json.NewDecoder(r.Body).Decode(&sys); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.catalog.Systems = append(s.catalog.Systems, sys)

	if err := s.catalog.Save(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
}

// --- /knowledge-graph/query (Neo4j Cypher queries) ---

// handleNeo4jQuery handles Cypher query requests to Neo4j.
func (s *extractServer) handleNeo4jQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	if s.neo4jPersistence == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Neo4j not configured. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.",
		})
		return
	}

	var request struct {
		Query  string         `json:"query"`
		Params map[string]any `json:"params,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	if request.Query == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": "query is required",
		})
		return
	}

	if request.Params == nil {
		request.Params = make(map[string]any)
	}

	// Execute query with timeout
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	result, err := s.neo4jPersistence.ExecuteQuery(ctx, request.Query, request.Params)
	if err != nil {
		s.logger.Printf("Neo4j query error: %v", err)
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("query execution failed: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, result)
}

// handlePetriNetToLangGraph converts a Petri net from catalog to LangGraph workflow.
func (s *extractServer) handlePetriNetToLangGraph(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		PetriNetID string `json:"petri_net_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Get Petri net from catalog
	if s.catalog == nil {
		http.Error(w, "catalog not available", http.StatusInternalServerError)
		return
	}

	petriNetData, exists := s.catalog.PetriNets[req.PetriNetID]

	if !exists {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]any{
			"error": fmt.Sprintf("Petri net '%s' not found in catalog", req.PetriNetID),
		})
		return
	}

	// Convert catalog data to workflow.PetriNet struct
	petriNetJSON, err := json.Marshal(petriNetData)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	var petriNet workflow.PetriNet
	if err := json.Unmarshal(petriNetJSON, &petriNet); err != nil {
		http.Error(w, fmt.Sprintf("failed to unmarshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to LangGraph workflow with semantic search enabled
	converter := workflow.NewWorkflowConverter(s.logger)

	// Set Extract service URL for semantic search (use same service)
	baseURL := fmt.Sprintf("http://localhost:%s", os.Getenv("PORT"))
	if baseURL == "http://localhost:" {
		baseURL = "http://localhost:8081"
	}
	converter.SetExtractServiceURL(baseURL)

	langGraphWorkflow := converter.ConvertPetriNetToLangGraph(&petriNet)

	handlers.WriteJSON(w, http.StatusOK, langGraphWorkflow)
}

// handlePetriNetToAgentFlow converts a Petri net from catalog to AgentFlow workflow.
func (s *extractServer) handlePetriNetToAgentFlow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		PetriNetID string `json:"petri_net_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Get Petri net from catalog
	if s.catalog == nil {
		http.Error(w, "catalog not available", http.StatusInternalServerError)
		return
	}

	petriNetData, exists := s.catalog.PetriNets[req.PetriNetID]

	if !exists {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]any{
			"error": fmt.Sprintf("Petri net '%s' not found in catalog", req.PetriNetID),
		})
		return
	}

	// Convert catalog data to workflow.PetriNet struct
	petriNetJSON, err := json.Marshal(petriNetData)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	var petriNet workflow.PetriNet
	if err := json.Unmarshal(petriNetJSON, &petriNet); err != nil {
		http.Error(w, fmt.Sprintf("failed to unmarshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to AgentFlow workflow with semantic search enabled
	converter := workflow.NewWorkflowConverter(s.logger)

	// Set Extract service URL for semantic search (use same service)
	baseURL := fmt.Sprintf("http://localhost:%s", os.Getenv("PORT"))
	if baseURL == "http://localhost:" {
		baseURL = "http://localhost:8081"
	}
	converter.SetExtractServiceURL(baseURL)

	agentFlowWorkflow := converter.ConvertPetriNetToAgentFlow(&petriNet)

	handlers.WriteJSON(w, http.StatusOK, agentFlowWorkflow)
}

// handleAgentFlowRun executes an AgentFlow flow directly from Extract service
func (s *extractServer) handleAgentFlowRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req clients.AgentFlowRunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if s.agentFlowClient == nil {
		http.Error(w, "AgentFlow client not available", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	resp, err := s.agentFlowClient.RunFlow(ctx, &req)
	if err != nil {
		s.logger.Printf("AgentFlow execution failed: %v", err)
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]any{
			"error": err.Error(),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, resp)
}

// handlePetriNetToAdvancedLangGraph converts a Petri net to an advanced LangGraph workflow (Phase 7.3).
func (s *extractServer) handlePetriNetToAdvancedLangGraph(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		PetriNetID string `json:"petri_net_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Get Petri net from catalog
	if s.catalog == nil {
		http.Error(w, "catalog not available", http.StatusInternalServerError)
		return
	}

	petriNetData, exists := s.catalog.PetriNets[req.PetriNetID]

	if !exists {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]any{
			"error": fmt.Sprintf("Petri net '%s' not found in catalog", req.PetriNetID),
		})
		return
	}

	// Convert catalog data to workflow.PetriNet struct
	petriNetJSON, err := json.Marshal(petriNetData)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	var petriNet workflow.PetriNet
	if err := json.Unmarshal(petriNetJSON, &petriNet); err != nil {
		http.Error(w, fmt.Sprintf("failed to unmarshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to advanced LangGraph workflow
	converter := workflow.NewAdvancedWorkflowConverter(s.logger)

	// Set Extract service URL for semantic search
	baseURL := fmt.Sprintf("http://localhost:%s", os.Getenv("PORT"))
	if baseURL == "http://localhost:" {
		baseURL = "http://localhost:8081"
	}
	converter.SetExtractServiceURL(baseURL)

	advancedWorkflow := converter.ConvertPetriNetToAdvancedLangGraph(&petriNet)

	handlers.WriteJSON(w, http.StatusOK, advancedWorkflow)
}

// handleHealthStatus returns health status for all monitored services (Phase 9.2).
func (s *extractServer) handleHealthStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.selfHealingSystem == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Self-healing system not configured",
		})
		return
	}

	// Get health status for Neo4j
	healthStatus := map[string]any{
		"services": make(map[string]any),
	}

	if s.neo4jPersistence != nil {
		neo4jHealthy, err := s.selfHealingSystem.GetHealthStatus("neo4j")
		if err == nil {
			healthStatus["services"].(map[string]any)["neo4j"] = map[string]any{
				"healthy": neo4jHealthy,
				"status":  "healthy",
			}
		}
	}

	handlers.WriteJSON(w, http.StatusOK, healthStatus)
}

// handleVectorSearch performs vector similarity search (RAG endpoint).
func (s *extractServer) handleVectorSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	if s.vectorPersistence == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Vector persistence not configured",
		})
		return
	}

	var request struct {
		Query           string    `json:"query"`
		QueryVector     []float32 `json:"query_vector,omitempty"`
		ArtifactType    string    `json:"artifact_type,omitempty"`
		Limit           int       `json:"limit,omitempty"`
		Threshold       float32   `json:"threshold,omitempty"`
		UseSemantic     bool      `json:"use_semantic,omitempty"`      // Use semantic embeddings for query
		UseHybridSearch bool      `json:"use_hybrid_search,omitempty"` // Search both embedding types
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	if request.Limit <= 0 {
		request.Limit = 10
	}
	if request.Threshold <= 0 {
		request.Threshold = 0.5
	}

	// Generate embedding for query if not provided
	queryVector := request.QueryVector
	var semanticQueryVector []float32
	ctx := r.Context()

	if queryVector == nil && request.Query != "" {
		// Intelligent routing: determine if query is semantic or structural
		useSemantic := request.UseSemantic
		if !useSemantic {
			// Auto-detect: semantic queries are typically natural language
			useSemantic = s.isSemanticQuery(request.Query)
		}

		var err error
		artifactType := request.ArtifactType
		if artifactType == "" {
			artifactType = "sql-query"
		}

		// Generate appropriate embedding based on query type
		if useSemantic && os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true" {
			// Generate semantic embedding using sap-rpt-1-oss
			semanticQueryVector, err = embeddings.GenerateSemanticEmbedding(ctx, request.Query)
			if err != nil {
				s.logger.Printf("semantic embedding generation failed, falling back to relational: %v", err)
				// Fallback to relational embedding
				queryVector, err = s.generateQueryEmbedding(ctx, request.Query, artifactType)
			} else {
				// Use semantic embedding for search
				queryVector = semanticQueryVector
			}
		} else {
			// Generate relational/structural embedding
			queryVector, err = s.generateQueryEmbedding(ctx, request.Query, artifactType)
		}

		if err != nil {
			handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
				"error": fmt.Sprintf("failed to generate embedding: %v", err),
			})
			return
		}
	}

	if queryVector == nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": "query or query_vector is required",
		})
		return
	}

	// Perform search with hybrid capability
	var results []persistence.VectorSearchResult
	var err error

	// For hybrid search, we need both embeddings
	if request.UseHybridSearch && os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true" {
		// Generate both embeddings if not already done
		if len(semanticQueryVector) == 0 && request.Query != "" {
			semanticQueryVector, _ = embeddings.GenerateSemanticEmbedding(ctx, request.Query)
		}
		// Generate relational embedding if needed
		if len(queryVector) == 0 && request.Query != "" {
			queryVector, _ = s.generateQueryEmbedding(ctx, request.Query, request.ArtifactType)
		}
		// Perform hybrid search if both embeddings available
		if len(semanticQueryVector) > 0 && len(queryVector) > 0 {
			results, err = s.performHybridSearch(ctx, queryVector, semanticQueryVector, request.Query, request.ArtifactType, request.Limit, request.Threshold)
		}
	}

	// Fallback to standard search if hybrid failed or not requested
	if err != nil || len(results) == 0 {
		if request.Query != "" {
			// Text search first (if supported), then fallback to vector search
			results, err = s.vectorPersistence.SearchByText(request.Query, request.ArtifactType, request.Limit)
			if err != nil || len(results) == 0 {
				// Fallback to vector similarity search
				results, err = s.vectorPersistence.SearchSimilar(queryVector, request.ArtifactType, request.Limit, request.Threshold)
			}
		} else {
			// Vector similarity search only
			results, err = s.vectorPersistence.SearchSimilar(queryVector, request.ArtifactType, request.Limit, request.Threshold)
		}
	}

	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("search failed: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"results":        results,
		"total":          len(results),
		"query_type":     s.detectQueryType(request.Query),
		"embedding_type": s.getEmbeddingType(queryVector, semanticQueryVector),
		"hybrid_search":  request.UseHybridSearch,
		"semantic_used":  len(semanticQueryVector) > 0,
		"results_count":  len(results),
	})
}

// generateQueryEmbedding generates relational/structural embedding for query
func (s *extractServer) generateQueryEmbedding(ctx context.Context, query, artifactType string) ([]float32, error) {
	switch artifactType {
	case "sql-query":
		return embeddings.GenerateEmbedding(ctx, query)
	case "table":
		node := graph.Node{Label: query, Type: graph.NodeTypeTable}
		relational, _, err := embeddings.GenerateTableEmbedding(ctx, node)
		return relational, err
	default:
		return embeddings.GenerateEmbedding(ctx, query)
	}
}

// isSemanticQuery determines if a query is semantic (natural language) vs structural (SQL/technical)
func (s *extractServer) isSemanticQuery(query string) bool {
	queryLower := strings.ToLower(query)

	// Semantic query indicators
	semanticIndicators := []string{
		"find", "search", "show", "list", "get", "what", "where", "which", "how",
		"customer", "order", "transaction", "process", "workflow", "table about",
		"related to", "similar to", "like", "containing",
	}

	// Structural query indicators
	structuralIndicators := []string{
		"select", "from", "where", "join", "create", "insert", "update", "delete",
		"table", "column", "schema", "ddl", "sql",
	}

	// Count semantic and structural indicators
	semanticCount := 0
	structuralCount := 0

	for _, indicator := range semanticIndicators {
		if strings.Contains(queryLower, indicator) {
			semanticCount++
		}
	}

	for _, indicator := range structuralIndicators {
		if strings.Contains(queryLower, indicator) {
			structuralCount++
		}
	}

	// If more semantic indicators, it's a semantic query
	// If structural indicators present, it's likely structural
	if structuralCount > 0 {
		return false // Structural query
	}
	if semanticCount > 0 {
		return true // Semantic query
	}

	// Default: if query is short and doesn't look like SQL, treat as semantic
	if len(query) < 50 && !strings.Contains(queryLower, "select") && !strings.Contains(queryLower, "from") {
		return true
	}

	return false // Default to structural
}

// detectQueryType returns the detected query type
func (s *extractServer) detectQueryType(query string) string {
	if query == "" {
		return "unknown"
	}
	if s.isSemanticQuery(query) {
		return "semantic"
	}
	return "structural"
}

// getEmbeddingType returns which embedding type was used
func (s *extractServer) getEmbeddingType(relationalVector, semanticVector []float32) string {
	if len(semanticVector) > 0 {
		return "semantic"
	}
	if len(relationalVector) > 0 {
		return "relational"
	}
	return "unknown"
}

// performHybridSearch searches both relational and semantic embeddings and fuses results
func (s *extractServer) performHybridSearch(ctx context.Context, relationalVector, semanticVector []float32, queryText, artifactType string, limit int, threshold float32) ([]persistence.VectorSearchResult, error) {
	// Search relational embeddings
	relationalResults, err1 := s.searchRelationalEmbeddings(relationalVector, artifactType, limit, threshold)
	if err1 != nil {
		s.logger.Printf("relational search failed: %v", err1)
		relationalResults = []persistence.VectorSearchResult{}
	}

	// Search semantic embeddings
	semanticResults, err2 := s.searchSemanticEmbeddings(semanticVector, artifactType, limit, threshold)
	if err2 != nil {
		s.logger.Printf("semantic search failed: %v", err2)
		semanticResults = []persistence.VectorSearchResult{}
	}

	// Fuse results intelligently
	fusedResults := s.fuseSearchResults(relationalResults, semanticResults, limit)

	return fusedResults, nil
}

// searchRelationalEmbeddings searches relational embeddings
func (s *extractServer) searchRelationalEmbeddings(queryVector []float32, artifactType string, limit int, threshold float32) ([]persistence.VectorSearchResult, error) {
	// Search for embeddings with embedding_type = "relational_transformer"
	// We need to search all embeddings and filter by type
	results, err := s.vectorPersistence.SearchSimilar(queryVector, artifactType, limit*2, threshold)
	if err != nil {
		return nil, err
	}

	// Filter for relational embeddings
	relationalResults := []persistence.VectorSearchResult{}
	for _, result := range results {
		if result.Metadata != nil {
			if embeddingType, ok := result.Metadata["embedding_type"].(string); ok {
				if embeddingType == "relational_transformer" {
					relationalResults = append(relationalResults, result)
				}
			} else {
				// If no embedding_type specified, assume relational (legacy)
				relationalResults = append(relationalResults, result)
			}
		}
	}

	// Limit results
	if len(relationalResults) > limit {
		relationalResults = relationalResults[:limit]
	}

	return relationalResults, nil
}

// searchSemanticEmbeddings searches semantic embeddings
func (s *extractServer) searchSemanticEmbeddings(queryVector []float32, artifactType string, limit int, threshold float32) ([]persistence.VectorSearchResult, error) {
	// Search for embeddings with embedding_type = "sap_rpt_semantic"
	// We need to search all embeddings and filter by type
	results, err := s.vectorPersistence.SearchSimilar(queryVector, artifactType, limit*2, threshold)
	if err != nil {
		return nil, err
	}

	// Filter for semantic embeddings
	semanticResults := []persistence.VectorSearchResult{}
	for _, result := range results {
		if result.Metadata != nil {
			if embeddingType, ok := result.Metadata["embedding_type"].(string); ok {
				if embeddingType == "sap_rpt_semantic" {
					semanticResults = append(semanticResults, result)
				}
			}
		}
	}

	// Limit results
	if len(semanticResults) > limit {
		semanticResults = semanticResults[:limit]
	}

	return semanticResults, nil
}

// fuseSearchResults intelligently fuses results from both embedding types
func (s *extractServer) fuseSearchResults(relationalResults, semanticResults []persistence.VectorSearchResult, limit int) []persistence.VectorSearchResult {
	// Create a map to deduplicate by artifact_id
	resultMap := make(map[string]persistence.VectorSearchResult)
	scoreMap := make(map[string]float32)

	// Process relational results
	for _, result := range relationalResults {
		key := result.ArtifactID
		if existing, exists := resultMap[key]; exists {
			// If already exists, boost score (weighted average)
			const relationalWeight float32 = 0.4
			const semanticWeight float32 = 0.6
			combinedScore := (existing.Score * semanticWeight) + (result.Score * relationalWeight)
			result.Score = combinedScore
			// Merge metadata
			if result.Metadata != nil {
				if existing.Metadata == nil {
					existing.Metadata = make(map[string]any)
				}
				for k, v := range result.Metadata {
					existing.Metadata[k] = v
				}
				result.Metadata = existing.Metadata
			}
		}
		resultMap[key] = result
		scoreMap[key] = result.Score
	}

	// Process semantic results
	for _, result := range semanticResults {
		key := result.ArtifactID
		if existing, exists := resultMap[key]; exists {
			// If already exists, boost score (weighted average)
			const relationalWeight float32 = 0.4
			const semanticWeight float32 = 0.6
			existingScore := scoreMap[key]
			combinedScore := (existingScore * relationalWeight) + (result.Score * semanticWeight)
			result.Score = combinedScore
			scoreMap[key] = combinedScore
			// Merge metadata
			if result.Metadata != nil {
				if existing.Metadata == nil {
					existing.Metadata = make(map[string]any)
				}
				for k, v := range result.Metadata {
					existing.Metadata[k] = v
				}
				result.Metadata = existing.Metadata
			}
			resultMap[key] = result
		} else {
			// New result, add it
			resultMap[key] = result
			scoreMap[key] = result.Score
		}
	}

	// Convert map to slice and sort by score
	fusedResults := make([]persistence.VectorSearchResult, 0, len(resultMap))
	for _, result := range resultMap {
		result.Score = scoreMap[result.ArtifactID]
		fusedResults = append(fusedResults, result)
	}

	// Sort by score (descending)
	sort.Slice(fusedResults, func(i, j int) bool {
		return fusedResults[i].Score > fusedResults[j].Score
	})

	// Limit results
	if len(fusedResults) > limit {
		fusedResults = fusedResults[:limit]
	}

	return fusedResults
}

// handleGenerateEmbedding generates embedding for arbitrary text.
func (s *extractServer) handleGenerateEmbedding(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	var request struct {
		Text         string `json:"text"`
		ArtifactType string `json:"artifact_type"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	if request.Text == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": "text is required",
		})
		return
	}

	ctx := r.Context()
	var embedding []float32
	var err error

	artifactType := request.ArtifactType
	if artifactType == "" {
		artifactType = "sql-query"
	}

	switch artifactType {
	case "sql-query":
		embedding, err = embeddings.GenerateEmbedding(ctx, request.Text)
	case "table":
		node := graph.Node{Label: request.Text, Type: graph.NodeTypeTable}
		var relational []float32
		relational, _, err = embeddings.GenerateTableEmbedding(ctx, node)
		embedding = relational
	case "column":
		node := graph.Node{Label: request.Text, Type: graph.NodeTypeColumn}
		embedding, err = embeddings.GenerateColumnEmbedding(ctx, node)
	default:
		// Default to SQL embedding
		embedding, err = embeddings.GenerateEmbedding(ctx, request.Text)
	}

	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("failed to generate embedding: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"embedding": embedding,
		"dimension": len(embedding),
	})
}

// handleGetEmbedding retrieves embedding and metadata by key.
func (s *extractServer) handleGetEmbedding(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.vectorPersistence == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Vector persistence not configured",
		})
		return
	}

	// Extract key from path: /knowledge-graph/embed/{key}
	path := strings.TrimPrefix(r.URL.Path, "/knowledge-graph/embed/")
	if path == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": "key is required in path",
		})
		return
	}

	vector, metadata, err := s.vectorPersistence.GetVector(path)
	if err != nil {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]string{
			"error": fmt.Sprintf("vector not found: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"key":       path,
		"embedding": vector,
		"metadata":  metadata,
	})
}

// handleTrainingDataStats returns statistics about collected training data (Phase 4)
func (s *extractServer) handleTrainingDataStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.trainingDataCollector == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Training data collector not configured",
		})
		return
	}

	stats, err := s.trainingDataCollector.GetTrainingDataStats()
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("failed to get training data stats: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, stats)
}

// handleExportTrainingData exports collected training data (Phase 4)
func (s *extractServer) handleExportTrainingData(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	if s.trainingDataCollector == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Training data collector not configured",
		})
		return
	}

	var request struct {
		DestinationPath string `json:"destination_path"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	if request.DestinationPath == "" {
		request.DestinationPath = fmt.Sprintf("./training_data_export_%d.json", time.Now().Unix())
	}

	if err := s.trainingDataCollector.ExportTrainingData(request.DestinationPath); err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("failed to export training data: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"status":           "exported",
		"destination_path": request.DestinationPath,
		"exported_at":      time.Now().UTC().Format(time.RFC3339),
	})
}

// handleModelMetrics returns model performance metrics (Phase 5)
func (s *extractServer) handleModelMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.modelMonitor == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Model monitor not configured",
		})
		return
	}

	metrics := s.modelMonitor.GetMetrics()
	handlers.WriteJSON(w, http.StatusOK, metrics)
}

// handleUncertainPredictions returns predictions that need manual review (Phase 5 active learning)
func (s *extractServer) handleUncertainPredictions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.modelMonitor == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Model monitor not configured",
		})
		return
	}

	limit := 10
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	uncertain := s.modelMonitor.GetUncertainPredictions(limit)
	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"uncertain_predictions": uncertain,
		"count":                 len(uncertain),
	})
}

// handleOCR extracts text and tables from images using DeepSeek-OCR (Phase 6)
func (s *extractServer) handleOCR(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	if s.multiModalExtractor == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Multi-modal extractor not configured",
		})
		return
	}

	var request struct {
		ImagePath   string `json:"image_path"`
		ImageBase64 string `json:"image_base64"`
		Prompt      string `json:"prompt"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	var ocrResult *extraction.OCRResult
	var err error

	if request.ImageBase64 != "" {
		ocrResult, err = s.multiModalExtractor.ExtractFromImageBase64(request.ImageBase64, request.Prompt)
	} else if request.ImagePath != "" {
		ocrResult, err = s.multiModalExtractor.ExtractFromImage(request.ImagePath, request.Prompt)
	} else {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": "image_path or image_base64 required",
		})
		return
	}

	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("OCR extraction failed: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, ocrResult)
}

// handleUnifiedExtraction performs unified multi-modal extraction (Phase 6)
func (s *extractServer) handleUnifiedExtraction(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	if s.multiModalExtractor == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Multi-modal extractor not configured",
		})
		return
	}

	var request struct {
		ImagePath        string           `json:"image_path"`
		ImageBase64      string           `json:"image_base64"`
		TableName        string           `json:"table_name"`
		Columns          []map[string]any `json:"columns"`
		Text             string           `json:"text"`
		Prompt           string           `json:"prompt"`
		TrainingDataPath string           `json:"training_data_path"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	result, err := s.multiModalExtractor.ExtractUnified(
		request.ImagePath,
		request.ImageBase64,
		request.TableName,
		request.Columns,
		request.Text,
		request.Prompt,
		request.TrainingDataPath,
	)

	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("unified extraction failed: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, result)
}

// handleMultimodalEmbeddings generates unified embeddings (Phase 6)
func (s *extractServer) handleMultimodalEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	defer r.Body.Close()

	if s.multiModalExtractor == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Multi-modal extractor not configured",
		})
		return
	}

	var request struct {
		Text      string           `json:"text"`
		ImagePath string           `json:"image_path"`
		TableName string           `json:"table_name"`
		Columns   []map[string]any `json:"columns"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
		return
	}

	embeddings, err := s.multiModalExtractor.GenerateUnifiedEmbeddings(
		request.Text,
		request.ImagePath,
		request.TableName,
		request.Columns,
		nil, // tables
	)

	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("embedding generation failed: %v", err),
		})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, embeddings)
}

// handleGraphQueryHelpers returns common graph query helpers.
func (s *extractServer) handleGraphQueryHelpers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	helpers := graph.NewGraphQueryHelpers(s.logger)

	queries := map[string]string{
		"petri_nets":                  helpers.QueryPetriNets(),
		"petri_net_transitions":       helpers.QueryPetriNetTransitions("${petri_net_id}"),
		"workflow_paths":              helpers.QueryWorkflowPaths("${petri_net_id}"),
		"transaction_tables":          helpers.QueryTransactionTables(),
		"processing_sequences":        helpers.QueryProcessingSequences(),
		"table_classifications":       helpers.QueryTableClassifications(),
		"code_parameters":             helpers.QueryCodeParameters(""),
		"hardcoded_lists":             helpers.QueryHardcodedLists(),
		"testing_endpoints":           helpers.QueryTestingEndpoints(),
		"advanced_extraction_summary": helpers.QueryAdvancedExtractionSummary(),
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"queries": queries,
		"usage": map[string]string{
			"petri_nets":                  "Find all Petri nets in the knowledge graph",
			"petri_net_transitions":       "Find transitions with SQL subprocesses (replace ${petri_net_id})",
			"workflow_paths":              "Find workflow paths in a Petri net (replace ${petri_net_id})",
			"transaction_tables":          "Find all transaction tables",
			"processing_sequences":        "Find table processing sequences",
			"table_classifications":       "Find all table classifications",
			"code_parameters":             "Find code parameters (optionally filter by source type)",
			"hardcoded_lists":             "Find hardcoded lists/constants",
			"testing_endpoints":           "Find testing endpoints",
			"advanced_extraction_summary": "Get summary of advanced extraction results",
		},
	})
}

// --- /catalog/information-systems ---

func (s *extractServer) handleGetInformationSystems(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.catalog.InformationSystems)
}

func (s *extractServer) handleAddInformationSystem(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var is catalog.InformationSystem
	if err := json.NewDecoder(r.Body).Decode(&is); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

		s.catalog.InformationSystems = append(s.catalog.InformationSystems, is)

	if err := s.catalog.Save(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
}

func (s *extractServer) extractSchemaFromJSON(path string) ([]graph.Node, []graph.Edge, []map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read file: %w", err)
	}

	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return nil, nil, nil, fmt.Errorf("unmarshal json: %w", err)
	}

	var objects []map[string]any
	switch value := v.(type) {
	case map[string]any:
		objects = []map[string]any{value}
	case []any:
		for _, item := range value {
			if obj, ok := item.(map[string]any); ok {
				objects = append(objects, obj)
			}
		}
	}

	if len(objects) == 0 {
		return nil, nil, nil, errors.New("json file does not contain any valid objects")
	}

	columnProfiles := profileJSONColumns(objects)
	columnNames := make([]string, 0, len(columnProfiles))
	for name := range columnProfiles {
		columnNames = append(columnNames, name)
	}
	sort.Strings(columnNames)

	var nodes []graph.Node
	var edges []graph.Edge

	tableID := filepath.Base(path)
	tableNode := graph.Node{
		ID:    tableID,
		Type:  graph.NodeTypeTable,
		Label: tableID,
	}
	nodes = append(nodes, tableNode)

	for _, key := range columnNames {
		profile := columnProfiles[key]
		columnID := fmt.Sprintf("%s.%s", tableID, key)
		columnNode := graph.Node{
			ID:    columnID,
			Type:  graph.NodeTypeColumn,
			Label: key,
			Props: profile.toProps(),
		}
		nodes = append(nodes, columnNode)

		edges = append(edges, graph.Edge{
			SourceID: tableID,
			TargetID: columnID,
			Label:    "HAS_COLUMN",
		})
	}

	return nodes, edges, objects, nil
}

type columnProfile struct {
	counts       map[string]int
	nullCount    int
	presentCount int
	totalRows    int
	examples     []any
}

func profileJSONColumns(objects []map[string]any) map[string]*columnProfile {
	total := len(objects)
	profiles := map[string]*columnProfile{}

	for _, obj := range objects {
		for key, value := range obj {
			profile, ok := profiles[key]
			if !ok {
				profile = &columnProfile{
					counts:    map[string]int{},
					totalRows: total,
				}
				profiles[key] = profile
			}

			profile.presentCount++

			valueType := inferJSONType(value)
			if valueType == "null" {
				profile.nullCount++
			} else {
				profile.counts[valueType]++
				if len(profile.examples) < maxExamplePreviews {
					profile.examples = append(profile.examples, value)
				}
			}
		}
	}

	return profiles
}

func (p *columnProfile) toProps() map[string]any {
	props := map[string]any{}

	typeKeys := make([]string, 0, len(p.counts))
	for key := range p.counts {
		typeKeys = append(typeKeys, key)
	}
	sort.Strings(typeKeys)

	switch len(typeKeys) {
	case 0:
		props["type"] = "unknown"
	case 1:
		props["type"] = normalizeColumnType(typeKeys[0])
	default:
		props["type"] = "mixed"
		normalizedTypes := make([]string, len(typeKeys))
		for i, t := range typeKeys {
			normalizedTypes[i] = normalizeColumnType(t)
		}
		props["types"] = normalizedTypes
	}

	nullable := p.nullCount > 0 || (p.totalRows > 0 && p.presentCount < p.totalRows)
	props["nullable"] = nullable

	if p.totalRows > 0 {
		props["presence_ratio"] = float64(p.presentCount) / float64(p.totalRows)
	}

	if len(p.examples) > 0 {
		props["examples"] = p.examples
	}

	return storage.MapOrNil(props)
}

func inferJSONType(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64:
		return "number"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	default:
		return fmt.Sprintf("%T", value)
	}
}

// validateGraph performs validation checks on the graph before persistence
func validateGraph(nodes []graph.Node, edges []graph.Edge) []string {
	var warnings []string

	// Check for orphan columns
	tableMap := make(map[string]bool)
	columnMap := make(map[string]bool)
	hasColumnEdges := make(map[string]bool)

	for _, node := range nodes {
		if node.Type == "table" {
			tableMap[node.ID] = true
		} else if node.Type == "column" {
			columnMap[node.ID] = true
		}
	}

	for _, edge := range edges {
		if edge.Label == "HAS_COLUMN" && columnMap[edge.TargetID] {
			hasColumnEdges[edge.TargetID] = true
		}
	}

	orphanCount := 0
	for colID := range columnMap {
		if !hasColumnEdges[colID] {
			orphanCount++
		}
	}

	if orphanCount > 0 {
		warnings = append(warnings, fmt.Sprintf("validation: %d orphan columns found (missing HAS_COLUMN edges)", orphanCount))
	}

	// Check for orphan edges (edges pointing to non-existent nodes)
	nodeMap := make(map[string]bool)
	for _, node := range nodes {
		nodeMap[node.ID] = true
	}

	orphanEdgeCount := 0
	for _, edge := range edges {
		if !nodeMap[edge.SourceID] || !nodeMap[edge.TargetID] {
			orphanEdgeCount++
		}
	}

	if orphanEdgeCount > 0 {
		warnings = append(warnings, fmt.Sprintf("validation: %d orphan edges found (pointing to non-existent nodes)", orphanEdgeCount))
	}

	return warnings
}

// normalizeColumnType normalizes column type names to a consistent format
func normalizeColumnType(rawType string) string {
	rawType = strings.ToLower(strings.TrimSpace(rawType))
	if rawType == "" {
		return "unknown"
	}

	// Normalize common variations
	typeMap := map[string]string{
		"string":    "string",
		"varchar":   "string",
		"text":      "string",
		"char":      "string",
		"decimal":   "decimal",
		"numeric":   "decimal",
		"number":    "decimal",
		"float":     "decimal",
		"double":    "decimal",
		"int":       "integer",
		"bigint":    "integer",
		"integer":   "integer",
		"smallint":  "integer",
		"tinyint":   "integer",
		"date":      "date",
		"timestamp": "timestamp",
		"datetime":  "timestamp",
		"boolean":   "boolean",
		"bool":      "boolean",
	}

	if normalized, ok := typeMap[rawType]; ok {
		return normalized
	}
	return rawType
}

// --- langextract helpers ---

type exampleExtraction struct {
	ExtractionClass string         `json:"extraction_class"`
	ExtractionText  string         `json:"extraction_text"`
	Attributes      map[string]any `json:"attributes,omitempty"`
}

type exampleData struct {
	Text        string              `json:"text"`
	Extractions []exampleExtraction `json:"extractions"`
}

type extractRequest struct {
	Document           string          `json:"document"`
	Documents          []string        `json:"documents"`
	TextOrDocumentsRaw json.RawMessage `json:"text_or_documents"`
	PromptDescription  string          `json:"prompt_description"`
	ModelID            string          `json:"model_id"`
	Examples           []exampleData   `json:"examples"`
	APIKey             string          `json:"api_key"`
}

type langextractPayload struct {
	TextOrDocuments any           `json:"text_or_documents"`
	Prompt          string        `json:"prompt_description"`
	ModelID         string        `json:"model_id"`
	Examples        []exampleData `json:"examples,omitempty"`
	APIKey          string        `json:"api_key,omitempty"`
}

type extractionResult struct {
	ExtractionClass string         `json:"extraction_class"`
	ExtractionText  string         `json:"extraction_text"`
	Attributes      map[string]any `json:"attributes,omitempty"`
	StartIndex      *int           `json:"start_index,omitempty"`
	EndIndex        *int           `json:"end_index,omitempty"`
}

type langextractResponse struct {
	Extractions []extractionResult `json:"extractions"`
	Error       string             `json:"error,omitempty"`
}

type extractResponse struct {
	Entities    map[string][]string `json:"entities"`
	Extractions []extractionResult  `json:"extractions"`
}

func (s *extractServer) buildLangextractPayload(ctx context.Context, req extractRequest) (*langextractPayload, []llms.Token, error) {
	var textOrDocs any
	if len(req.TextOrDocumentsRaw) > 0 {
		if err := json.Unmarshal(req.TextOrDocumentsRaw, &textOrDocs); err != nil {
			return nil, nil, fmt.Errorf("text_or_documents: %w", err)
		}
	} else if len(req.Documents) > 0 {
		textOrDocs = req.Documents
	} else if strings.TrimSpace(req.Document) != "" {
		textOrDocs = req.Document
	} else {
		return nil, nil, errors.New("document content is required")
	}

	// Build token-aware prompt using TOON
	promptTemplate := strings.TrimSpace(req.PromptDescription)
	if promptTemplate == "" {
		promptTemplate = defaultPromptDescription
	}

	// Create a prompt template with variables
	// Extract variables from prompt if it contains template syntax
	var promptTokens []llms.Token
	var prompt string
	
	// Check if prompt contains template variables (e.g., {{.variable}})
	if strings.Contains(promptTemplate, "{{") {
		// Parse template variables
		pt := prompts.NewPromptTemplate(promptTemplate, []string{})
		promptValue, err := pt.FormatPrompt(map[string]any{})
		if err == nil {
			if tokenAware, ok := promptValue.(llms.TokenAwarePromptValue); ok {
				promptTokens = tokenAware.Tokens()
				prompt = promptValue.String()
			} else {
				prompt = promptValue.String()
			}
		} else {
			// Fallback to plain string
			prompt = promptTemplate
		}
	} else {
		// Plain string prompt - create simple token structure
		prompt = promptTemplate
		promptTokens = []llms.Token{
			{
				Type:  "text",
				Value: prompt,
				Metadata: map[string]string{
					"length": strconv.Itoa(len(prompt)),
				},
			},
		}
	}

	modelID := strings.TrimSpace(req.ModelID)
	if modelID == "" {
		modelID = defaultModelID
	}

	apiKey := strings.TrimSpace(req.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(s.apiKey)
	}

	return &langextractPayload{
		TextOrDocuments: textOrDocs,
		Prompt:          prompt,
		ModelID:         modelID,
		Examples:        req.Examples,
		APIKey:          apiKey,
	}, promptTokens, nil
}

func (s *extractServer) invokeLangextract(ctx context.Context, payload *langextractPayload) (*langextractResponse, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.langextractURL+"/extract", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post to langextract: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("langextract responded with %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	var langResp langextractResponse
	if err := json.NewDecoder(resp.Body).Decode(&langResp); err != nil {
		return nil, fmt.Errorf("decode langextract response: %w", err)
	}

	return &langResp, nil
}

type extractError struct {
	err    error
	status int
}

func (e *extractError) Error() string {
	if e.err != nil {
		return e.err.Error()
	}
	return "extract error"
}

func (s *extractServer) runExtract(ctx context.Context, req extractRequest) (extractResponse, error) {
	payload, promptTokens, err := s.buildLangextractPayload(ctx, req)
	if err != nil {
		return extractResponse{}, &extractError{err: err, status: http.StatusBadRequest}
	}

	// Log prompt tokens to telemetry if enabled (graceful - doesn't fail if telemetry unavailable)
	if s.telemetry != nil && len(promptTokens) > 0 {
		// Use type assertion with recovery to handle telemetry client gracefully
		if telemetryClient, ok := s.telemetry.(*monitoring.TelemetryClient); ok && telemetryClient != nil {
			promptID := monitoring.PromptID(req.PromptDescription, map[string]any{
				"model_id": payload.ModelID,
			})
			templateType := "template"
			if !strings.Contains(req.PromptDescription, "{{") {
				templateType = "text"
			}
			variableCount := 0
			if strings.Contains(req.PromptDescription, "{{") {
				// Count variables in template
				variableCount = strings.Count(req.PromptDescription, "{{")
			}
			
			// Log asynchronously to avoid blocking extraction
			go func() {
				defer func() {
					if r := recover(); r != nil {
						s.logger.Printf("telemetry token logging recovered from panic: %v", r)
					}
				}()
				telemetryCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				defer cancel()
				if err := telemetryClient.LogPromptTokens(telemetryCtx, promptID, templateType, promptTokens, variableCount); err != nil {
					s.logger.Printf("failed to log prompt tokens (non-fatal): %v", err)
				}
			}()
		}
	}

	langResp, err := s.invokeLangextract(ctx, payload)
	if err != nil {
		return extractResponse{}, &extractError{err: err, status: http.StatusInternalServerError}
	}

	if langResp.Error != "" {
		return extractResponse{}, &extractError{err: errors.New(langResp.Error), status: http.StatusInternalServerError}
	}

	entities := groupExtractions(langResp.Extractions)
	response := extractResponse{
		Entities:    entities,
		Extractions: langResp.Extractions,
	}

	return response, nil
}

func telemetryInputFromRequest(req extractRequest) map[string]any {
	summary := map[string]any{
		"prompt_description": strings.TrimSpace(req.PromptDescription),
		"model_id":           strings.TrimSpace(req.ModelID),
		"documents_count":    len(req.Documents),
		"examples_count":     len(req.Examples),
	}

	if doc := strings.TrimSpace(req.Document); doc != "" {
		if len(doc) > documentPreviewLength {
			doc = doc[:documentPreviewLength]
		}
		summary["document_preview"] = doc
	}

	return summary
}

func telemetryOutputFromResponse(resp *extractResponse) map[string]any {
	if resp == nil {
		return map[string]any{}
	}

	entityKeys := make([]string, 0, len(resp.Entities))
	for key := range resp.Entities {
		entityKeys = append(entityKeys, key)
	}

	return map[string]any{
		"extraction_count": len(resp.Extractions),
		"entity_keys":      entityKeys,
	}
}

type markitdownClientAdapter struct {
	client *clients.MarkItDownClient
}

func (a *markitdownClientAdapter) ConvertFile(ctx context.Context, filePath string) (*integrations.MarkItDownResponse, error) {
	resp, err := a.client.ConvertFile(ctx, filePath)
	if err != nil {
		return nil, err
	}
	return &integrations.MarkItDownResponse{
		TextContent: resp.TextContent,
		Metadata:    resp.Metadata,
		Format:      resp.Format,
		Error:       resp.Error,
	}, nil
}

func (a *markitdownClientAdapter) ConvertBytes(ctx context.Context, fileData []byte, fileExtension string) (*integrations.MarkItDownResponse, error) {
	resp, err := a.client.ConvertBytes(ctx, fileData, fileExtension)
	if err != nil {
		return nil, err
	}
	return &integrations.MarkItDownResponse{
		TextContent: resp.TextContent,
		Metadata:    resp.Metadata,
		Format:      resp.Format,
		Error:       resp.Error,
	}, nil
}

func (a *markitdownClientAdapter) IsFormatSupported(fileExtension string) bool {
	return a.client.IsFormatSupported(fileExtension)
}

func (a *markitdownClientAdapter) HealthCheck(ctx context.Context) (bool, error) {
	return a.client.HealthCheck(ctx)
}

func groupExtractions(extractions []extractionResult) map[string][]string {
	grouped := map[string][]string{}
	for _, ext := range extractions {
		class := strings.ToLower(strings.TrimSpace(ext.ExtractionClass))
		text := strings.TrimSpace(ext.ExtractionText)
		if class == "" || text == "" {
			continue
		}
		if !utils.ContainsString(grouped[class], text) {
			grouped[class] = append(grouped[class], text)
		}
	}
	ensureEntityBuckets(grouped)
	return grouped
}

func ensureEntityBuckets(grouped map[string][]string) {
	for _, key := range []string{"people", "projects", "dates", "locations"} {
		if _, ok := grouped[key]; !ok {
			grouped[key] = []string{}
		}
	}
}

// --- training generation ---

type trainingGenerationRequest struct {
	Mode            string                  `json:"mode"`
	TableOptions    tableGenerationInput    `json:"table,omitempty"`
	DocumentOptions documentGenerationInput `json:"document,omitempty"`
}

type tableGenerationInput struct {
	Schema      string   `json:"schema"`
	Tables      []string `json:"tables"`
	Limit       int      `json:"limit"`
	OutputDir   string   `json:"output_dir"`
	Format      string   `json:"format"` // csv or jsonl
	Description string   `json:"description"`
}

type documentGenerationInput struct {
	Inputs    []string `json:"inputs"`
	OutputDir string   `json:"output_dir"`
	Prompt    string   `json:"prompt"`
	Format    string   `json:"format"` // markdown, json
}

type generationResult struct {
	FilePaths    []string
	ManifestPath string
}

func (s *extractServer) generateTableExtract(ctx context.Context, input tableGenerationInput) (generationResult, error) {
	host := os.Getenv("HANA_HOST")
	user := os.Getenv("HANA_USER")
	password := os.Getenv("HANA_PASSWORD")
	if host == "" || user == "" || password == "" {
		return generationResult{}, errors.New("HANA connection details not configured (HANA_HOST, HANA_USER, HANA_PASSWORD)")
	}

	port := 39015
	if raw := strings.TrimSpace(os.Getenv("HANA_PORT")); raw != "" {
		if p, err := strconv.Atoi(raw); err == nil {
			port = p
		}
	}
	database := os.Getenv("HANA_DATABASE")
	if database == "" {
		database = "HXE"
	}

	dsn := fmt.Sprintf("hdb://%s:%s@%s:%d/%s", user, password, host, port, database)
	db, err := sql.Open("hdb", dsn)
	if err != nil {
		return generationResult{}, fmt.Errorf("open hana: %w", err)
	}
	defer db.Close()

	if err := db.PingContext(ctx); err != nil {
		return generationResult{}, fmt.Errorf("ping hana: %w", err)
	}

	schema := input.Schema
	if schema == "" {
		schema = os.Getenv("HANA_SCHEMA")
	}
	if schema == "" {
		schema = strings.ToUpper(user)
	}

	tables := input.Tables
	if len(tables) == 0 {
		rows, err := db.QueryContext(ctx, `SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = ? LIMIT 5`, schema)
		if err != nil {
			return generationResult{}, fmt.Errorf("list tables: %w", err)
		}
		defer rows.Close()
		for rows.Next() {
			var name string
			if err := rows.Scan(&name); err != nil {
				return generationResult{}, fmt.Errorf("scan table: %w", err)
			}
			tables = append(tables, name)
		}
		if err := rows.Err(); err != nil {
			return generationResult{}, fmt.Errorf("iterate tables: %w", err)
		}
	}

	limit := input.Limit
	if limit <= 0 {
		limit = 500
	}
	format := strings.ToLower(input.Format)
	if format == "" {
		format = "jsonl"
	}

	timestamp := time.Now().UTC().Format("20060102T150405Z")
	outputBase := input.OutputDir
	if outputBase == "" {
		outputBase = filepath.Join(s.trainingDir, "tables", timestamp)
	}
	if err := os.MkdirAll(outputBase, 0o755); err != nil {
		return generationResult{}, fmt.Errorf("create output dir: %w", err)
	}

	var generated []string
	for _, table := range tables {
		// Sanitize schema and table names to prevent SQL injection
		sanitizedSchema, sanitizedTable, err := utils.SanitizeSchemaAndTable(schema, table)
		if err != nil {
			s.logger.Printf("table %s.%s skipped (invalid identifier): %v", schema, table, err)
			continue
		}
		query := fmt.Sprintf(`SELECT * FROM "%s"."%s" LIMIT %d`, sanitizedSchema, sanitizedTable, limit)
		rows, err := db.QueryContext(ctx, query)
		if err != nil {
			s.logger.Printf("table %s skipped (query error): %v", table, err)
			continue
		}

		outputPath := filepath.Join(outputBase, fmt.Sprintf("%s.%s", strings.ToLower(table), format))
		if err := writeRows(rows, format, outputPath); err != nil {
			s.logger.Printf("table %s skipped (write error): %v", table, err)
			rows.Close()
			continue
		}
		rows.Close()
		generated = append(generated, outputPath)
	}

	if len(generated) == 0 {
		return generationResult{}, errors.New("no tables exported")
	}

	manifestData := map[string]any{
		"type":        "table",
		"schema":      schema,
		"tables":      tables,
		"limit":       limit,
		"format":      format,
		"description": input.Description,
		"timestamp":   timestamp,
		"files":       generated,
	}

	manifestPath := filepath.Join(outputBase, "manifest.json")
	if err := handlers.WriteJSONFile(manifestPath, manifestData); err != nil {
		return generationResult{}, err
	}

	// Optional orchestration integration via LangGraph workflow:
	// If ORCHESTRATION_ENABLED is set, orchestration chains can be executed
	// via the graph service /orchestration/process endpoint.
	// This allows orchestration to be used without direct dependency on chains package.
	// For direct chain usage, see services/graph/pkg/workflows/orchestration_processor.go
	// Note: Orchestration is now enabled via LangGraph workflows, not direct chain calls

	// Adapt actual output as needed for your system:
	return generationResult{
		FilePaths:    generated,
		ManifestPath: manifestPath,
	}, nil
}

func writeRows(rows *sql.Rows, format, outputPath string) error {
	defer rows.Close()

	columns, err := rows.Columns()
	if err != nil {
		return fmt.Errorf("columns: %w", err)
	}
	raw := make([]any, len(columns))
	ptrs := make([]any, len(columns))
	for i := range raw {
		ptrs[i] = &raw[i]
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", outputPath, err)
	}
	defer f.Close()

	switch format {
	case "csv":
		writer := csv.NewWriter(f)
		if err := writer.Write(columns); err != nil {
			return fmt.Errorf("csv header: %w", err)
		}
		for rows.Next() {
			if err := rows.Scan(ptrs...); err != nil {
				return fmt.Errorf("scan row: %w", err)
			}
			record := make([]string, len(columns))
			for i, v := range raw {
				record[i] = fmt.Sprintf("%v", normalizeValue(v))
			}
			if err := writer.Write(record); err != nil {
				return fmt.Errorf("csv write: %w", err)
			}
		}
		writer.Flush()
		return writer.Error()
	default: // jsonl
		enc := json.NewEncoder(f)
		for rows.Next() {
			if err := rows.Scan(ptrs...); err != nil {
				return fmt.Errorf("scan row: %w", err)
			}
			rowMap := make(map[string]any, len(columns))
			for i, col := range columns {
				rowMap[col] = normalizeValue(raw[i])
			}
			if err := enc.Encode(rowMap); err != nil {
				return fmt.Errorf("json encode: %w", err)
			}
		}
		return nil
	}
}

func normalizeValue(v any) any {
	switch val := v.(type) {
	case nil:
		return nil
	case []byte:
		return string(val)
	case time.Time:
		return val.UTC().Format(time.RFC3339Nano)
	default:
		return val
	}
}

// writeJSONFile moved to internal/handlers/helpers.go
func _writeJSONFile(path string, payload any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create manifest dir: %w", err)
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

func (s *extractServer) generateDocumentExtract(ctx context.Context, input documentGenerationInput) (generationResult, error) {
	if len(input.Inputs) == 0 {
		return generationResult{}, errors.New("document.inputs is required")
	}

	timestamp := time.Now().UTC().Format("20060102T150405Z")
	outputDir := input.OutputDir
	if outputDir == "" {
		outputDir = filepath.Join(s.trainingDir, "documents", timestamp)
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return generationResult{}, fmt.Errorf("create output dir: %w", err)
	}

	format := strings.ToLower(input.Format)
	if format == "" {
		format = "markdown"
	}
	prompt := input.Prompt
	if prompt == "" {
		prompt = "<image>\n<|grounding|>Convert the document to markdown."
	}

	var outputs []string
	for _, in := range input.Inputs {
		absInput, err := filepath.Abs(in)
		if err != nil {
			return generationResult{}, fmt.Errorf("resolve input %s: %w", in, err)
		}
		base := strings.TrimSuffix(filepath.Base(absInput), filepath.Ext(absInput))
		outPath := filepath.Join(outputDir, base+".md")
		if format == "json" {
			outPath = filepath.Join(outputDir, base+".json")
		}

		// Try markitdown first if enabled and format is supported
		if s.markitdownIntegration != nil && s.markitdownIntegration.ShouldUseMarkItDown(absInput) {
			markdownContent, err := s.markitdownIntegration.ConvertDocument(ctx, absInput)
			if err == nil {
				// Successfully converted with markitdown, write to output
				if err := os.WriteFile(outPath, []byte(markdownContent), 0o644); err != nil {
					return generationResult{}, fmt.Errorf("failed to write markitdown output: %w", err)
				}
				outputs = append(outputs, outPath)
				if s.logger != nil {
					s.logger.Printf("Converted %s to markdown using MarkItDown", absInput)
				}
				continue
			}
			// MarkItDown failed, fallback to OCR if enabled (handled internally by markitdownIntegration)
			if s.logger != nil {
				s.logger.Printf("MarkItDown conversion failed for %s, falling back to OCR", absInput)
			}
		}

		// Fallback to OCR command
		if len(s.ocrCommand) == 0 {
			return generationResult{}, errors.New("OCR command not configured (set DEEPSEEK_OCR_SCRIPT or OCR_COMMAND) and MarkItDown unavailable")
		}

		cmdArgs := append([]string{}, s.ocrCommand...)
		cmdArgs = append(cmdArgs, "--input", absInput, "--output", outPath, "--prompt", prompt, "--format", format)

		cmd := exec.CommandContext(ctx, cmdArgs[0], cmdArgs[1:]...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return generationResult{}, fmt.Errorf("ocr command failed for %s: %w", absInput, err)
		}
		outputs = append(outputs, outPath)
	}

	manifest := map[string]any{
		"type":      "document",
		"format":    format,
		"prompt":    input.Prompt,
		"inputs":    input.Inputs,
		"timestamp": timestamp,
		"files":     outputs,
	}
	manifestPath := filepath.Join(outputDir, "manifest.json")
	if err := handlers.WriteJSONFile(manifestPath, manifest); err != nil {
		return generationResult{}, err
	}

	return generationResult{
		FilePaths:    outputs,
		ManifestPath: manifestPath,
	}, nil
}

// deriveOCRCommand moved to internal/handlers/helpers.go
func _deriveOCRCommand() []string {
	if cmd := strings.TrimSpace(os.Getenv("OCR_COMMAND")); cmd != "" {
		return strings.Fields(cmd)
	}
	script := strings.TrimSpace(os.Getenv("DEEPSEEK_OCR_SCRIPT"))
	if script == "" {
		script = "./scripts/utils/deepseek_ocr_cli.py"
	}
	if _, err := os.Stat(script); err != nil {
		return nil
	}
	python := strings.TrimSpace(os.Getenv("OCR_PYTHON"))
	if python == "" {
		python = "python3"
	}
	return []string{python, script}
}

// --- utilities ---

// writeJSON moved to internal/handlers/helpers.go

type CatalogAsset struct {
	EntityID      string
	AssetType     string // "table", "etl", "doc", etc.
	Path          string
	DataProductID string
	Concepts      []string
	Topics        []string
}

func toStringSlice(val any) []string {
	s, ok := val.([]string)
	if ok {
		return s
	}
	// handle []any
	anys, ok := val.([]any)
	if !ok {
		return nil
	}
	ss := make([]string, len(anys))
	for i, v := range anys {
		ss[i], _ = v.(string)
	}
	return ss
}

func UpsertCatalogAsset(db *sql.DB, asset CatalogAsset) error {
	_, err := db.Exec(`
        INSERT INTO assets (entity_id, asset_type, path, data_product_id, concepts, topics)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (entity_id, asset_type, path) DO UPDATE
        SET data_product_id=$4, concepts=$5, topics=$6
    `, asset.EntityID, asset.AssetType, asset.Path, asset.DataProductID, pq.Array(asset.Concepts), pq.Array(asset.Topics))
	return err
}

func RegisterAllCatalogOutputs(db *sql.DB, kind string, manifest map[string]any, paths ...string) error {
	if manifest == nil {
		return errors.New("manifest cannot be nil")
	}

	entityID, ok := manifest["entity_id"].(string)
	if !ok || strings.TrimSpace(entityID) == "" {
		return fmt.Errorf("manifest missing entity_id for asset kind %q", kind)
	}
	entityID = strings.TrimSpace(entityID)

	dataProductID, ok := manifest["data_product_id"].(string)
	if !ok || strings.TrimSpace(dataProductID) == "" {
		return fmt.Errorf("manifest missing data_product_id for asset kind %q", kind)
	}
	dataProductID = strings.TrimSpace(dataProductID)

	for _, path := range paths {
		asset := CatalogAsset{
			EntityID:      entityID,
			AssetType:     kind,
			Path:          path,
			DataProductID: dataProductID,
			Concepts:      toStringSlice(manifest["concepts"]),
			Topics:        toStringSlice(manifest["topics"]),
		}
		if err := UpsertCatalogAsset(db, asset); err != nil {
			return err
		}
	}
	return nil
}

// handleSAPBDCExtract handles extraction requests from SAP Business Data Cloud.
func (s *extractServer) handleSAPBDCExtract(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		FormationID   string `json:"formation_id"`
		SourceSystem  string `json:"source_system"`
		DataProductID string `json:"data_product_id,omitempty"`
		SpaceID       string `json:"space_id,omitempty"`
		Database      string `json:"database,omitempty"`
		ProjectID     string `json:"project_id"`
		SystemID      string `json:"system_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.FormationID == "" {
		http.Error(w, "formation_id is required", http.StatusBadRequest)
		return
	}
	if req.SourceSystem == "" {
		http.Error(w, "source_system is required", http.StatusBadRequest)
		return
	}
	if req.ProjectID == "" {
		req.ProjectID = "default"
	}
	if req.SystemID == "" {
		req.SystemID = "sap_bdc"
	}

	// Extract from SAP BDC
	nodes, edges, err := s.sapBDCIntegration.ExtractFromSAPBDC(
		r.Context(),
		req.FormationID,
		req.SourceSystem,
		req.DataProductID,
		req.SpaceID,
		req.Database,
		req.ProjectID,
		req.SystemID,
	)
	if err != nil {
		s.logger.Printf("Failed to extract from SAP BDC: %v", err)
		http.Error(w, fmt.Sprintf("Extraction failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Process the extracted graph through the normal pipeline
	// Save graph to persistence layers
	if s.graphPersistence != nil {
		if err := s.graphPersistence.SaveGraph(nodes, edges); err != nil {
			s.logger.Printf("failed to save SAP BDC graph: %v", err)
			http.Error(w, fmt.Sprintf("failed to save graph: %v", err), http.StatusInternalServerError)
			return
		}

		// Phase 10: Learn terminology from this extraction run
		if terminologyLearner := embeddings.GetGlobalTerminologyLearner(); terminologyLearner != nil {
			if err := terminologyLearner.LearnFromExtraction(r.Context(), nodes, edges); err != nil {
				s.logger.Printf("Warning: Failed to learn terminology from SAP BDC extraction: %v", err)
			}
		}
	}

	// Generate embeddings for tables and columns
	if s.vectorPersistence != nil {
		ctx := r.Context()
		for _, node := range nodes {
			if node.Type == "table" {
				relationalEmbedding, semanticEmbedding, err := embeddings.GenerateTableEmbedding(ctx, node)
				if err == nil {
					metadata := map[string]any{
						"artifact_type":  "table",
						"artifact_id":    node.ID,
						"label":          node.Label,
						"properties":     node.Props,
						"project_id":     req.ProjectID,
						"system_id":      req.SystemID,
						"graph_node_id":  node.ID,
						"created_at":     time.Now().UTC().Format(time.RFC3339Nano),
						"table_name":     node.Label,
						"embedding_type": "relational_transformer",
						"source":         "sap_bdc",
					}

					key := fmt.Sprintf("table:%s", node.ID)
					if err := s.vectorPersistence.SaveVector(key, relationalEmbedding, metadata); err != nil {
						s.logger.Printf("failed to save SAP BDC table embedding %q: %v", node.Label, err)
					}

					if len(semanticEmbedding) > 0 {
						semanticMetadata := make(map[string]any)
						for k, v := range metadata {
							semanticMetadata[k] = v
						}
						semanticMetadata["embedding_type"] = "sap_rpt_semantic"
						semanticKey := fmt.Sprintf("table_semantic:%s", node.ID)
						if err := s.vectorPersistence.SaveVector(semanticKey, semanticEmbedding, semanticMetadata); err != nil {
							s.logger.Printf("failed to save SAP BDC semantic table embedding %q: %v", node.Label, err)
						}
					}
				}
			}
		}
	}

	// Return response
	handlers.WriteJSON(w, http.StatusOK, map[string]any{
		"success": true,
		"nodes":   nodes,
		"edges":   edges,
		"count": map[string]int{
			"nodes": len(nodes),
			"edges": len(edges),
		},
		"source": "sap_bdc",
	})
}

func (s *extractServer) handleTerminologyDomains(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	learner := embeddings.GetGlobalTerminologyLearner()
	if learner == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "terminology learner unavailable"})
		return
	}

	domains, err := learner.GetLearnedDomains(r.Context())
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("failed to retrieve domains: %v", err)})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{"domains": domains})
}

func (s *extractServer) handleTerminologyRoles(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	learner := embeddings.GetGlobalTerminologyLearner()
	if learner == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "terminology learner unavailable"})
		return
	}

	roles, err := learner.GetLearnedRoles(r.Context())
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("failed to retrieve roles: %v", err)})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{"roles": roles})
}

func (s *extractServer) handleTerminologyPatterns(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	learner := embeddings.GetGlobalTerminologyLearner()
	if learner == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "terminology learner unavailable"})
		return
	}

	patterns, err := learner.GetLearnedPatterns(r.Context())
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("failed to retrieve patterns: %v", err)})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{"patterns": patterns})
}

func (s *extractServer) handleTerminologyLearn(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	learner := embeddings.GetGlobalTerminologyLearner()
	if learner == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "terminology learner unavailable"})
		return
	}

	if err := learner.LoadTerminology(r.Context()); err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("failed to reload terminology: %v", err)})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{"status": "reload_triggered"})
}

func (s *extractServer) handleTerminologyEvolution(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, map[string]any{"status": "not_tracked"})
}

func (s *extractServer) handleAnalyzeColumnSemantics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.semanticSchemaAnalyzer == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "semantic analyzer unavailable"})
		return
	}

	var req struct {
		ColumnName   string         `json:"column_name"`
		ColumnType   string         `json:"column_type"`
		TableName    string         `json:"table_name"`
		TableContext map[string]any `json:"table_context"`
		DomainID     string         `json:"domain_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}

	if req.ColumnName == "" || req.TableName == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "column_name and table_name are required"})
		return
	}

	analysis, err := s.semanticSchemaAnalyzer.AnalyzeColumnSemantics(
		r.Context(),
		req.ColumnName,
		req.ColumnType,
		req.TableName,
		req.TableContext,
		req.DomainID,
	)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("analysis failed: %v", err)})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, analysis)
}

func (s *extractServer) handleAnalyzeDataLineage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.semanticSchemaAnalyzer == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "semantic analyzer unavailable"})
		return
	}

	var req struct {
		SourceTable   string   `json:"source_table"`
		TargetTable   string   `json:"target_table"`
		SourceColumns []string `json:"source_columns"`
		TargetColumns []string `json:"target_columns"`
		SQL           string   `json:"sql"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}

	if req.SourceTable == "" || req.TargetTable == "" {
		handlers.WriteJSON(w, http.StatusBadRequest, map[string]string{"error": "source_table and target_table are required"})
		return
	}

	analysis, err := s.semanticSchemaAnalyzer.AnalyzeDataLineage(
		r.Context(),
		req.SourceTable,
		req.TargetTable,
		req.SourceColumns,
		req.TargetColumns,
		req.SQL,
	)
	if err != nil {
		handlers.WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("analysis failed: %v", err)})
		return
	}

	handlers.WriteJSON(w, http.StatusOK, analysis)
}

// populateExecutionTracking creates Execution nodes for Control-M jobs and SQL queries
func (s *extractServer) populateExecutionTracking(ctx context.Context, req *graphRequest, allControlMJobs []integrations.ControlMJob, nodes []graph.Node, edges []graph.Edge) {
	if s.neo4jPersistence == nil {
		return
	}

	extractionStartTime := time.Now()

	// Create execution node for the overall extraction
	extractionID := fmt.Sprintf("extraction:%s:%s:%d", req.ProjectID, req.SystemID, extractionStartTime.Unix())
	if err := s.neo4jPersistence.CreateExecution(
		ctx,
		extractionID,
		"extraction",
		req.ProjectID,
		"completed",
		extractionStartTime,
		map[string]any{
			"project_id":     req.ProjectID,
			"system_id":      req.SystemID,
			"node_count":     len(nodes),
			"edge_count":     len(edges),
			"control_m_jobs": len(allControlMJobs),
			"sql_queries":    len(req.SqlQueries),
			"hive_ddls":      len(req.HiveDDLs),
		},
	); err != nil {
		s.logger.Printf("failed to create extraction execution node: %v", err)
	}

	// Create execution nodes for Control-M jobs
	for _, job := range allControlMJobs {
		jobID := fmt.Sprintf("control-m:%s", job.JobName)
		executionID := fmt.Sprintf("execution:control-m:%s:%d", job.JobName, extractionStartTime.Unix())

		if err := s.neo4jPersistence.CreateExecution(
			ctx,
			executionID,
			"control-m-job",
			jobID,
			"completed",
			extractionStartTime,
			job.Properties(),
		); err != nil {
			s.logger.Printf("failed to create execution node for Control-M job %s: %v", job.JobName, err)
			continue
		}

		// Create execution metrics
		metrics := map[string]any{
			"job_name":        job.JobName,
			"application":     job.Application,
			"sub_application": job.SubApplication,
			"host":            job.Host,
			"task_type":       job.TaskType,
		}
		if err := s.neo4jPersistence.CreateExecutionMetrics(ctx, executionID, metrics); err != nil {
			s.logger.Printf("failed to create execution metrics for %s: %v", executionID, err)
		}
	}

	// Create execution nodes for SQL queries
	for i, sql := range req.SqlQueries {
		h := sha256.New()
		h.Write([]byte(sql))
		sqlQueryID := fmt.Sprintf("sql:%x", h.Sum(nil))
		executionID := fmt.Sprintf("execution:sql:%d:%d", i, extractionStartTime.Unix())

		if err := s.neo4jPersistence.CreateExecution(
			ctx,
			executionID,
			"sql-query",
			sqlQueryID,
			"completed",
			extractionStartTime,
			map[string]any{
				"sql":        sql,
				"query_hash": sqlQueryID,
			},
		); err != nil {
			s.logger.Printf("failed to create execution node for SQL query %d: %v", i, err)
		}
	}
}

// populateDataQualityMetrics creates QualityIssue nodes for data quality problems
func (s *extractServer) populateDataQualityMetrics(ctx context.Context, interpretation processing.MetricsInterpretation, nodes []graph.Node) {
	if s.neo4jPersistence == nil {
		return
	}

	// Create quality issues for each issue identified
	for i, issue := range interpretation.Issues {
		issueID := fmt.Sprintf("quality-issue:%d:%d", i, time.Now().Unix())

		// Determine severity based on quality level
		severity := "medium"
		if interpretation.QualityLevel == "critical" {
			severity = "critical"
		} else if interpretation.QualityLevel == "poor" {
			severity = "high"
		} else if interpretation.QualityLevel == "fair" {
			severity = "medium"
		} else {
			severity = "low"
		}

		// Determine issue type from issue description
		issueType := "data_quality"
		if strings.Contains(strings.ToLower(issue), "entropy") {
			issueType = "schema_diversity"
		} else if strings.Contains(strings.ToLower(issue), "kl divergence") {
			issueType = "type_distribution"
		} else if strings.Contains(strings.ToLower(issue), "column count") {
			issueType = "data_completeness"
		}

		// Link to root node or project node if available
		entityID := ""
		for _, node := range nodes {
			if node.Type == "project" || node.Type == "system" {
				entityID = node.ID
				break
			}
		}

		if err := s.neo4jPersistence.CreateQualityIssue(
			ctx,
			issueID,
			entityID,
			issueType,
			severity,
			issue,
			map[string]any{
				"quality_score":    interpretation.QualityScore,
				"quality_level":    interpretation.QualityLevel,
				"metadata_entropy": interpretation.MetadataEntropy,
				"kl_divergence":    interpretation.KLDivergence,
				"column_count":     interpretation.ColumnCount,
			},
		); err != nil {
			s.logger.Printf("failed to create quality issue node: %v", err)
		}
	}

	// Create quality metric for the overall extraction
	if len(nodes) > 0 {
		entityID := ""
		for _, node := range nodes {
			if node.Type == "project" || node.Type == "system" {
				entityID = node.ID
				break
			}
		}

		if entityID != "" {
			metricID := fmt.Sprintf("quality-metric:%s:%d", entityID, time.Now().Unix())
			// Store as PerformanceMetric with metric_type="quality_score"
			if err := s.neo4jPersistence.CreatePerformanceMetric(
				ctx,
				metricID,
				entityID,
				"quality_score",
				interpretation.QualityScore,
				map[string]any{
					"quality_level":    interpretation.QualityLevel,
					"metadata_entropy": interpretation.MetadataEntropy,
					"kl_divergence":    interpretation.KLDivergence,
					"column_count":     interpretation.ColumnCount,
				},
			); err != nil {
				s.logger.Printf("failed to create quality metric: %v", err)
			}
		}
	}
}

// populatePerformanceMetrics creates PerformanceMetric nodes for query and batch processing performance
func (s *extractServer) populatePerformanceMetrics(ctx context.Context, req *graphRequest, nodes []graph.Node) {
	if s.neo4jPersistence == nil || s.metricsCollector == nil {
		return
	}

	metrics := s.metricsCollector.GetMetrics()

	// Create performance metrics for Neo4j batch processing
	if neo4jMetrics, ok := metrics["neo4j_batch"].(map[string]interface{}); ok {
		if totalNodes, ok := neo4jMetrics["total_nodes"].(int64); ok && totalNodes > 0 {
			entityID := req.ProjectID
			if entityID == "" {
				entityID = req.SystemID
			}

			metricID := fmt.Sprintf("perf-metric:neo4j-batch:%d", time.Now().Unix())
			// Parse duration string (e.g., "100ms")
			var avgBatchTimeMs float64
			if avgBatchTimeStr, ok := neo4jMetrics["avg_batch_time"].(string); ok {
				if d, err := time.ParseDuration(avgBatchTimeStr); err == nil {
					avgBatchTimeMs = float64(d.Milliseconds())
				}
			}

			if err := s.neo4jPersistence.CreatePerformanceMetric(
				ctx,
				metricID,
				entityID,
				"neo4j_batch_time",
				avgBatchTimeMs,
				map[string]any{
					"total_batches":  neo4jMetrics["total_batches"],
					"total_nodes":    totalNodes,
					"total_edges":    neo4jMetrics["total_edges"],
					"avg_batch_size": neo4jMetrics["avg_batch_size"],
					"batch_errors":   neo4jMetrics["batch_errors"],
				},
			); err != nil {
				s.logger.Printf("failed to create Neo4j batch performance metric: %v", err)
			}
		}
	}

	// Create performance metrics for validation
	if validationMetrics, ok := metrics["validation"].(map[string]interface{}); ok {
		entityID := req.ProjectID
		if entityID == "" {
			entityID = req.SystemID
		}

		metricID := fmt.Sprintf("perf-metric:validation:%d", time.Now().Unix())
		// Parse duration string
		var validationTimeMs float64
		if validationTimeStr, ok := validationMetrics["avg_validation_time"].(string); ok {
			if d, err := time.ParseDuration(validationTimeStr); err == nil {
				validationTimeMs = float64(d.Milliseconds())
			}
		}

		if err := s.neo4jPersistence.CreatePerformanceMetric(
			ctx,
			metricID,
			entityID,
			"validation_time",
			validationTimeMs,
			map[string]any{
				"total_validated": validationMetrics["total_validated"],
				"nodes_validated": validationMetrics["nodes_validated"],
				"edges_validated": validationMetrics["edges_validated"],
				"nodes_rejected":  validationMetrics["nodes_rejected"],
				"edges_rejected":  validationMetrics["edges_rejected"],
			},
		); err != nil {
			s.logger.Printf("failed to create validation performance metric: %v", err)
		}
	}

	// Create performance metrics for node/edge counts
	entityID := req.ProjectID
	if entityID == "" {
		entityID = req.SystemID
	}

	if entityID != "" {
		nodeCountMetricID := fmt.Sprintf("perf-metric:node-count:%d", time.Now().Unix())
		if err := s.neo4jPersistence.CreatePerformanceMetric(
			ctx,
			nodeCountMetricID,
			entityID,
			"node_count",
			float64(len(nodes)),
			map[string]any{
				"extraction_time": time.Now().UTC().Format(time.RFC3339Nano),
			},
		); err != nil {
			s.logger.Printf("failed to create node count metric: %v", err)
		}
	}
}

// handleImprovementsMetrics returns metrics for all 6 improvements
func (s *extractServer) handleImprovementsMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handlers.WriteJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	if s.metricsCollector == nil {
		handlers.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "Metrics collector not initialized",
		})
		return
	}

	metrics := s.metricsCollector.GetMetrics()
	handlers.WriteJSON(w, http.StatusOK, metrics)
}

// startExplorer starts the catalog explorer interactive shell
func (s *extractServer) startExplorer() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Welcome to the Catalog Explorer!")

	for {
		fmt.Print("> ")
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)

		s.handleExplorerCommand(text)
	}
}

// handleExplorerCommand handles commands in the catalog explorer
func (s *extractServer) handleExplorerCommand(command string) {
	args := strings.Split(command, " ")
	cmd := args[0]

	switch cmd {
	case "help":
		fmt.Println("Available commands:")
		fmt.Println("  projects - list all projects")
		fmt.Println("  systems - list all systems")
		fmt.Println("  isystems - list all information systems")
		fmt.Println("  exit - exit the explorer")
	case "projects":
		for _, p := range s.catalog.Projects {
			fmt.Printf("- %s (%s)\n", p.Name, p.ID)
		}
	case "systems":
		for _, sys := range s.catalog.Systems {
			fmt.Printf("- %s (%s)\n", sys.Name, sys.ID)
		}
	case "isystems":
		for _, is := range s.catalog.InformationSystems {
			fmt.Printf("- %s (%s)\n", is.Name, is.ID)
		}
	case "exit":
		os.Exit(0)
	default:
		fmt.Println("Unknown command. Type 'help' for a list of commands.")
	}
}

// replicateSchema replicates the schema to various persistence layers
func (s *extractServer) replicateSchema(ctx context.Context, nodes []graph.Node, edges []graph.Edge) {
	if len(nodes) == 0 && len(edges) == 0 {
		return
	}

	// Improvement 1: Add data validation before storage
	validationStart := time.Now()
	validationResult := utils.ValidateGraph(nodes, edges, s.logger)
	validationDuration := time.Since(validationStart)
	
	// Record validation metrics
	if s.metricsCollector != nil {
		s.metricsCollector.RecordValidation(validationResult, validationDuration)
	}
	
	if !validationResult.Valid {
		s.logger.Printf("WARNING: Graph validation found %d errors, %d warnings. Filtering invalid data...", 
			len(validationResult.Errors), len(validationResult.Warnings))
		
		// Filter out invalid nodes and edges
		nodes = utils.FilterValidNodes(nodes, validationResult)
		edges = utils.FilterValidEdges(edges, validationResult)
		
		s.logger.Printf("After filtering: %d nodes, %d edges remain", len(nodes), len(edges))
	}

	// Store validation metrics for monitoring
	if validationResult.Metrics.ValidationErrors > 0 {
		s.logger.Printf("Validation metrics: %d nodes rejected, %d edges rejected", 
			validationResult.Metrics.NodesRejected, validationResult.Metrics.EdgesRejected)
	}

	if s.tablePersistence != nil {
		// Improvement 2: Add retry logic for storage operations
		retryStart := time.Now()
		retrySuccess := true
		if err := utils.RetryPostgresOperation(ctx, func() error {
			return schema.ReplicateSchemaToSQLite(s.tablePersistence, nodes, edges)
		}, s.logger); err != nil {
			retrySuccess = false
			s.logger.Printf("failed to replicate schema to sqlite after retries: %v", err)
		}
		if s.metricsCollector != nil {
			s.metricsCollector.RecordRetry(retrySuccess, time.Since(retryStart))
		}
	}

	// Note: Redis, HANA, and Postgres replication would go here if those fields exist
	// For now, we'll skip them to avoid compilation errors
}
