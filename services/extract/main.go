package main

import (
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
	"sort"
	"strconv"
	"strings"
	"time"

	_ "github.com/Chahine-tech/sql-parser-go/pkg/parser"
	_ "github.com/SAP/go-hdb/driver"
	"github.com/lib/pq"
	extractpb "github.com/plturrell/aModels/services/extract/gen/extractpb"

	"github.com/plturrell/aModels/services/extract/internal/config"
	handlers "github.com/plturrell/aModels/services/extract/internal/handlers"
	"github.com/plturrell/aModels/services/extract/internal/processing"
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
		deepAgentsClient: NewDeepAgentsClient(logger),
		
		// Domain detector for associating extracted data with domains
		domainDetector: NewDomainDetector(os.Getenv("LOCALAI_URL"), logger),
	}

	// Create persistence layer
	var graphPersistences []GraphPersistence
	var neo4jPersistence *Neo4jPersistence
	var realTimeGleanExporter *RealTimeGleanExporter

	if server.neo4jURI != "" {
		var err error
		neo4jPersistence, err = NewNeo4jPersistence(server.neo4jURI, server.neo4jUsername, server.neo4jPassword)
		if err != nil {
			logger.Fatalf("failed to create neo4j persistence: %v", err)
		}
		// Verify connectivity
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := neo4jPersistence.driver.VerifyConnectivity(ctx); err != nil {
			logger.Fatalf("failed to connect to neo4j: %v", err)
		}
		graphPersistences = append(graphPersistences, neo4jPersistence)
		server.neo4jPersistence = neo4jPersistence
		logger.Println("connected to neo4j")
	}

	var gleanPersistence *GleanPersistence
	if exportDir := strings.TrimSpace(os.Getenv("GLEAN_EXPORT_DIR")); exportDir != "" {
		predicatePrefix := strings.TrimSpace(os.Getenv("GLEAN_PREDICATE_PREFIX"))
		var err error
		gleanPersistence, err = NewGleanPersistence(exportDir, predicatePrefix, logger)
		if err != nil {
			logger.Fatalf("failed to create glean persistence: %v", err)
		}
		graphPersistences = append(graphPersistences, gleanPersistence)
		logger.Printf("glean export enabled (dir=%s, prefix=%s)", gleanPersistence.ExportDir(), gleanPersistence.PredicatePrefix())

		// Initialize real-time Glean exporter if enabled
		dbName := strings.TrimSpace(os.Getenv("GLEAN_DB_NAME"))
		schemaPath := gleanPersistence.schemaPath
		realTimeGleanExporter = NewRealTimeGleanExporter(gleanPersistence, dbName, schemaPath, logger)
		server.realTimeGleanExporter = realTimeGleanExporter
	}

	server.graphPersistence = newCompositeGraphPersistence(graphPersistences...)

	if hr := newHANASchemaReplication(logger); hr != nil {
		server.hanaReplication = hr
		logger.Println("hana schema replication configured")
		defer hr.Close()
	}

	if pr := newPostgresSchemaReplication(logger); pr != nil {
		server.postgresReplication = pr
		logger.Println("postgres schema replication configured")
		defer pr.Close()
	}

	// Create document persistence layer
	if server.docStorePath != "" {
		docPersistence, err := NewFilePersistence(server.docStorePath)
		if err != nil {
			logger.Fatalf("failed to create file persistence: %v", err)
		}
		server.docPersistence = docPersistence
		logger.Println("document persistence enabled")
	}

	// Create table persistence layer
	if server.sqlitePath != "" {
		sqlitePersistence, err := NewSQLitePersistence(server.sqlitePath)
		if err != nil {
			logger.Fatalf("failed to create sqlite persistence: %v", err)
		}
		server.tablePersistence = sqlitePersistence
		logger.Println("sqlite persistence enabled")
	}

	// Create vector persistence layers (Phase 2 & 3: pgvector and OpenSearch integration)
	var vectorStores []VectorPersistence
	var primaryStore VectorPersistence

	// Initialize pgvector (primary store for structured queries)
	if pgVectorDSN := os.Getenv("POSTGRES_VECTOR_DSN"); pgVectorDSN != "" {
		pgVectorPersistence, err := NewPgVectorPersistence(pgVectorDSN, logger)
		if err != nil {
			logger.Printf("failed to initialize pgvector persistence: %v", err)
		} else {
			primaryStore = pgVectorPersistence
			vectorStores = append(vectorStores, pgVectorPersistence)
			logger.Println("pgvector persistence enabled")
		}
	}

	// Initialize OpenSearch (secondary store for semantic/hybrid search)
	if opensearchURL := os.Getenv("OPENSEARCH_URL"); opensearchURL != "" {
		opensearchPersistence, err := NewOpenSearchPersistence(opensearchURL, logger)
		if err != nil {
			logger.Printf("failed to initialize OpenSearch persistence: %v", err)
		} else {
			vectorStores = append(vectorStores, opensearchPersistence)
			logger.Println("OpenSearch persistence enabled")
		}
	}

	// Initialize Redis (fallback/cache store)
	if server.redisAddr != "" {
		redisPersistence, err := NewRedisPersistence(server.redisAddr, server.redisPassword, server.redisDB)
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
		var secondary []VectorPersistence
		for _, store := range vectorStores {
			if store != primaryStore {
				secondary = append(secondary, store)
			}
		}
		server.vectorPersistence = NewCompositeVectorPersistence(primaryStore, secondary, logger)
		logger.Println("composite vector persistence enabled")
	} else if len(vectorStores) == 1 {
		// Single store
		server.vectorPersistence = vectorStores[0]
		logger.Println("single vector persistence enabled")
	} else {
		logger.Println("no vector persistence configured")
	}

	// Initialize Orchestration chain matcher (Phase 2 integration)
	chainMatcher := NewOrchestrationChainMatcher(logger)
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
	server.embeddingCache = NewEmbeddingCache(cacheMaxSize, cacheTTL, logger)

	batchSize := parseIntEnv(os.Getenv("EMBEDDING_BATCH_SIZE"), 10)
	server.batchEmbeddingGen = NewBatchEmbeddingGenerator(logger, server.embeddingCache, batchSize)
	logger.Printf("Embedding cache initialized (max_size=%d, ttl=%v, batch_size=%d)", cacheMaxSize, cacheTTL, batchSize)

	// Initialize training data collector (Phase 4 full model utilization)
	trainingDataPath := os.Getenv("SAP_RPT_TRAINING_DATA_PATH")
	if trainingDataPath == "" {
		trainingDataPath = "./training_data/sap_rpt_classifications.json"
	}
	server.trainingDataCollector = NewTrainingDataCollector(trainingDataPath, logger)
	if os.Getenv("COLLECT_TRAINING_DATA") == "true" {
		logger.Printf("Training data collection enabled (path=%s)", trainingDataPath)
	}

	// Initialize model monitor (Phase 5 advanced capabilities)
	metricsPath := os.Getenv("MODEL_METRICS_PATH")
	if metricsPath == "" {
		metricsPath = "./training_data/model_metrics.json"
	}
	server.modelMonitor = NewModelMonitor(metricsPath, logger)
	if os.Getenv("MODEL_MONITORING_ENABLED") == "true" {
		logger.Printf("Model monitoring enabled (path=%s)", metricsPath)
	}

	// Initialize multi-modal extractor (Phase 6 unified integration)
	server.multiModalExtractor = NewMultiModalExtractor(logger)
	if os.Getenv("USE_MULTIMODAL_EXTRACTION") == "true" {
		logger.Printf("Multi-modal extraction enabled (OCR: %v)", os.Getenv("USE_DEEPSEEK_OCR") == "true")
	}

	// Phase 8.1: Initialize semantic schema analyzer
	server.semanticSchemaAnalyzer = NewSemanticSchemaAnalyzer(logger)
	logger.Println("Semantic schema analyzer initialized (Phase 8.1)")

	// Phase 10: Initialize terminology learner with LNN
	terminologyStore := NewNeo4jTerminologyStore(server.neo4jPersistence, logger)
	terminologyLearner := NewTerminologyLearner(terminologyStore, logger)
	
	// Load existing terminology from store
	if err := terminologyLearner.LoadTerminology(context.Background()); err != nil {
		logger.Printf("Warning: Failed to load existing terminology: %v", err)
	}
	
	// Set global terminology learner for embedding enhancement
	SetGlobalTerminologyLearner(terminologyLearner)
	
	// Wire terminology learner to components
	server.semanticSchemaAnalyzer.SetTerminologyLearner(terminologyLearner)
	logger.Println("Terminology learner initialized (Phase 10)")

	// Phase 9.2: Initialize self-healing system
	server.selfHealingSystem = NewSelfHealingSystem(logger)
	logger.Println("Self-healing system initialized (Phase 9.2)")
	
	// Register health monitors for critical services
	if server.neo4jPersistence != nil {
		server.selfHealingSystem.RegisterHealthMonitor(
			"neo4j",
			30*time.Second,
			func() error {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				return server.neo4jPersistence.driver.VerifyConnectivity(ctx)
			},
		)
	}

	server.telemetryOperation = cfg.Telemetry.Operation

	if cfg.Telemetry.Enabled && cfg.Telemetry.Address != "" {
		telemetryClient, err := newTelemetryClient(context.Background(), telemetryConfig{
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

	catalog, err := NewCatalog("catalog.json")
	if err != nil {
		logger.Fatalf("failed to create catalog: %v", err)
	}
	server.catalog = catalog

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

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", server.handleHealthz)
	mux.HandleFunc("/extract", server.handleExtract)
	mux.HandleFunc("/generate/training", server.handleGenerateTraining)
	mux.HandleFunc("/knowledge-graph", server.handleGraph)                           // Main knowledge graph processing endpoint
	mux.HandleFunc("/graph", server.handleGraph)                                     // Legacy alias for backward compatibility
	mux.HandleFunc("/knowledge-graph/query", server.handleNeo4jQuery)                // Neo4j Cypher query endpoint
	mux.HandleFunc("/workflow/petri-to-langgraph", server.handlePetriNetToLangGraph) // Convert Petri net to LangGraph
	mux.HandleFunc("/workflow/petri-to-langgraph-advanced", server.handlePetriNetToAdvancedLangGraph) // Convert Petri net to advanced LangGraph (Phase 7.3)
	mux.HandleFunc("/workflow/petri-to-agentflow", server.handlePetriNetToAgentFlow) // Convert Petri net to AgentFlow
	// Phase 10: Terminology learning endpoints
	mux.HandleFunc("/terminology/domains", server.handleTerminologyDomains)     // List learned domains
	mux.HandleFunc("/terminology/roles", server.handleTerminologyRoles)          // List learned roles
	mux.HandleFunc("/terminology/patterns", server.handleTerminologyPatterns)    // List learned naming patterns
	mux.HandleFunc("/terminology/learn", server.handleTerminologyLearn)          // Trigger manual learning
	mux.HandleFunc("/terminology/evolution", server.handleTerminologyEvolution)  // Terminology evolution over time
	mux.HandleFunc("/semantic/analyze-column", server.handleAnalyzeColumnSemantics) // Semantic column analysis (Phase 8.1)
	mux.HandleFunc("/semantic/analyze-lineage", server.handleAnalyzeDataLineage) // Semantic data lineage analysis (Phase 8.1)
	mux.HandleFunc("/health/status", server.handleHealthStatus) // Health status for all services (Phase 9.2)
	mux.HandleFunc("/knowledge-graph/queries", server.handleGraphQueryHelpers)       // Get common graph query helpers
	mux.HandleFunc("/knowledge-graph/search", server.handleVectorSearch)             // Vector similarity search (RAG)
	mux.HandleFunc("/knowledge-graph/embed", server.handleGenerateEmbedding)         // Generate embedding for text
	mux.HandleFunc("/knowledge-graph/embed/", server.handleGetEmbedding)             // Get embedding by key
	mux.HandleFunc("/training-data/stats", server.handleTrainingDataStats)           // Get training data statistics (Phase 4)
	mux.HandleFunc("/training-data/export", server.handleExportTrainingData)         // Export training data (Phase 4)
	mux.HandleFunc("/model/metrics", server.handleModelMetrics)                      // Get model performance metrics (Phase 5)
	mux.HandleFunc("/model/uncertain", server.handleUncertainPredictions)            // Get uncertain predictions for review (Phase 5)
	mux.HandleFunc("/catalog/projects", server.handleGetProjects)
	mux.HandleFunc("/catalog/projects/add", server.handleAddProject)
	mux.HandleFunc("/catalog/systems", server.handleGetSystems)
	mux.HandleFunc("/catalog/systems/add", server.handleAddSystem)
	mux.HandleFunc("/catalog/information-systems", server.handleGetInformationSystems)
	mux.HandleFunc("/catalog/information-systems/add", server.handleAddInformationSystem)
	mux.HandleFunc("/ui", server.handleWebUI)

	if *explorer {
		server.startExplorer()
	} else {
		addr := ":" + cfg.Server.Port
		logger.Printf("extract service listening on %s (proxying %s)", addr, server.langextractURL)
		if err := http.ListenAndServe(addr, mux); err != nil && !errors.Is(err, http.ErrServerClosed) {
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
	docPersistence DocumentPersistence

	// DeepAgents client
	deepAgentsClient *DeepAgentsClient
	
	// Domain detector for associating extracted data with domains
	domainDetector *DomainDetector

	tablePersistence      TablePersistence
	vectorPersistence     VectorPersistence
	graphPersistence      GraphPersistence
	neo4jPersistence      *Neo4jPersistence      // Direct Neo4j access for queries
	realTimeGleanExporter *RealTimeGleanExporter // Real-time Glean synchronization
	flight                *extractFlightServer
	semanticSchemaAnalyzer *SemanticSchemaAnalyzer // Phase 8.1: Semantic schema understanding
	selfHealingSystem      *SelfHealingSystem      // Phase 9.2: Self-healing system
	hanaReplication       *hanaReplication
	postgresReplication   *postgresReplication

	telemetry          *telemetryClient
	telemetryOperation string
	catalog            *Catalog

	// DeepAgents client (enabled by default, 10/10 integration)
	deepAgentsClient *DeepAgentsClient

	// Orchestration chain matcher (for Phase 2 integration)
	chainMatcher *OrchestrationChainMatcher

	// Embedding cache and batch generator (for Phase 3 optimization)
	embeddingCache    *EmbeddingCache
	batchEmbeddingGen *BatchEmbeddingGenerator

	// Training data collector (for Phase 4 full model utilization)
	trainingDataCollector *TrainingDataCollector

	// Model monitor (for Phase 5 advanced capabilities)
	modelMonitor *ModelMonitor

	// Multi-modal extractor (for Phase 6 unified integration)
	multiModalExtractor *MultiModalExtractor
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
	if s.deepAgentsClient != nil && s.deepAgentsClient.enabled {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		deepAgentsHealthy := s.deepAgentsClient.checkHealth(ctx)
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
		return nil
	}

	if latency < 0 {
		latency = 0
	}

	completed := started.Add(latency)

	record := telemetryRecord{
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

	return s.telemetry.Log(ctx, record)
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

type Node struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Label string         `json:"label"`
	Props map[string]any `json:"properties,omitempty"`
}

type Edge struct {
	SourceID string         `json:"source"`
	TargetID string         `json:"target"`
	Label    string         `json:"label"`
	Props    map[string]any `json:"properties,omitempty"`
}

type graphRequest struct {
	JSONTables          []string           `json:"json_tables"`
	HiveDDLs            []string           `json:"hive_ddls"`
	SqlQueries          []string           `json:"sql_queries"`
	ControlMFiles       []string           `json:"control_m_files"`
	IdealDistribution   map[string]float64 `json:"ideal_distribution"`
	ProjectID           string             `json:"project_id"`
	SystemID            string             `json:"system_id"`
	InformationSystemID string             `json:"information_system_id"`
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

	var nodes []Node
	var edges []Edge

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

	for i, ddl := range req.HiveDDLs {
		ddl = strings.TrimSpace(ddl)
		if ddl == "" {
			continue
		}

		parsed, err := parseHiveDDL(ctx, ddl)
		if err != nil {
			s.logger.Printf("failed to parse hive ddl #%d: %v", i+1, err)
			continue
		}

		ddlNodes, ddlEdges := ddlToGraph(parsed)
		nodes = append(nodes, ddlNodes...)
		edges = append(edges, ddlEdges...)
	}

	// Collect all Control-M jobs for Petri net conversion
	allControlMJobs := []ControlMJob{}
	controlMJobMap := make(map[string][]ControlMJob) // path -> jobs

	for _, path := range req.ControlMFiles {
		if strings.TrimSpace(path) == "" {
			continue
		}

		jobs, err := parseControlMXML(path)
		if err != nil {
			s.logger.Printf("failed to parse control-m xml file %q: %v", path, err)
			continue
		}

		allControlMJobs = append(allControlMJobs, jobs...)
		controlMJobMap[path] = jobs

		for _, job := range jobs {
			jobID := fmt.Sprintf("control-m:%s", job.JobName)
			nodes = append(nodes, Node{
				ID:    jobID,
				Type:  "control-m-job",
				Label: job.JobName,
				Props: job.Properties(),
			})

			if calendar := strings.TrimSpace(job.CalendarName); calendar != "" {
				calendarID := fmt.Sprintf("control-m-calendar:%s", calendar)
				nodes = append(nodes, Node{
					ID:    calendarID,
					Type:  "control-m-calendar",
					Label: calendar,
				})
				edges = append(edges, Edge{
					SourceID: calendarID,
					TargetID: jobID,
					Label:    "SCHEDULES",
				})
			}

			for _, inCond := range job.InConds {
				condID := fmt.Sprintf("control-m-cond:%s", inCond.Name)
				nodes = append(nodes, Node{
					ID:    condID,
					Type:  "control-m-condition",
					Label: inCond.Name,
					Props: inCond.Properties(),
				})
				edges = append(edges, Edge{
					SourceID: condID,
					TargetID: jobID,
					Label:    "BLOCKS",
					Props:    inCond.Properties(),
				})
			}

			for _, outCond := range job.OutConds {
				condID := fmt.Sprintf("control-m-cond:%s", outCond.Name)
				nodes = append(nodes, Node{
					ID:    condID,
					Type:  "control-m-condition",
					Label: outCond.Name,
					Props: outCond.Properties(),
				})
				edges = append(edges, Edge{
					SourceID: jobID,
					TargetID: condID,
					Label:    "RELEASES",
					Props:    outCond.Properties(),
				})
			}
		}
	}

	// Convert Control-M jobs to Petri net and add to knowledge graph
	if len(allControlMJobs) > 0 {
		// Map SQL queries to job names (if we can infer from context)
		// For now, we'll create a simple mapping
		sqlQueriesByJob := make(map[string][]string)

		// Try to match SQL queries to jobs based on table names or patterns
		// This is a simplified approach - in practice, you'd have better job-to-SQL mapping
		for i, sql := range req.SqlQueries {
			// Use a simple heuristic: assign SQL to jobs in order
			if i < len(allControlMJobs) {
				jobName := allControlMJobs[i].JobName
				sqlQueriesByJob[jobName] = append(sqlQueriesByJob[jobName], sql)
			}
		}

		// Convert Control-M to Petri net
		petriNetConverter := NewPetriNetConverter(s.logger)
		petriNet := petriNetConverter.ConvertControlMToPetriNet(allControlMJobs, sqlQueriesByJob)

		// Convert Petri net to graph nodes/edges
		petriNodes, petriEdges, petriRootID := petriNetConverter.PetriNetToGraphNodes(petriNet)
		nodes = append(nodes, petriNodes...)
		edges = append(edges, petriEdges...)

		// Link Petri net to root node
		if rootID != "" {
			edges = append(edges, Edge{
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

	for _, sql := range req.SqlQueries {
		sql = strings.TrimSpace(sql)
		if sql == "" {
			continue
		}

		lineage, err := parseSQL(sql)
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

					edges = append(edges, Edge{
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

				nodes = append(nodes, Node{ID: sourceColumnID, Type: "column", Label: sourceCol})
				nodes = append(nodes, Node{ID: targetColumnID, Type: "column", Label: targetCol})

				edges = append(edges, Edge{
					SourceID: sourceColumnID,
					TargetID: targetColumnID,
					Label:    "DATA_FLOW",
				})
			}
		}

		embedding, err := generateEmbedding(ctx, sql)
		if err != nil {
			s.logger.Printf("failed to generate embedding for sql query %q: %v", sql, err)
			continue
		}

		if s.vectorPersistence != nil {
			h := sha256.New()
			h.Write([]byte(sql))
			key := fmt.Sprintf("sql:%x", h.Sum(nil))

			// Create rich metadata for SQL query
			metadata := map[string]any{
				"artifact_type": "sql-query",
				"artifact_id":   key,
				"label":         sql,
				"project_id":    req.ProjectID,
				"system_id":     req.SystemID,
				"created_at":    time.Now().UTC().Format(time.RFC3339Nano),
				"sql":           sql,
			}

			if err := s.vectorPersistence.SaveVector(key, embedding, metadata); err != nil {
				s.logger.Printf("failed to save vector for sql query %q: %v", sql, err)
			}
		}
	}

	normResult := normalizeGraph(normalizationInput{
		Nodes:               nodes,
		Edges:               edges,
		ProjectID:           req.ProjectID,
		SystemID:            req.SystemID,
		InformationSystemID: req.InformationSystemID,
		Catalog:             s.catalog,
	})
	nodes = normResult.Nodes
	edges = normResult.Edges
	rootID := normResult.RootNodeID

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
		nodeMap := make(map[string]*Node)
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

	columnDtypes := make([]string, 0)
	for _, node := range nodes {
		if node.Type != "column" || node.Props == nil {
			continue
		}
		if dtype, ok := node.Props["type"].(string); ok && dtype != "" {
			columnDtypes = append(columnDtypes, dtype)
		}
	}

	metadataEntropy := calculateEntropy(columnDtypes)

	actualDistribution := make(map[string]float64)
	totalColumns := float64(len(columnDtypes))
	for _, dtype := range columnDtypes {
		actualDistribution[dtype]++
	}
	if totalColumns > 0 {
		for dtype, count := range actualDistribution {
			actualDistribution[dtype] = count / totalColumns
		}
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
	klDivergence := calculateKLDivergence(actualDistribution, idealDistribution)

	// Interpret metrics and determine actionable insights
	thresholds := processing.DefaultMetricsThresholds()
	interpretation := processing.InterpretMetrics(
		metadataEntropy,
		klDivergence,
		len(columnDtypes),
		actualDistribution,
		idealDistribution,
		thresholds,
		s.logger,
	)

	// Take action based on interpretation
	if interpretation.ShouldReject {
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
				"column_count":     len(columnDtypes),
			},
		})
		return
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
				nodes[i].Props["column_count"] = len(columnDtypes)
				// Store metrics timestamp for tracking over time
				nodes[i].Props["metrics_calculated_at"] = time.Now().UTC().Format(time.RFC3339Nano)
				break
			}
		}
	}

	s.replicateSchema(ctx, nodes, edges)

	// Export Petri net to catalog if available
	if len(allControlMJobs) > 0 {
		petriNetConverter := NewPetriNetConverter(s.logger)
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
			s.catalog.mu.Lock()
			if s.catalog.PetriNets == nil {
				s.catalog.PetriNets = make(map[string]any)
			}
			s.catalog.PetriNets[petriNet.ID] = catalogEntry
			s.catalog.mu.Unlock()

			// Save catalog to persist Petri net
			if err := s.catalog.Save(); err != nil {
				s.logger.Printf("failed to save Petri net to catalog: %v", err)
			} else {
				s.logger.Printf("Saved Petri net '%s' to catalog", petriNet.ID)
			}
		}
	}

	// Save graph to persistence layers (Neo4j, Glean, etc.)
	// Information theory metrics are included:
	// 1. In root node properties (accessible in Neo4j queries)
	// 2. In Glean export manifest (via glean_persistence.go)
	if s.graphPersistence != nil {
		if err := s.graphPersistence.SaveGraph(nodes, edges); err != nil {
			s.logger.Printf("failed to save graph: %v", err)
		}

		// Phase 10: Learn terminology from this extraction run (incremental learning)
		if terminologyLearner := GetGlobalTerminologyLearner(); terminologyLearner != nil {
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
	advancedExtractor := NewAdvancedExtractor(s.logger)
	
	// Phase 10: Wire terminology learner to advanced extractor
	if terminologyLearner := GetGlobalTerminologyLearner(); terminologyLearner != nil {
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
					edges = append(edges, Edge{
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
		tableNodes := []Node{}
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
				s.logger.Printf("batch embedding generation failed: %v, falling back to individual", err)
				goto individualTableEmbeddings
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

		tableEmbeddingsDone:
		}

		// Generate embeddings for columns (with batch processing and caching)
		columnNodes := []Node{}
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
				s.logger.Printf("batch column embedding generation failed: %v, falling back to individual", err)
				goto individualColumnEmbeddings
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

					cmdSemantic := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
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
			embedding, err := generateJobEmbedding(ctx, job)
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
				embedding, err := generateSequenceEmbedding(ctx, seq)
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

		// Generate embedding for Petri net (if available)
		if len(allControlMJobs) > 0 {
			petriNetConverter := NewPetriNetConverter(s.logger)
			sqlQueriesByJob := make(map[string][]string)
			for i, sql := range req.SqlQueries {
				if i < len(allControlMJobs) {
					jobName := allControlMJobs[i].JobName
					sqlQueriesByJob[jobName] = append(sqlQueriesByJob[jobName], sql)
				}
			}
			petriNet := petriNetConverter.ConvertControlMToPetriNet(allControlMJobs, sqlQueriesByJob)

			embedding, err := generatePetriNetEmbedding(ctx, petriNet)
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
	var deepAgentsAnalysis *AnalyzeGraphResponse
	if s.deepAgentsClient != nil && s.deepAgentsClient.enabled {
		graphSummary := FormatGraphSummary(nodes, edges, map[string]any{
			"score":  interpretation.QualityScore,
			"level":  interpretation.QualityLevel,
			"issues": interpretation.Issues,
		}, map[string]any{
			"metadata_entropy": metadataEntropy,
			"kl_divergence":    klDivergence,
			"column_count":     float64(len(columnDtypes)),
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
			"column_count":        len(columnDtypes),
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
		telemetryRecord := telemetryRecord{
			LibraryType: "layer4_extract",
			Operation:   "graph_processing",
			Input: map[string]any{
				"json_tables_count":     len(req.JSONTables),
				"hive_ddls_count":       len(req.HiveDDLs),
				"sql_queries_count":     len(req.SqlQueries),
				"control_m_files_count": len(req.ControlMFiles),
				"project_id":            req.ProjectID,
				"system_id":             req.SystemID,
				"information_system_id": req.InformationSystemID,
			},
			Output: map[string]any{
				"nodes_count":         len(nodes),
				"edges_count":         len(edges),
				"metadata_entropy":    metadataEntropy,
				"kl_divergence":       klDivergence,
				"actual_distribution": actualDistribution,
				"ideal_distribution":  idealDistribution,
				"column_count":        len(columnDtypes),
				"root_node_id":        rootID,
			},
			StartedAt:   started,
			CompletedAt: time.Now(),
			Latency:     time.Since(started),
		}
		if normResult.Stats != nil {
			telemetryRecord.Output["normalization_stats"] = normResult.Stats
		}
		if err := s.telemetry.Log(telemetryCtx, telemetryRecord); err != nil {
			s.logger.Printf("telemetry warning: %v", err)
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

	var p Project
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.catalog.mu.Lock()
	s.catalog.Projects = append(s.catalog.Projects, p)
	s.catalog.mu.Unlock()

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

	var sys System
	if err := json.NewDecoder(r.Body).Decode(&sys); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.catalog.mu.Lock()
	s.catalog.Systems = append(s.catalog.Systems, sys)
	s.catalog.mu.Unlock()

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

	s.catalog.mu.RLock()
	petriNetData, exists := s.catalog.PetriNets[req.PetriNetID]
	s.catalog.mu.RUnlock()

	if !exists {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]any{
			"error": fmt.Sprintf("Petri net '%s' not found in catalog", req.PetriNetID),
		})
		return
	}

	// Convert catalog data to PetriNet struct
	petriNetJSON, err := json.Marshal(petriNetData)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	var petriNet PetriNet
	if err := json.Unmarshal(petriNetJSON, &petriNet); err != nil {
		http.Error(w, fmt.Sprintf("failed to unmarshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to LangGraph workflow with semantic search enabled
	converter := NewWorkflowConverter(s.logger)

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

	s.catalog.mu.RLock()
	petriNetData, exists := s.catalog.PetriNets[req.PetriNetID]
	s.catalog.mu.RUnlock()

	if !exists {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]any{
			"error": fmt.Sprintf("Petri net '%s' not found in catalog", req.PetriNetID),
		})
		return
	}

	// Convert catalog data to PetriNet struct
	petriNetJSON, err := json.Marshal(petriNetData)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	var petriNet PetriNet
	if err := json.Unmarshal(petriNetJSON, &petriNet); err != nil {
		http.Error(w, fmt.Sprintf("failed to unmarshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to AgentFlow workflow with semantic search enabled
	converter := NewWorkflowConverter(s.logger)

	// Set Extract service URL for semantic search (use same service)
	baseURL := fmt.Sprintf("http://localhost:%s", os.Getenv("PORT"))
	if baseURL == "http://localhost:" {
		baseURL = "http://localhost:8081"
	}
	converter.SetExtractServiceURL(baseURL)

	agentFlowWorkflow := converter.ConvertPetriNetToAgentFlow(&petriNet)

	handlers.WriteJSON(w, http.StatusOK, agentFlowWorkflow)
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

	s.catalog.mu.RLock()
	petriNetData, exists := s.catalog.PetriNets[req.PetriNetID]
	s.catalog.mu.RUnlock()

	if !exists {
		handlers.WriteJSON(w, http.StatusNotFound, map[string]any{
			"error": fmt.Sprintf("Petri net '%s' not found in catalog", req.PetriNetID),
		})
		return
	}

	// Convert catalog data to PetriNet struct
	petriNetJSON, err := json.Marshal(petriNetData)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	var petriNet PetriNet
	if err := json.Unmarshal(petriNetJSON, &petriNet); err != nil {
		http.Error(w, fmt.Sprintf("failed to unmarshal petri net: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to advanced LangGraph workflow
	converter := NewAdvancedWorkflowConverter(s.logger)

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
			semanticQueryVector, err = generateSemanticEmbedding(ctx, request.Query)
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
	var results []VectorSearchResult
	var err error

	// For hybrid search, we need both embeddings
	if request.UseHybridSearch && os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true" {
		// Generate both embeddings if not already done
		if len(semanticQueryVector) == 0 && request.Query != "" {
			semanticQueryVector, _ = generateSemanticEmbedding(ctx, request.Query)
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
		return generateSQLEmbedding(ctx, query)
	case "table":
		node := Node{Label: query, Type: "table"}
		relational, _, err := generateTableEmbedding(ctx, node)
		return relational, err
	default:
		return generateSQLEmbedding(ctx, query)
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
func (s *extractServer) performHybridSearch(ctx context.Context, relationalVector, semanticVector []float32, queryText, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	// Search relational embeddings
	relationalResults, err1 := s.searchRelationalEmbeddings(relationalVector, artifactType, limit, threshold)
	if err1 != nil {
		s.logger.Printf("relational search failed: %v", err1)
		relationalResults = []VectorSearchResult{}
	}

	// Search semantic embeddings
	semanticResults, err2 := s.searchSemanticEmbeddings(semanticVector, artifactType, limit, threshold)
	if err2 != nil {
		s.logger.Printf("semantic search failed: %v", err2)
		semanticResults = []VectorSearchResult{}
	}

	// Fuse results intelligently
	fusedResults := s.fuseSearchResults(relationalResults, semanticResults, limit)

	return fusedResults, nil
}

// searchRelationalEmbeddings searches relational embeddings
func (s *extractServer) searchRelationalEmbeddings(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	// Search for embeddings with embedding_type = "relational_transformer"
	// We need to search all embeddings and filter by type
	results, err := s.vectorPersistence.SearchSimilar(queryVector, artifactType, limit*2, threshold)
	if err != nil {
		return nil, err
	}

	// Filter for relational embeddings
	relationalResults := []VectorSearchResult{}
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
func (s *extractServer) searchSemanticEmbeddings(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	// Search for embeddings with embedding_type = "sap_rpt_semantic"
	// We need to search all embeddings and filter by type
	results, err := s.vectorPersistence.SearchSimilar(queryVector, artifactType, limit*2, threshold)
	if err != nil {
		return nil, err
	}

	// Filter for semantic embeddings
	semanticResults := []VectorSearchResult{}
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
func (s *extractServer) fuseSearchResults(relationalResults, semanticResults []VectorSearchResult, limit int) []VectorSearchResult {
	// Create a map to deduplicate by artifact_id
	resultMap := make(map[string]VectorSearchResult)
	scoreMap := make(map[string]float32)

	// Process relational results
	for _, result := range relationalResults {
		key := result.ArtifactID
		if existing, exists := resultMap[key]; exists {
			// If already exists, boost score (weighted average)
			relationalWeight := 0.4
			semanticWeight := 0.6
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
			relationalWeight := 0.4
			semanticWeight := 0.6
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
	fusedResults := make([]VectorSearchResult, 0, len(resultMap))
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
		embedding, err = generateSQLEmbedding(ctx, request.Text)
	case "table":
		node := Node{Label: request.Text, Type: "table"}
		embedding, err = generateTableEmbedding(ctx, node)
	case "column":
		node := Node{Label: request.Text, Type: "column"}
		embedding, err = generateColumnEmbedding(ctx, node)
	default:
		// Default to SQL embedding
		embedding, err = generateSQLEmbedding(ctx, request.Text)
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

	var ocrResult *OCRResult
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
		ImagePath        string                 `json:"image_path"`
		ImageBase64      string                 `json:"image_base64"`
		TableName        string                 `json:"table_name"`
		Columns          []map[string]any       `json:"columns"`
		Text             string                 `json:"text"`
		Prompt           string                 `json:"prompt"`
		TrainingDataPath string                 `json:"training_data_path"`
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
		Text      string                 `json:"text"`
		ImagePath string                 `json:"image_path"`
		TableName string                 `json:"table_name"`
		Columns   []map[string]any       `json:"columns"`
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

	helpers := NewGraphQueryHelpers(s.logger)

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

	var is InformationSystem
	if err := json.NewDecoder(r.Body).Decode(&is); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.catalog.mu.Lock()
	s.catalog.InformationSystems = append(s.catalog.InformationSystems, is)
	s.catalog.mu.Unlock()

	if err := s.catalog.Save(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
}

func (s *extractServer) extractSchemaFromJSON(path string) ([]Node, []Edge, []map[string]any, error) {
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

	var nodes []Node
	var edges []Edge

	tableID := filepath.Base(path)
	tableNode := Node{
		ID:    tableID,
		Type:  "table",
		Label: tableID,
	}
	nodes = append(nodes, tableNode)

	for _, key := range columnNames {
		profile := columnProfiles[key]
		columnID := fmt.Sprintf("%s.%s", tableID, key)
		columnNode := Node{
			ID:    columnID,
			Type:  "column",
			Label: key,
			Props: profile.toProps(),
		}
		nodes = append(nodes, columnNode)

		edges = append(edges, Edge{
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

	return mapOrNil(props)
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
func validateGraph(nodes []Node, edges []Edge) []string {
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

func (s *extractServer) buildLangextractPayload(req extractRequest) (*langextractPayload, error) {
	var textOrDocs any
	if len(req.TextOrDocumentsRaw) > 0 {
		if err := json.Unmarshal(req.TextOrDocumentsRaw, &textOrDocs); err != nil {
			return nil, fmt.Errorf("text_or_documents: %w", err)
		}
	} else if len(req.Documents) > 0 {
		textOrDocs = req.Documents
	} else if strings.TrimSpace(req.Document) != "" {
		textOrDocs = req.Document
	} else {
		return nil, errors.New("document content is required")
	}

	prompt := strings.TrimSpace(req.PromptDescription)
	if prompt == "" {
		prompt = defaultPromptDescription
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
	}, nil
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

func groupExtractions(extractions []extractionResult) map[string][]string {
	grouped := map[string][]string{}
	for _, ext := range extractions {
		class := strings.ToLower(strings.TrimSpace(ext.ExtractionClass))
		text := strings.TrimSpace(ext.ExtractionText)
		if class == "" || text == "" {
			continue
		}
		if !contains(grouped[class], text) {
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

func contains(values []string, candidate string) bool {
	for _, v := range values {
		if v == candidate {
			return true
		}
	}
	return false
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
		query := fmt.Sprintf(`SELECT * FROM "%s"."%s" LIMIT %d`, schema, table, limit)
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
	if len(s.ocrCommand) == 0 {
		return generationResult{}, errors.New("OCR command not configured (set DEEPSEEK_OCR_SCRIPT or OCR_COMMAND)")
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
		script = "./scripts/deepseek_ocr_cli.py"
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
