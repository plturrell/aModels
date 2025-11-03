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
	extractpb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract/gen/extractpb"
	ch "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
)

const (
	defaultPort               = "8081"
	defaultGRPCPort           = "9090"
	defaultFlightAddr         = ":8815"
	defaultLangextractURL     = "http://langextract-api:5000"
	defaultPromptDescription  = "Extract the key entities (people, projects, dates, locations) from the document text."
	defaultModelID            = "gemini-2.5-flash"
	defaultTrainingDir        = "../agenticAiETH_layer4_Training/data/extracts"
	defaultTelemetryLibrary   = "layer4_extract"
	defaultTelemetryOperation = "run_extract"
)

func main() {
	explorer := flag.Bool("explorer", false, "start the catalog explorer")
	flag.Parse()

	logger := log.New(os.Stdout, "[extract-service] ", log.LstdFlags|log.Lmsgprefix)

	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = defaultPort
	}

	langURL := strings.TrimSpace(os.Getenv("LANGEXTRACT_API_URL"))
	if langURL == "" {
		langURL = defaultLangextractURL
	}

	apiKey := os.Getenv("LANGEXTRACT_API_KEY")

	trainingDir := strings.TrimSpace(os.Getenv("TRAINING_OUTPUT_DIR"))
	if trainingDir == "" {
		trainingDir = defaultTrainingDir
	}

	server := &extractServer{
		client:         &http.Client{Timeout: 45 * time.Second},
		langextractURL: strings.TrimRight(langURL, "/"),
		apiKey:         apiKey,
		trainingDir:    trainingDir,
		logger:         logger,
		ocrCommand:     deriveOCRCommand(),

		// Persistence config
		sqlitePath:    os.Getenv("SQLITE_PATH"),
		redisAddr:     os.Getenv("REDIS_ADDR"),
		redisPassword: os.Getenv("REDIS_PASSWORD"),
		redisDB:       parseIntEnv(os.Getenv("REDIS_DB"), 0),
		neo4jURI:      os.Getenv("NEO4J_URI"),
		neo4jUsername: os.Getenv("NEO4J_USERNAME"),
		neo4jPassword: os.Getenv("NEO4J_PASSWORD"),

		// Document store
		docStorePath: os.Getenv("DOCUMENT_STORE_PATH"),
	}

	// Create persistence layer
	var graphPersistences []GraphPersistence
	if server.neo4jURI != "" {
		neo4jPersistence, err := NewNeo4jPersistence(server.neo4jURI, server.neo4jUsername, server.neo4jPassword)
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
		logger.Println("connected to neo4j")
	}

	if exportDir := strings.TrimSpace(os.Getenv("GLEAN_EXPORT_DIR")); exportDir != "" {
		predicatePrefix := strings.TrimSpace(os.Getenv("GLEAN_PREDICATE_PREFIX"))
		gleanPersistence, err := NewGleanPersistence(exportDir, predicatePrefix, logger)
		if err != nil {
			logger.Fatalf("failed to create glean persistence: %v", err)
		}
		graphPersistences = append(graphPersistences, gleanPersistence)
		logger.Printf("glean export enabled (dir=%s, prefix=%s)", gleanPersistence.ExportDir(), gleanPersistence.PredicatePrefix())
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

	// Create vector persistence layer
	if server.redisAddr != "" {
		redisPersistence, err := NewRedisPersistence(server.redisAddr, server.redisPassword, server.redisDB)
		if err != nil {
			logger.Fatalf("failed to create redis persistence: %v", err)
		}
		server.vectorPersistence = redisPersistence
		logger.Println("redis persistence enabled")
	}

	telemetryEnabled := parseBoolEnv(os.Getenv("POSTGRES_LANG_SERVICE_ENABLED"), true)
	telemetryAddr := strings.TrimSpace(os.Getenv("POSTGRES_LANG_SERVICE_ADDR"))
	telemetryPrivacy := strings.TrimSpace(os.Getenv("POSTGRES_LANG_SERVICE_PRIVACY"))
	telemetryUser := strings.TrimSpace(os.Getenv("POSTGRES_LANG_SERVICE_USER_ID"))
	telemetryLibrary := strings.TrimSpace(os.Getenv("POSTGRES_LANG_SERVICE_LIBRARY_TYPE"))
	if telemetryLibrary == "" {
		telemetryLibrary = defaultTelemetryLibrary
	}

	telemetryOperation := strings.TrimSpace(os.Getenv("POSTGRES_LANG_SERVICE_OPERATION"))
	if telemetryOperation == "" {
		telemetryOperation = defaultTelemetryOperation
	}
	server.telemetryOperation = telemetryOperation

	if telemetryEnabled && telemetryAddr != "" {
		telemetryClient, err := newTelemetryClient(context.Background(), telemetryConfig{
			Address:          telemetryAddr,
			LibraryType:      telemetryLibrary,
			DefaultOperation: telemetryOperation,
			PrivacyLevel:     telemetryPrivacy,
			UserIDHash:       telemetryUser,
			DialTimeout:      5 * time.Second,
			CallTimeout:      3 * time.Second,
		})
		if err != nil {
			logger.Printf("telemetry disabled: %v", err)
		} else {
			server.telemetry = telemetryClient
			logger.Printf("telemetry enabled (addr=%s, library=%s)", telemetryAddr, telemetryLibrary)
			defer telemetryClient.Close()
		}
	} else if telemetryEnabled && telemetryAddr == "" {
		logger.Printf("telemetry disabled: POSTGRES_LANG_SERVICE_ADDR not set")
	}

	if err := os.MkdirAll(trainingDir, 0o755); err != nil {
		logger.Fatalf("failed to prepare training directory: %v", err)
	}

	grpcPort := strings.TrimSpace(os.Getenv("GRPC_PORT"))
	if grpcPort == "" {
		grpcPort = defaultGRPCPort
	}

	grpcAddr := ":" + grpcPort

	flightAddr := strings.TrimSpace(os.Getenv("FLIGHT_ADDR"))
	if flightAddr == "" {
		flightAddr = defaultFlightAddr
	}

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
	mux.HandleFunc("/graph", server.handleGraph)
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
		addr := ":" + port
		logger.Printf("agenticAiETH_layer4_Extract listening on %s (proxying %s)", addr, server.langextractURL)
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

	tablePersistence    TablePersistence
	vectorPersistence   VectorPersistence
	graphPersistence    GraphPersistence
	flight              *extractFlightServer
	hanaReplication     *hanaReplication
	postgresReplication *postgresReplication

	telemetry          *telemetryClient
	telemetryOperation string
	catalog            *Catalog
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
func (s *extractServer) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":        "ok",
		"langextract":   s.langextractURL,
		"training_dir":  s.trainingDir,
		"ocr_available": len(s.ocrCommand) > 0,
	})
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

	writeJSON(w, http.StatusOK, response)
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
		writeJSON(w, http.StatusOK, map[string]any{
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
		writeJSON(w, http.StatusOK, map[string]any{
			"success":  true,
			"mode":     "document",
			"manifest": result.ManifestPath,
			"files":    result.FilePaths,
		})
	default:
		http.Error(w, fmt.Sprintf("unsupported mode %q", mode), http.StatusBadRequest)
	}
}

// --- /graph ---

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

	for _, path := range req.ControlMFiles {
		if strings.TrimSpace(path) == "" {
			continue
		}

		jobs, err := parseControlMXML(path)
		if err != nil {
			s.logger.Printf("failed to parse control-m xml file %q: %v", path, err)
			continue
		}

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

			if err := s.vectorPersistence.SaveVector(key, embedding); err != nil {
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
			"string":  0.4,
			"number":  0.4,
			"boolean": 0.1,
			"date":    0.05,
			"array":   0.03,
			"object":  0.02,
		}
	}
	klDivergence := calculateKLDivergence(actualDistribution, idealDistribution)

	s.replicateSchema(ctx, nodes, edges)

	if s.graphPersistence != nil {
		if err := s.graphPersistence.SaveGraph(nodes, edges); err != nil {
			s.logger.Printf("failed to save graph: %v", err)
		}
	}

	if s.flight != nil {
		s.flight.UpdateGraph(nodes, edges)
	}

	response := map[string]any{
		"nodes":            nodes,
		"edges":            edges,
		"metadata_entropy": metadataEntropy,
		"kl_divergence":    klDivergence,
		"root_node_id":     rootID,
	}

	if normResult.Stats != nil {
		response["normalization"] = map[string]any{
			"root_node_id": rootID,
			"stats":        normResult.Stats,
			"warnings":     normResult.Warnings,
		}
	}

	writeJSON(w, http.StatusOK, response)
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
				if len(profile.examples) < 3 {
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
		props["type"] = typeKeys[0]
	default:
		props["type"] = "mixed"
		props["types"] = typeKeys
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
	if err := writeJSONFile(manifestPath, manifestData); err != nil {
		return generationResult{}, err
	}

	// New orchestration integration:
	orchChain, err := ch.GetChainByName("relational_table_extraction")
	if err != nil {
		return generationResult{}, fmt.Errorf("orchestration: %w", err)
	}
	inputs := map[string]any{
		"input_path":    manifestPath,
		"output_format": format,
		"hints": map[string]any{ // use hints as needed
			"schema": schema,
			"tables": tables,
		},
	}
	if _, err := ch.Call(ctx, orchChain, inputs); err != nil {
		return generationResult{}, fmt.Errorf("orchestration chain: %w", err)
	}
	// Attach/Log manifest: optionally write or embed in result.
	// ... downstream processing ...

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

func writeJSONFile(path string, payload any) error {
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
	if err := writeJSONFile(manifestPath, manifest); err != nil {
		return generationResult{}, err
	}

	return generationResult{
		FilePaths:    outputs,
		ManifestPath: manifestPath,
	}, nil
}

func deriveOCRCommand() []string {
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

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("failed to write JSON response: %v", err)
	}
}

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
