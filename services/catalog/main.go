package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/plturrell/aModels/services/catalog/api"
	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/migrations"
	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/security"
	"github.com/plturrell/aModels/services/catalog/triplestore"
	"github.com/plturrell/aModels/services/catalog/workflows"
)

func main() {
	logger := log.New(os.Stdout, "[catalog] ", log.LstdFlags|log.Lmsgprefix)

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

	// Initialize ISO 11179 registry
	registry := iso11179.NewMetadataRegistry("catalog", "aModels Catalog", baseURI)
	logger.Println("ISO 11179 metadata registry initialized")

	// Run migrations if enabled
	runMigrations := os.Getenv("RUN_MIGRATIONS") == "true"
	if runMigrations {
		migrationRunner := migrations.NewMigrationRunner(neo4jURI, neo4jUsername, neo4jPassword, logger)
		if err := migrationRunner.RunMigrations(context.Background()); err != nil {
			logger.Printf("Warning: Failed to run migrations: %v", err)
			// Don't fail startup if migrations fail - allow manual intervention
		} else {
			logger.Println("Migrations completed successfully")
		}
	}

	// Initialize triplestore client
	triplestoreClient, err := triplestore.NewTriplestoreClient(neo4jURI, neo4jUsername, neo4jPassword, logger)
	if err != nil {
		logger.Fatalf("Failed to create triplestore client: %v", err)
	}
	defer triplestoreClient.Close()
	logger.Println("Triplestore client initialized")

	// Initialize SPARQL client and endpoint
	sparqlClient := triplestore.NewSPARQLClient(triplestoreClient, logger)
	sparqlEndpoint := triplestore.NewSPARQLEndpoint(sparqlClient, logger)
	logger.Println("SPARQL endpoint initialized")

	// Initialize quality monitor (connects to Extract service)
	qualityMonitor := quality.NewQualityMonitor(extractServiceURL, logger)
	logger.Println("Quality monitor initialized (connected to Extract service)")

	// Initialize unified workflow integration
	unifiedWorkflow := workflows.NewUnifiedWorkflowIntegration(
		graphServiceURL,
		graphServiceURL, // Orchestration via graph service
		"http://localhost:9001", // AgentFlow
		localaiURL,
		deepResearchURL,
		registry,
		qualityMonitor,
		logger,
	)
	logger.Println("Unified workflow integration initialized")

	// Initialize API handlers
	catalogHandlers := api.NewCatalogHandlers(registry, logger)
	sparqlHandler := api.NewSPARQLHandler(sparqlEndpoint, logger)
	dataProductHandler := api.NewDataProductHandler(unifiedWorkflow, logger)
	
	// Initialize auth middleware
	authMiddleware := security.NewAuthMiddleware(logger)
	// Register default token for testing (in production, load from config)
	authMiddleware.RegisterToken("test-token", "test-user")

	// Setup HTTP routes
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

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
		logger.Println("Authentication enabled")
	}

	logger.Printf("Catalog service listening on :%s", port)
	logger.Println("Complete data product endpoints available at /catalog/data-products/build")
	logger.Fatal(http.ListenAndServe(":"+port, mux))
}

