package main

import (
	"log"
	"net/http"
	"os"

	"github.com/plturrell/aModels/services/catalog/api"
	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/triplestore"
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

	// Initialize ISO 11179 registry
	registry := iso11179.NewMetadataRegistry("catalog", "aModels Catalog", baseURI)
	logger.Println("ISO 11179 metadata registry initialized")

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

	// Initialize API handlers
	catalogHandlers := api.NewCatalogHandlers(registry, logger)
	sparqlHandler := api.NewSPARQLHandler(sparqlEndpoint, logger)

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

	logger.Printf("Catalog service listening on :%s", port)
	logger.Fatal(http.ListenAndServe(":"+port, mux))
}

