package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/regulatory/api"
)

func main() {
	// Configuration
	addr := getEnv("AUDIT_SERVER_ADDR", ":8099")
	neo4jURL := getEnv("NEO4J_URL", "bolt://localhost:7687")
	neo4jUser := getEnv("NEO4J_USER", "neo4j")
	neo4jPass := getEnv("NEO4J_PASSWORD", "password")
	localAIURL := getEnv("LOCALAI_URL", "http://localhost:8080")
	gnnURL := getEnv("GNN_SERVICE_URL", "http://localhost:8081")
	gooseURL := getEnv("GOOSE_SERVER_URL", "http://localhost:8082")
	deepAgentsURL := getEnv("DEEPAGENTS_URL", "http://localhost:8083")

	logger := log.New(os.Stdout, "[audit-server] ", log.LstdFlags)
	ctx := context.Background()

	// Connect to Neo4j
	driver, err := neo4j.NewDriverWithContext(
		neo4jURL,
		neo4j.BasicAuth(neo4jUser, neo4jPass, ""),
	)
	if err != nil {
		logger.Fatalf("Failed to connect to Neo4j: %v", err)
	}
	defer driver.Close(ctx)

	// Verify Neo4j connectivity
	if err := driver.VerifyConnectivity(ctx); err != nil {
		logger.Fatalf("Neo4j connectivity check failed: %v", err)
	}
	logger.Println("✓ Connected to Neo4j")

	// Create audit handler
	auditHandler := api.NewAuditHandler(
		driver,
		localAIURL,
		gnnURL,
		gooseURL,
		deepAgentsURL,
		logger,
	)

	// Setup HTTP server
	mux := http.NewServeMux()

	// API endpoints
	mux.Handle("/api/compliance/audit/", auditHandler)

	// Serve static UI
	uiPath := getEnv("UI_PATH", "./services/regulatory/ui")
	mux.Handle("/", http.FileServer(http.Dir(uiPath)))

	// Health check
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok","service":"bcbs239-audit"}`))
	})

	// Apply middleware
	handler := corsMiddleware(loggingMiddleware(logger, mux))

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      120 * time.Second, // Audits can take time
	}

	logger.Printf("🚀 BCBS239 Audit Server listening on %s", addr)
	logger.Printf("   Neo4j: %s", neo4jURL)
	logger.Printf("   LocalAI: %s", localAIURL)
	logger.Printf("   GNN: %s", gnnURL)
	logger.Printf("   Goose: %s", gooseURL)
	logger.Printf("   DeepAgents: %s", deepAgentsURL)
	logger.Printf("   UI: http://localhost%s", addr)

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Fatalf("Server error: %v", err)
	}
}

func getEnv(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}

func loggingMiddleware(logger *log.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		logger.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
