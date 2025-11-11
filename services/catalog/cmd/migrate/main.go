package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/plturrell/aModels/services/catalog/migrations"
)

func main() {
	ctx := context.Background()
	logger := log.New(os.Stdout, "[catalog:migrate] ", log.LstdFlags|log.Lmsgprefix)

	cmd := "up"
	if len(os.Args) > 1 {
		cmd = strings.ToLower(os.Args[1])
	}

	migrationsDir, err := migrations.DefaultMigrationsDir()
	if err != nil {
		logger.Fatalf("failed to locate migrations directory: %v", err)
	}

	switch cmd {
	case "up":
		runUp(ctx, logger, migrationsDir)
	case "status":
		runStatus(ctx, logger)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		fmt.Fprintf(os.Stderr, "usage: go run ./cmd/migrate [up|status]\n")
		os.Exit(1)
	}
}

func runUp(ctx context.Context, logger *log.Logger, migrationsDir string) {
	neo4jURI := envOrDefault("NEO4J_URI", "bolt://localhost:7687")
	neo4jUsername := envOrDefault("NEO4J_USERNAME", "neo4j")
	neo4jPassword := envOrDefault("NEO4J_PASSWORD", "password")

	runnerLogger := log.New(os.Stdout, "[catalog:neo4j] ", log.LstdFlags|log.Lmsgprefix)
	runner := migrations.NewMigrationRunner(neo4jURI, neo4jUsername, neo4jPassword, runnerLogger)
	if err := runner.RunMigrations(ctx); err != nil {
		logger.Fatalf("neo4j migrations failed: %v", err)
	}

	sqlDriver := strings.TrimSpace(os.Getenv("SQL_MIGRATIONS_DRIVER"))
	sqlDSN := strings.TrimSpace(os.Getenv("SQL_MIGRATIONS_DSN"))
	if sqlDSN == "" {
		sqlDSN = strings.TrimSpace(os.Getenv("CATALOG_DATABASE_URL"))
	}
	if sqlDriver == "" && sqlDSN != "" {
		sqlDriver = "postgres"
	}
	if sqlDriver != "" && sqlDSN != "" {
		sqlLogger := log.New(os.Stdout, "[catalog:sql] ", log.LstdFlags|log.Lmsgprefix)
		if err := migrations.RunGooseMigrations(sqlDriver, sqlDSN, migrationsDir, sqlLogger); err != nil {
			logger.Fatalf("sql migrations failed: %v", err)
		}
	}

	logger.Println("migrations completed successfully")
}

func runStatus(ctx context.Context, logger *log.Logger) {
	neo4jURI := envOrDefault("NEO4J_URI", "bolt://localhost:7687")
	neo4jUsername := envOrDefault("NEO4J_USERNAME", "neo4j")
	neo4jPassword := envOrDefault("NEO4J_PASSWORD", "password")

	runner := migrations.NewMigrationRunner(neo4jURI, neo4jUsername, neo4jPassword, logger)
	status, err := runner.CheckMigrationStatus(ctx)
	if err != nil {
		logger.Fatalf("failed to check migration status: %v", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(status); err != nil {
		logger.Fatalf("failed to encode status: %v", err)
	}
}

func envOrDefault(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}
