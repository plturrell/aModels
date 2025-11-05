package migrations

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/lib/pq"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	goose "github.com/pressly/goose/v3"
)

// MigrationRunner handles database migrations for the catalog service.
type MigrationRunner struct {
	neo4jURI      string
	neo4jUsername string
	neo4jPassword string
	logger        *log.Logger
}

// NewMigrationRunner creates a new migration runner.
func NewMigrationRunner(neo4jURI, neo4jUsername, neo4jPassword string, logger *log.Logger) *MigrationRunner {
	return &MigrationRunner{
		neo4jURI:      neo4jURI,
		neo4jUsername: neo4jUsername,
		neo4jPassword: neo4jPassword,
		logger:        logger,
	}
}

// RunMigrations executes all pending migrations.
func (mr *MigrationRunner) RunMigrations(ctx context.Context) error {
	migrationsDir, err := DefaultMigrationsDir()
	if err != nil {
		return err
	}

	if mr.logger != nil {
		mr.logger.Println("Running Neo4j migrations...")
	}

	// Run Neo4j migrations using Cypher scripts
	// Note: Goose doesn't natively support Neo4j, so we'll execute Cypher directly
	err = mr.runNeo4jMigrations(ctx, migrationsDir)
	if err != nil {
		return fmt.Errorf("failed to run Neo4j migrations: %w", err)
	}

	if mr.logger != nil {
		mr.logger.Println("Migrations completed successfully")
	}

	return nil
}

// runNeo4jMigrations executes Cypher migration scripts.
func (mr *MigrationRunner) runNeo4jMigrations(ctx context.Context, migrationsDir string) error {
	driver, err := neo4j.NewDriverWithContext(mr.neo4jURI, neo4j.BasicAuth(mr.neo4jUsername, mr.neo4jPassword, ""))
	if err != nil {
		return fmt.Errorf("failed to create Neo4j driver: %w", err)
	}
	defer driver.Close(ctx)

	session := driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// Read migration files in order
	files, err := filepath.Glob(filepath.Join(migrationsDir, "*.cypher"))
	if err != nil {
		return fmt.Errorf("failed to read migration files: %w", err)
	}

	// Execute each migration file
	for _, file := range files {
		if mr.logger != nil {
			mr.logger.Printf("Running migration: %s", filepath.Base(file))
		}

		content, err := os.ReadFile(file)
		if err != nil {
			return fmt.Errorf("failed to read migration file %s: %w", file, err)
		}

		// Split by goose directives and execute "Up" sections
		migrationQueries := parseGooseMigration(string(content))
		for _, query := range migrationQueries {
			if query == "" {
				continue
			}

			_, err = session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
				result, err := tx.Run(ctx, query, nil)
				if err != nil {
					return nil, err
				}
				return result.Consume(ctx)
			})

			if err != nil {
				return fmt.Errorf("failed to execute migration %s: %w", filepath.Base(file), err)
			}
		}
	}

	return nil
}

// parseGooseMigration parses a goose migration file and extracts the "Up" section.
func parseGooseMigration(content string) []string {
	var queries []string
	var inUpSection bool
	var currentQuery string

	lines := splitLines(content)
	for _, line := range lines {
		if contains(line, "-- +goose Up") {
			inUpSection = true
			continue
		}
		if contains(line, "-- +goose Down") {
			inUpSection = false
			break
		}
		if inUpSection {
			// Skip empty lines and comments
			if line == "" || (len(line) > 0 && line[0] == '-') {
				continue
			}
			currentQuery += line + "\n"
			// If line ends with semicolon, it's a complete query
			if len(line) > 0 && line[len(line)-1] == ';' {
				if currentQuery != "" {
					queries = append(queries, currentQuery)
					currentQuery = ""
				}
			}
		}
	}

	// Add remaining query if any
	if currentQuery != "" {
		queries = append(queries, currentQuery)
	}

	return queries
}

func splitLines(s string) []string {
	return strings.Split(s, "\n")
}

func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}

// CheckMigrationStatus checks the current migration status.
func (mr *MigrationRunner) CheckMigrationStatus(ctx context.Context) (map[string]interface{}, error) {
	driver, err := neo4j.NewDriverWithContext(mr.neo4jURI, neo4j.BasicAuth(mr.neo4jUsername, mr.neo4jPassword, ""))
	if err != nil {
		return nil, err
	}
	defer driver.Close(ctx)

	session := driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// Check if constraints exist (indicates migrations have run)
	result, err := session.Run(ctx, "SHOW CONSTRAINTS", nil)
	if err != nil {
		return nil, err
	}

	var constraints []map[string]interface{}
	for result.Next(ctx) {
		record := result.Record()
		constraint := make(map[string]interface{})
		for _, key := range record.Keys {
			value, _ := record.Get(key)
			constraint[key] = value
		}
		constraints = append(constraints, constraint)
	}

	return map[string]interface{}{
		"constraints": constraints,
		"status":      "ok",
	}, nil
}

// RunGooseMigrations runs SQL migrations using the goose library.
func RunGooseMigrations(driver, dsn, migrationsDir string, logger *log.Logger) error {
	if migrationsDir == "" {
		var err error
		migrationsDir, err = DefaultMigrationsDir()
		if err != nil {
			return err
		}
	}

	db, err := goose.OpenDBWithDriver(driver, dsn)
	if err != nil {
		return fmt.Errorf("open %s database: %w", driver, err)
	}
	defer db.Close()

	if err := goose.SetDialect(driver); err != nil {
		return fmt.Errorf("set goose dialect: %w", err)
	}

	if logger != nil {
		goose.SetLogger(&gooseStdLogger{logger: logger})
	} else {
		goose.SetLogger(goose.NopLogger())
	}

	if err := goose.Up(db, migrationsDir); err != nil {
		return fmt.Errorf("run goose migrations: %w", err)
	}

	return nil
}

// DefaultMigrationsDir returns the default directory containing migration files.
func DefaultMigrationsDir() (string, error) {
	candidates := []string{
		"migrations",
		filepath.Join("services", "catalog", "migrations"),
	}
	for _, dir := range candidates {
		if _, err := os.Stat(dir); err == nil {
			return dir, nil
		}
	}
	return "", fmt.Errorf("migrations directory not found")
}

type gooseStdLogger struct {
	logger *log.Logger
}

func (l *gooseStdLogger) Fatalf(format string, v ...interface{}) {
	l.logger.Fatalf(format, v...)
}

func (l *gooseStdLogger) Printf(format string, v ...interface{}) {
	l.logger.Printf(format, v...)
}
