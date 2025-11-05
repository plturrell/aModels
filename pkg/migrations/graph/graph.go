package graph

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/mattn/go-sqlite3"
	goose "github.com/pressly/goose/v3"
)

func migrationsDir() (string, error) {
	path := filepath.Join("services", "graph", "migrations")
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	return "", fmt.Errorf("graph migrations directory not found")
}

func resolveDSN(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	if raw := strings.TrimSpace(os.Getenv("SQLITE_MIGRATIONS_DSN")); raw != "" {
		return raw, nil
	}
	if raw := strings.TrimSpace(os.Getenv("EXTRACT_SQLITE_PATH")); raw != "" {
		return raw, nil
	}
	if raw := strings.TrimSpace(os.Getenv("CHECKPOINT_STORE_URL")); strings.HasPrefix(strings.ToLower(raw), "sqlite:") {
		return raw[len("sqlite:"):], nil
	}
	return "", fmt.Errorf("missing SQLITE_MIGRATIONS_DSN, EXTRACT_SQLITE_PATH, or CHECKPOINT_STORE_URL")
}

func withDB(dsn string, logger *log.Logger, fn func(dir string, db *sql.DB) error) error {
	dir, err := migrationsDir()
	if err != nil {
		return err
	}
	if err := goose.SetDialect("sqlite3"); err != nil {
		return fmt.Errorf("set goose dialect: %w", err)
	}
	if logger != nil {
		goose.SetLogger(logger)
	}
	db, err := goose.OpenDBWithDriver("sqlite3", dsn)
	if err != nil {
		return fmt.Errorf("open sqlite database: %w", err)
	}
	defer db.Close()
	return fn(dir, db)
}

// Up applies all pending migrations to the graph SQLite datastore.
func Up(dsn string, logger *log.Logger) error {
	resolved, err := resolveDSN(dsn)
	if err != nil {
		return err
	}
	return withDB(resolved, logger, func(dir string, db *sql.DB) error {
		return goose.Up(db, dir)
	})
}

// Down rolls back the most recent migration.
func Down(dsn string, logger *log.Logger) error {
	resolved, err := resolveDSN(dsn)
	if err != nil {
		return err
	}
	return withDB(resolved, logger, func(dir string, db *sql.DB) error {
		return goose.Down(db, dir)
	})
}

// Status prints the migration status.
func Status(dsn string, logger *log.Logger) error {
	resolved, err := resolveDSN(dsn)
	if err != nil {
		return err
	}
	return withDB(resolved, logger, func(dir string, db *sql.DB) error {
		return goose.Status(db, dir)
	})
}
