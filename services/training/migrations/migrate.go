package migrations

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"

	_ "github.com/lib/pq"
	goose "github.com/pressly/goose/v3"
)

func migrationsDir() (string, error) {
	candidates := []string{
		"migrations",
		filepath.Join("services", "training", "migrations"),
	}
	for _, dir := range candidates {
		if _, err := os.Stat(dir); err == nil {
			return dir, nil
		}
	}
	return "", fmt.Errorf("training migrations directory not found")
}

func withDB(dsn string, logger *log.Logger, fn func(dir string, db *sql.DB) error) error {
	dir, err := migrationsDir()
	if err != nil {
		return err
	}
	if err := goose.SetDialect("postgres"); err != nil {
		return fmt.Errorf("set goose dialect: %w", err)
	}
	if logger != nil {
		goose.SetLogger(logger)
	}
	db, err := goose.OpenDBWithDriver("postgres", dsn)
	if err != nil {
		return fmt.Errorf("open postgres database: %w", err)
	}
	defer db.Close()
	return fn(dir, db)
}

// Up applies all pending migrations.
func Up(dsn string, logger *log.Logger) error {
	return withDB(dsn, logger, func(dir string, db *sql.DB) error {
		return goose.Up(db, dir)
	})
}

// Down rolls back the most recent migration.
func Down(dsn string, logger *log.Logger) error {
	return withDB(dsn, logger, func(dir string, db *sql.DB) error {
		return goose.Down(db, dir)
	})
}

// Status prints migration status information.
func Status(dsn string, logger *log.Logger) error {
	return withDB(dsn, logger, func(dir string, db *sql.DB) error {
		return goose.Status(db, dir)
	})
}
