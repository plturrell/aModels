package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/plturrell/aModels/services/extract/migrations"
)

func main() {
	logger := log.New(os.Stdout, "[extract:migrate] ", log.Lmsgprefix|log.LstdFlags)

	cmd := "up"
	if len(os.Args) > 1 {
		cmd = strings.ToLower(os.Args[1])
	}

	dsn := strings.TrimSpace(os.Getenv("SQL_MIGRATIONS_DSN"))
	if dsn == "" {
		dsn = strings.TrimSpace(os.Getenv("POSTGRES_CATALOG_DSN"))
	}
	if dsn == "" {
		logger.Fatalln("missing SQL_MIGRATIONS_DSN or POSTGRES_CATALOG_DSN environment variable")
	}

	switch cmd {
	case "up":
		if err := migrations.Up(dsn, logger); err != nil {
			logger.Fatalf("postgres migrations failed: %v", err)
		}
	case "down":
		if err := migrations.Down(dsn, logger); err != nil {
			logger.Fatalf("postgres migrations down failed: %v", err)
		}
	case "status":
		if err := migrations.Status(dsn, logger); err != nil {
			logger.Fatalf("postgres migrations status failed: %v", err)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		fmt.Fprintln(os.Stderr, "usage: go run ./cmd/migrate [up|down|status]")
		os.Exit(1)
	}
}
