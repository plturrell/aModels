package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	trainingmigrations "ai_benchmarks/services/training/migrations"
)

func main() {
	logger := log.New(os.Stdout, "[training:migrate] ", log.Lmsgprefix|log.LstdFlags)

	cmd := "up"
	if len(os.Args) > 1 {
		cmd = strings.ToLower(os.Args[1])
	}

	dsn := strings.TrimSpace(os.Getenv("SQL_MIGRATIONS_DSN"))
	if dsn == "" {
		dsn = strings.TrimSpace(os.Getenv("POSTGRES_DSN"))
	}
	if dsn == "" {
		logger.Fatalln("missing SQL_MIGRATIONS_DSN or POSTGRES_DSN environment variable")
	}

	var err error
	switch cmd {
	case "up":
		err = trainingmigrations.Up(dsn, logger)
	case "down":
		err = trainingmigrations.Down(dsn, logger)
	case "status":
		err = trainingmigrations.Status(dsn, logger)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		fmt.Fprintln(os.Stderr, "usage: go run ./services/training/cmd/migrate [up|down|status]")
		os.Exit(1)
	}

	if err != nil {
		logger.Fatalf("migration command %q failed: %v", cmd, err)
	}
}
