package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	graphmigrations "ai_benchmarks/pkg/migrations/graph"
)

func main() {
	logger := log.New(os.Stdout, "[graph:migrate] ", log.Lmsgprefix|log.LstdFlags)

	cmd := "up"
	dsn := ""
	if len(os.Args) > 1 {
		cmd = strings.ToLower(os.Args[1])
	}
	if len(os.Args) > 2 {
		dsn = strings.TrimSpace(os.Args[2])
	}

	var err error
	switch cmd {
	case "up":
		err = graphmigrations.Up(dsn, logger)
	case "down":
		err = graphmigrations.Down(dsn, logger)
	case "status":
		err = graphmigrations.Status(dsn, logger)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		fmt.Fprintln(os.Stderr, "usage: go run ./cmd/migrate-graph [up|down|status] [sqlite_path]")
		os.Exit(1)
	}

	if err != nil {
		logger.Fatalf("migration command %q failed: %v", cmd, err)
	}
}
