package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

type validationResult struct {
	DSN          string `json:"dsn"`
	Nodes        int    `json:"nodes"`
	Edges        int    `json:"edges"`
	MinNodes     int    `json:"min_nodes"`
	MinEdges     int    `json:"min_edges"`
	TimeoutMS    int    `json:"timeout_ms"`
	Status       string `json:"status"`
	ErrorMessage string `json:"error,omitempty"`
}

func main() {
	var (
		dsnFlag      string
		minNodesFlag int
		minEdgesFlag int
		timeoutFlag  time.Duration
	)

	flag.StringVar(&dsnFlag, "dsn", "", "Postgres DSN; defaults to POSTGRES_CATALOG_DSN env var")
	flag.IntVar(&minNodesFlag, "require-nodes", 1, "Minimum glean_nodes rows required")
	flag.IntVar(&minEdgesFlag, "require-edges", 1, "Minimum glean_edges rows required")
	flag.DurationVar(&timeoutFlag, "timeout", 5*time.Second, "Timeout for connectivity checks")
	flag.Parse()

	dsn := strings.TrimSpace(dsnFlag)
	if dsn == "" {
		dsn = strings.TrimSpace(os.Getenv("POSTGRES_CATALOG_DSN"))
	}

	result := validationResult{
		DSN:       redactPassword(dsn),
		MinNodes:  minNodesFlag,
		MinEdges:  minEdgesFlag,
		TimeoutMS: int(timeoutFlag / time.Millisecond),
		Status:    "ok",
	}

	if dsn == "" {
		fail(&result, "missing Postgres DSN (set POSTGRES_CATALOG_DSN or provide --dsn)")
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		fail(&result, fmt.Sprintf("open connection: %v", err))
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), timeoutFlag)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		fail(&result, fmt.Sprintf("ping: %v", err))
	}

	result.Nodes, err = countRows(ctx, db, "SELECT COUNT(*) FROM glean_nodes")
	if err != nil {
		fail(&result, fmt.Sprintf("glean_nodes count: %v", err))
	}

	result.Edges, err = countRows(ctx, db, "SELECT COUNT(*) FROM glean_edges")
	if err != nil {
		fail(&result, fmt.Sprintf("glean_edges count: %v", err))
	}

	if result.Nodes < minNodesFlag {
		fail(&result, fmt.Sprintf("glean_nodes rows (%d) below threshold %d", result.Nodes, minNodesFlag))
	}
	if result.Edges < minEdgesFlag {
		fail(&result, fmt.Sprintf("glean_edges rows (%d) below threshold %d", result.Edges, minEdgesFlag))
	}

	writeResult(result, 0)
}

func countRows(ctx context.Context, db *sql.DB, query string) (int, error) {
	var count int
	if err := db.QueryRowContext(ctx, query).Scan(&count); err != nil {
		return 0, err
	}
	return count, nil
}

func fail(result *validationResult, message string) {
	result.Status = "error"
	result.ErrorMessage = message
	writeResult(*result, 1)
}

func writeResult(result validationResult, exitCode int) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		log.Printf("failed to encode result: %v", err)
	}
	os.Exit(exitCode)
}

func redactPassword(dsn string) string {
	if dsn == "" {
		return ""
	}
	parts := strings.SplitN(dsn, "@", 2)
	if len(parts) != 2 {
		return dsn
	}
	userInfo := parts[0]
	if !strings.Contains(userInfo, "://") {
		return dsn
	}
	schemeSplit := strings.SplitN(userInfo, "://", 2)
	if len(schemeSplit) != 2 {
		return dsn
	}
	before := schemeSplit[0] + "://"
	after := schemeSplit[1]
	if strings.Contains(after, ":") {
		user := strings.SplitN(after, ":", 2)[0]
		return before + user + ":***@" + parts[1]
	}
	return dsn
}
