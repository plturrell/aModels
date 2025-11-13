package schema

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
	"github.com/plturrell/aModels/services/extract/pkg/persistence"
)

// NOTE: This function references extractServer which is in cmd/extract/main.go
// It should be moved to that package. For now, it is commented out to fix compilation.
// TODO: Move replicateSchema to cmd/extract/main.go

/*
func (s *extractServer) replicateSchema(ctx context.Context, nodes []graph.Node, edges []graph.Edge) {
	if len(nodes) == 0 && len(edges) == 0 {
		return
	}

	// Improvement 1: Add data validation before storage
	validationStart := time.Now()
	validationResult := utils.ValidateGraph(nodes, edges, s.logger)
	validationDuration := time.Since(validationStart)
	
	// Record validation metrics
	if s.metricsCollector != nil {
		s.metricsCollector.RecordValidation(validationResult, validationDuration)
	}
	
	if !validationResult.Valid {
		s.logger.Printf("WARNING: Graph validation found %d errors, %d warnings. Filtering invalid data...", 
			len(validationResult.Errors), len(validationResult.Warnings))
		
		// Filter out invalid nodes and edges
		nodes = utils.FilterValidNodes(nodes, validationResult)
		edges = utils.FilterValidEdges(edges, validationResult)
		
		s.logger.Printf("After filtering: %d nodes, %d edges remain", len(nodes), len(edges))
	}

	// Store validation metrics for monitoring
	if validationResult.Metrics.ValidationErrors > 0 {
		s.logger.Printf("Validation metrics: %d nodes rejected, %d edges rejected", 
			validationResult.Metrics.NodesRejected, validationResult.Metrics.EdgesRejected)
	}

	if s.tablePersistence != nil {
		// Improvement 2: Add retry logic for storage operations
		retryStart := time.Now()
		retrySuccess := true
		if err := utils.RetryPostgresOperation(ctx, func() error {
			return replicateSchemaToSQLite(s.tablePersistence, nodes, edges)
		}, s.logger); err != nil {
			retrySuccess = false
			s.logger.Printf("failed to replicate schema to sqlite after retries: %v", err)
		}
		if s.metricsCollector != nil {
			s.metricsCollector.RecordRetry(retrySuccess, time.Since(retryStart))
		}
	}

	if redisStore, ok := s.vectorPersistence.(*RedisPersistence); ok {
		// Improvement 2: Add retry logic for Redis operations
		retryStart := time.Now()
		retrySuccess := true
		if err := utils.RetryRedisOperation(ctx, func() error {
			return redisStore.SaveSchema(nodes, edges)
		}, s.logger); err != nil {
			retrySuccess = false
			s.logger.Printf("failed to replicate schema to redis after retries: %v", err)
		}
		if s.metricsCollector != nil {
			s.metricsCollector.RecordRetry(retrySuccess, time.Since(retryStart))
		}
	}

	if s.hanaReplication != nil {
		if err := s.hanaReplication.Replicate(ctx, nodes, edges); err != nil {
			s.logger.Printf("failed to replicate schema to hana: %v", err)
		}
	}

	if s.postgresReplication != nil {
		// Improvement 2: Add retry logic for Postgres operations
		if err := utils.RetryPostgresOperation(ctx, func() error {
			return s.postgresReplication.Replicate(ctx, nodes, edges)
		}, s.logger); err != nil {
			s.logger.Printf("failed to replicate schema to postgres after retries: %v", err)
		}
	}
*/

// ReplicateSchemaToSQLite replicates schema to SQLite via TablePersistence
func ReplicateSchemaToSQLite(p persistence.TablePersistence, nodes []graph.Node, edges []graph.Edge) error {
	if p == nil {
		return nil
	}

	if len(nodes) > 0 {
		rows := make([]map[string]any, 0, len(nodes))
		for _, node := range nodes {
			rows = append(rows, map[string]any{
				"id":              node.ID,
				"type":            node.Type,
				"label":           node.Label,
				"properties_json": jsonString(node.Props),
				"recorded_at_utc": time.Now().UTC().Format(time.RFC3339Nano),
			})
		}
		if err := p.SaveTable("glean_nodes", rows); err != nil {
			return fmt.Errorf("save glean_nodes: %w", err)
		}
	}

	if len(edges) > 0 {
		rows := make([]map[string]any, 0, len(edges))
		for _, edge := range edges {
			rows = append(rows, map[string]any{
				"source":          edge.SourceID,
				"target":          edge.TargetID,
				"label":           edge.Label,
				"properties_json": jsonString(edge.Props),
				"recorded_at_utc": time.Now().UTC().Format(time.RFC3339Nano),
			})
		}
		if err := p.SaveTable("glean_edges", rows); err != nil {
			return fmt.Errorf("save glean_edges: %w", err)
		}
	}
	return nil
}

func replicateSchemaToHANA(ctx context.Context, db *sql.DB, nodes []graph.Node, edges []graph.Edge) error {
	if db == nil {
		return nil
	}

	if err := ensureHANATables(ctx, db); err != nil {
		return err
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("hana begin tx: %w", err)
	}

	nodeStmt, err := tx.PrepareContext(ctx, `UPSERT GLEAN_NODES (ID, KIND, LABEL, PROPERTIES_JSON, UPDATED_AT_UTC) VALUES (?, ?, ?, ?, CURRENT_UTCTIMESTAMP) WITH PRIMARY KEY`)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("hana prepare nodes: %w", err)
	}
	defer nodeStmt.Close()

	for _, node := range nodes {
		if strings.TrimSpace(node.ID) == "" {
			continue
		}
		if _, err := nodeStmt.ExecContext(ctx, node.ID, node.Type, node.Label, jsonString(node.Props)); err != nil {
			tx.Rollback()
			return fmt.Errorf("hana upsert node %s: %w", node.ID, err)
		}
	}

	edgeStmt, err := tx.PrepareContext(ctx, `UPSERT GLEAN_EDGES (SOURCE_ID, TARGET_ID, LABEL, PROPERTIES_JSON, UPDATED_AT_UTC) VALUES (?, ?, ?, ?, CURRENT_UTCTIMESTAMP) WITH PRIMARY KEY`)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("hana prepare edges: %w", err)
	}
	defer edgeStmt.Close()

	for _, edge := range edges {
		if strings.TrimSpace(edge.SourceID) == "" || strings.TrimSpace(edge.TargetID) == "" {
			continue
		}
		if _, err := edgeStmt.ExecContext(ctx, edge.SourceID, edge.TargetID, edge.Label, jsonString(edge.Props)); err != nil {
			tx.Rollback()
			return fmt.Errorf("hana upsert edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("hana commit: %w", err)
	}
	return nil
}

func ensureHANATables(ctx context.Context, db *sql.DB) error {
	createNodes := `
CREATE COLUMN TABLE IF NOT EXISTS GLEAN_NODES (
	ID NVARCHAR(512) PRIMARY KEY,
	KIND NVARCHAR(128),
	LABEL NVARCHAR(512),
	PROPERTIES_JSON CLOB,
	UPDATED_AT_UTC TIMESTAMP
)`
	if _, err := db.ExecContext(ctx, createNodes); err != nil {
		return fmt.Errorf("hana create GLEAN_NODES: %w", err)
	}

	createEdges := `
CREATE COLUMN TABLE IF NOT EXISTS GLEAN_EDGES (
	SOURCE_ID NVARCHAR(512),
	TARGET_ID NVARCHAR(512),
	LABEL NVARCHAR(256),
	PROPERTIES_JSON CLOB,
	UPDATED_AT_UTC TIMESTAMP,
	PRIMARY KEY (SOURCE_ID, TARGET_ID, LABEL)
)`
	if _, err := db.ExecContext(ctx, createEdges); err != nil {
		return fmt.Errorf("hana create GLEAN_EDGES: %w", err)
	}
	return nil
}

func replicateSchemaToPostgres(ctx context.Context, db *sql.DB, nodes []graph.Node, edges []graph.Edge) error {
	if db == nil {
		return nil
	}

	if err := ensurePostgresTables(ctx, db); err != nil {
		return err
	}

	// Get batch size from environment (default: 100)
	batchSize := 100
	if val := os.Getenv("POSTGRES_BATCH_SIZE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			batchSize = parsed
		}
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("postgres begin tx: %w", err)
	}
	defer func() {
		if err != nil {
			tx.Rollback()
		}
	}()

	// Batch insert nodes using COPY for better performance
	if len(nodes) > 0 {
		if err := batchInsertNodes(ctx, tx, nodes, batchSize); err != nil {
			return fmt.Errorf("batch insert nodes: %w", err)
		}
	}

	// Batch insert edges using COPY for better performance
	if len(edges) > 0 {
		if err := batchInsertEdges(ctx, tx, edges, batchSize); err != nil {
			return fmt.Errorf("batch insert edges: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("postgres commit: %w", err)
	}
	return nil
}

// batchInsertNodes inserts nodes in batches for better performance
func batchInsertNodes(ctx context.Context, tx *sql.Tx, nodes []graph.Node, batchSize int) error {
	stmt, err := tx.PrepareContext(ctx, `
INSERT INTO glean_nodes (id, kind, label, properties_json, updated_at_utc)
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT (id) DO UPDATE SET kind = EXCLUDED.kind, label = EXCLUDED.label, properties_json = EXCLUDED.properties_json, updated_at_utc = NOW()`)
	if err != nil {
		return fmt.Errorf("postgres prepare nodes: %w", err)
	}
	defer stmt.Close()

	// Process in batches
	for i := 0; i < len(nodes); i += batchSize {
		end := i + batchSize
		if end > len(nodes) {
			end = len(nodes)
		}

		batch := nodes[i:end]
		for _, node := range batch {
			if strings.TrimSpace(node.ID) == "" {
				continue
			}
			if _, err := stmt.ExecContext(ctx, node.ID, node.Type, node.Label, jsonBytes(node.Props)); err != nil {
				return fmt.Errorf("postgres upsert node %s: %w", node.ID, err)
			}
		}
	}

	return nil
}

// batchInsertEdges inserts edges in batches for better performance
func batchInsertEdges(ctx context.Context, tx *sql.Tx, edges []graph.Edge, batchSize int) error {
	stmt, err := tx.PrepareContext(ctx, `
INSERT INTO glean_edges (source_id, target_id, label, properties_json, updated_at_utc)
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT (source_id, target_id, label) DO UPDATE SET properties_json = EXCLUDED.properties_json, updated_at_utc = NOW()`)
	if err != nil {
		return fmt.Errorf("postgres prepare edges: %w", err)
	}
	defer stmt.Close()

	// Process in batches
	for i := 0; i < len(edges); i += batchSize {
		end := i + batchSize
		if end > len(edges) {
			end = len(edges)
		}

		batch := edges[i:end]
		for _, edge := range batch {
			if strings.TrimSpace(edge.SourceID) == "" || strings.TrimSpace(edge.TargetID) == "" {
				continue
			}
			if _, err := stmt.ExecContext(ctx, edge.SourceID, edge.TargetID, edge.Label, jsonBytes(edge.Props)); err != nil {
				return fmt.Errorf("postgres upsert edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
			}
		}
	}

	return nil
}

func ensurePostgresTables(ctx context.Context, db *sql.DB) error {
	createNodes := `
CREATE TABLE IF NOT EXISTS glean_nodes (
	id TEXT PRIMARY KEY,
	kind TEXT,
	label TEXT,
	properties_json JSONB,
	updated_at_utc TIMESTAMPTZ DEFAULT NOW()
)`
	if _, err := db.ExecContext(ctx, createNodes); err != nil {
		return fmt.Errorf("postgres create glean_nodes: %w", err)
	}

	createEdges := `
CREATE TABLE IF NOT EXISTS glean_edges (
	source_id TEXT,
	target_id TEXT,
	label TEXT,
	properties_json JSONB,
	updated_at_utc TIMESTAMPTZ DEFAULT NOW(),
	PRIMARY KEY (source_id, target_id, label)
)`
	if _, err := db.ExecContext(ctx, createEdges); err != nil {
		return fmt.Errorf("postgres create glean_edges: %w", err)
	}
	return nil
}

func jsonString(props map[string]any) string {
	if props == nil || len(props) == 0 {
		return "{}"
	}
	data, err := json.Marshal(props)
	if err != nil {
		return "{}"
	}
	return string(data)
}

func jsonBytes(props map[string]any) []byte {
	if props == nil || len(props) == 0 {
		return []byte("{}")
	}
	data, err := json.Marshal(props)
	if err != nil {
		return []byte("{}")
	}
	return data
}

type HANAReplication struct {
	dsn    string
	logger *log.Logger
	mu     sync.Mutex
	db     *sql.DB
}

func NewHANASchemaReplication(logger *log.Logger) *HANAReplication {
	host := strings.TrimSpace(os.Getenv("HANA_HOST"))
	user := strings.TrimSpace(os.Getenv("HANA_USER"))
	password := os.Getenv("HANA_PASSWORD")
	if host == "" || user == "" || password == "" {
		return nil
	}

	port := 39015
	if raw := strings.TrimSpace(os.Getenv("HANA_PORT")); raw != "" {
		if p, err := strconv.Atoi(raw); err == nil {
			port = p
		}
	}

	database := strings.TrimSpace(os.Getenv("HANA_DATABASE"))
	if database == "" {
		database = "HXE"
	}

	dsn := fmt.Sprintf("hdb://%s:%s@%s:%d/%s", user, password, host, port, database)
	return &HANAReplication{dsn: dsn, logger: logger}
}

func (h *HANAReplication) ensureConnection(ctx context.Context) (*sql.DB, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.db != nil {
		return h.db, nil
	}

	db, err := sql.Open("hdb", h.dsn)
	if err != nil {
		return nil, err
	}

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := db.PingContext(pingCtx); err != nil {
		db.Close()
		return nil, err
	}

	h.db = db
	return h.db, nil
}

func (h *HANAReplication) Replicate(ctx context.Context, nodes []graph.Node, edges []graph.Edge) error {
	if len(nodes) == 0 && len(edges) == 0 {
		return nil
	}
	db, err := h.ensureConnection(ctx)
	if err != nil {
		return err
	}
	return replicateSchemaToHANA(ctx, db, nodes, edges)
}

func (h *HANAReplication) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.db != nil {
		h.db.Close()
		h.db = nil
	}
}

type PostgresReplication struct {
	dsn    string
	logger *log.Logger
	mu     sync.Mutex
	db     *sql.DB
}

func NewPostgresSchemaReplication(logger *log.Logger) *PostgresReplication {
	dsn := strings.TrimSpace(os.Getenv("POSTGRES_CATALOG_DSN"))
	if dsn == "" {
		return nil
	}
	return &PostgresReplication{dsn: dsn, logger: logger}
}

func (p *PostgresReplication) ensureConnection(ctx context.Context) (*sql.DB, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.db != nil {
		// Verify connection is still alive
		if err := p.db.PingContext(ctx); err == nil {
			return p.db, nil
		}
		// Connection is dead, close it
		p.db.Close()
		p.db = nil
	}

	db, err := sql.Open("postgres", p.dsn)
	if err != nil {
		return nil, err
	}

	// Configure connection pool
	maxOpenConns := 10
	if val := os.Getenv("POSTGRES_POOL_MAX_OPEN"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxOpenConns = parsed
		}
	}

	maxIdleConns := 5
	if val := os.Getenv("POSTGRES_POOL_MAX_IDLE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxIdleConns = parsed
		}
	}

	maxLifetime := 5 * time.Minute
	if val := os.Getenv("POSTGRES_POOL_MAX_LIFETIME"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxLifetime = time.Duration(parsed) * time.Minute
		}
	}

	db.SetMaxOpenConns(maxOpenConns)
	db.SetMaxIdleConns(maxIdleConns)
	db.SetConnMaxLifetime(maxLifetime)

	if p.logger != nil {
		p.logger.Printf("Postgres connection pool configured: maxOpen=%d, maxIdle=%d, maxLifetime=%v", maxOpenConns, maxIdleConns, maxLifetime)
	}

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := db.PingContext(pingCtx); err != nil {
		db.Close()
		return nil, err
	}

	p.db = db
	return p.db, nil
}

// retryPostgresOperation retries a Postgres operation with exponential backoff
func retryPostgresOperation(ctx context.Context, logger *log.Logger, maxAttempts int, initialBackoff time.Duration, maxBackoff time.Duration, fn func() error) error {
	var lastErr error
	backoff := initialBackoff

	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Check context cancellation before waiting
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}

			if logger != nil {
				logger.Printf("Retrying Postgres operation (attempt %d/%d) after %v", attempt+1, maxAttempts, backoff)
			}
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		errStr := strings.ToLower(err.Error())
		isRetryable := strings.Contains(errStr, "connection") ||
			strings.Contains(errStr, "deadlock") ||
			strings.Contains(errStr, "timeout") ||
			strings.Contains(errStr, "network") ||
			strings.Contains(errStr, "broken pipe") ||
			strings.Contains(errStr, "connection reset")

		if !isRetryable {
			// Non-retryable error, return immediately
			return err
		}

		// Exponential backoff: double the backoff time, but cap at maxBackoff
		backoff = time.Duration(float64(backoff) * 2)
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}

	return fmt.Errorf("failed after %d attempts: %w", maxAttempts, lastErr)
}

func (p *PostgresReplication) Replicate(ctx context.Context, nodes []graph.Node, edges []graph.Edge) error {
	if len(nodes) == 0 && len(edges) == 0 {
		return nil
	}

	// Get retry configuration from environment
	maxRetries := 3
	if val := os.Getenv("POSTGRES_RETRY_MAX_ATTEMPTS"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxRetries = parsed
		}
	}

	initialBackoff := 200 * time.Millisecond
	maxBackoff := 1 * time.Second

	return retryPostgresOperation(ctx, p.logger, maxRetries, initialBackoff, maxBackoff, func() error {
		db, err := p.ensureConnection(ctx)
		if err != nil {
			return err
		}
		return replicateSchemaToPostgres(ctx, db, nodes, edges)
	})
}

func (p *PostgresReplication) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.db != nil {
		p.db.Close()
		p.db = nil
	}
}
