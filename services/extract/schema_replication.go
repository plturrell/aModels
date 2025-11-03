package main

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
)

func (s *extractServer) replicateSchema(ctx context.Context, nodes []Node, edges []Edge) {
	if len(nodes) == 0 && len(edges) == 0 {
		return
	}

	if s.tablePersistence != nil {
		if err := replicateSchemaToSQLite(s.tablePersistence, nodes, edges); err != nil {
			s.logger.Printf("failed to replicate schema to sqlite: %v", err)
		}
	}

	if redisStore, ok := s.vectorPersistence.(*RedisPersistence); ok {
		if err := redisStore.SaveSchema(nodes, edges); err != nil {
			s.logger.Printf("failed to replicate schema to redis: %v", err)
		}
	}

	if s.hanaReplication != nil {
		if err := s.hanaReplication.Replicate(ctx, nodes, edges); err != nil {
			s.logger.Printf("failed to replicate schema to hana: %v", err)
		}
	}

	if s.postgresReplication != nil {
		if err := s.postgresReplication.Replicate(ctx, nodes, edges); err != nil {
			s.logger.Printf("failed to replicate schema to postgres: %v", err)
		}
	}
}

func replicateSchemaToSQLite(p TablePersistence, nodes []Node, edges []Edge) error {
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

func replicateSchemaToHANA(ctx context.Context, db *sql.DB, nodes []Node, edges []Edge) error {
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

func replicateSchemaToPostgres(ctx context.Context, db *sql.DB, nodes []Node, edges []Edge) error {
	if db == nil {
		return nil
	}

	if err := ensurePostgresTables(ctx, db); err != nil {
		return err
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("postgres begin tx: %w", err)
	}

	nodeStmt, err := tx.PrepareContext(ctx, `
INSERT INTO glean_nodes (id, kind, label, properties_json, updated_at_utc)
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT (id) DO UPDATE SET kind = EXCLUDED.kind, label = EXCLUDED.label, properties_json = EXCLUDED.properties_json, updated_at_utc = NOW()`)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("postgres prepare nodes: %w", err)
	}
	defer nodeStmt.Close()

	for _, node := range nodes {
		if strings.TrimSpace(node.ID) == "" {
			continue
		}
		if _, err := nodeStmt.ExecContext(ctx, node.ID, node.Type, node.Label, jsonBytes(node.Props)); err != nil {
			tx.Rollback()
			return fmt.Errorf("postgres upsert node %s: %w", node.ID, err)
		}
	}

	edgeStmt, err := tx.PrepareContext(ctx, `
INSERT INTO glean_edges (source_id, target_id, label, properties_json, updated_at_utc)
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT (source_id, target_id, label) DO UPDATE SET properties_json = EXCLUDED.properties_json, updated_at_utc = NOW()`)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("postgres prepare edges: %w", err)
	}
	defer edgeStmt.Close()

	for _, edge := range edges {
		if strings.TrimSpace(edge.SourceID) == "" || strings.TrimSpace(edge.TargetID) == "" {
			continue
		}
		if _, err := edgeStmt.ExecContext(ctx, edge.SourceID, edge.TargetID, edge.Label, jsonBytes(edge.Props)); err != nil {
			tx.Rollback()
			return fmt.Errorf("postgres upsert edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("postgres commit: %w", err)
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

type hanaReplication struct {
	dsn    string
	logger *log.Logger
	mu     sync.Mutex
	db     *sql.DB
}

func newHANASchemaReplication(logger *log.Logger) *hanaReplication {
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
	return &hanaReplication{dsn: dsn, logger: logger}
}

func (h *hanaReplication) ensureConnection(ctx context.Context) (*sql.DB, error) {
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

func (h *hanaReplication) Replicate(ctx context.Context, nodes []Node, edges []Edge) error {
	if len(nodes) == 0 && len(edges) == 0 {
		return nil
	}
	db, err := h.ensureConnection(ctx)
	if err != nil {
		return err
	}
	return replicateSchemaToHANA(ctx, db, nodes, edges)
}

func (h *hanaReplication) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.db != nil {
		h.db.Close()
		h.db = nil
	}
}

type postgresReplication struct {
	dsn    string
	logger *log.Logger
	mu     sync.Mutex
	db     *sql.DB
}

func newPostgresSchemaReplication(logger *log.Logger) *postgresReplication {
	dsn := strings.TrimSpace(os.Getenv("POSTGRES_CATALOG_DSN"))
	if dsn == "" {
		return nil
	}
	return &postgresReplication{dsn: dsn, logger: logger}
}

func (p *postgresReplication) ensureConnection(ctx context.Context) (*sql.DB, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.db != nil {
		return p.db, nil
	}

	db, err := sql.Open("postgres", p.dsn)
	if err != nil {
		return nil, err
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

func (p *postgresReplication) Replicate(ctx context.Context, nodes []Node, edges []Edge) error {
	if len(nodes) == 0 && len(edges) == 0 {
		return nil
	}
	db, err := p.ensureConnection(ctx)
	if err != nil {
		return err
	}
	return replicateSchemaToPostgres(ctx, db, nodes, edges)
}

func (p *postgresReplication) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.db != nil {
		p.db.Close()
		p.db = nil
	}
}
