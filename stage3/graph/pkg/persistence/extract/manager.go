package extract

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
	"github.com/redis/go-redis/v9"
)

// Record represents a single entity extraction to persist.
type Record struct {
	FilePath   string
	Class      string
	Text       string
	Attributes map[string]any
	Source     string
	ExtractedAt time.Time
}

// Summary captures persistence outcomes per backend.
type Summary struct {
	Records int
	Targets map[string]string
}

type manager struct {
	logger *log.Logger

	sqlite     *sql.DB
	sqliteOnce sync.Once
	sqliteErr  error

	redis *redis.Client

	neo4jDriver neo4j.DriverWithContext

	hanaPool *hanapool.Pool

	initOnce sync.Once
	initErr  error
}

var (
	defaultManager *manager
	managerOnce    sync.Once
)

// Persist stores the provided records using the configured backends.
func Persist(ctx context.Context, records []Record) (Summary, error) {
	mgr, err := getManager()
	if err != nil {
		return Summary{}, err
	}
	return mgr.persist(ctx, records)
}

func getManager() (*manager, error) {
	managerOnce.Do(func() {
		defaultManager = &manager{
			logger: log.New(os.Stdout, "[extract-persist] ", log.LstdFlags),
		}
		defaultManager.initOnce.Do(func() {
			defaultManager.initErr = defaultManager.initialize()
		})
	})
	if defaultManager.initErr != nil {
		return nil, defaultManager.initErr
	}
	return defaultManager, nil
}

func (m *manager) initialize() error {
	if err := m.initSQLite(); err != nil {
		return err
	}
	if err := m.initRedis(); err != nil {
		m.logger.Printf("redis disabled: %v", err)
	}
	if err := m.initNeo4j(); err != nil {
		m.logger.Printf("neo4j disabled: %v", err)
	}
	if err := m.initHANA(); err != nil {
		m.logger.Printf("hana disabled: %v", err)
	}
	if m.sqlite == nil && m.redis == nil && m.neo4jDriver == nil && m.hanaPool == nil {
		return errors.New("no persistence backends configured")
	}
	return nil
}

func (m *manager) persist(ctx context.Context, records []Record) (Summary, error) {
	summary := Summary{
		Records: len(records),
		Targets: map[string]string{},
	}

	if len(records) == 0 {
		return summary, nil
	}

	successful := 0

	if m.sqlite != nil {
		if err := m.persistSQLite(ctx, records); err != nil {
			m.logger.Printf("sqlite persistence failed: %v", err)
			summary.Targets["sqlite"] = err.Error()
		} else {
			summary.Targets["sqlite"] = "ok"
			successful++
		}
	}
	if m.redis != nil {
		if err := m.persistRedis(ctx, records); err != nil {
			m.logger.Printf("redis persistence failed: %v", err)
			summary.Targets["redis"] = err.Error()
		} else {
			summary.Targets["redis"] = "ok"
			successful++
		}
	}
	if m.neo4jDriver != nil {
		if err := m.persistNeo4j(ctx, records); err != nil {
			m.logger.Printf("neo4j persistence failed: %v", err)
			summary.Targets["neo4j"] = err.Error()
		} else {
			summary.Targets["neo4j"] = "ok"
			successful++
		}
	}
	if m.hanaPool != nil {
		if err := m.persistHANA(ctx, records); err != nil {
			m.logger.Printf("hana persistence failed: %v", err)
			summary.Targets["hana"] = err.Error()
		} else {
			summary.Targets["hana"] = "ok"
			successful++
		}
	}

	if successful == 0 {
		return summary, errors.New("failed to persist records to any backend")
	}

	return summary, nil
}

// ---- initialization helpers ----

func (m *manager) initSQLite() error {
	path := sqlitePath()
	if path == "" {
		return fmt.Errorf("sqlite path not configured (set EXTRACT_SQLITE_PATH or CHECKPOINT_STORE_URL)")
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("sqlite: ensure directory: %w", err)
	}

	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return fmt.Errorf("sqlite: open: %w", err)
	}
	db.SetMaxOpenConns(1)

	m.sqlite = db
	return nil
}

func (m *manager) initRedis() error {
	raw := os.Getenv("REDIS_URL")
	if raw == "" {
		return fmt.Errorf("REDIS_URL not set")
 	}
	opts, err := redis.ParseURL(raw)
	if err != nil {
		return fmt.Errorf("redis parse url: %w", err)
	}
	client := redis.NewClient(opts)
	if err := client.Ping(context.Background()).Err(); err != nil {
		return fmt.Errorf("redis ping: %w", err)
	}
	m.redis = client
	return nil
}

func (m *manager) initNeo4j() error {
	raw := os.Getenv("NEO4J_URL")
	if raw == "" {
		return fmt.Errorf("NEO4J_URL not set")
	}
	username := os.Getenv("NEO4J_USERNAME")
	if username == "" {
		username = "neo4j"
	}
	password := os.Getenv("NEO4J_PASSWORD")
	if password == "" {
		password = "password"
	}

	driver, err := neo4j.NewDriverWithContext(raw, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return fmt.Errorf("neo4j driver: %w", err)
	}
	if err := driver.VerifyConnectivity(context.Background()); err != nil {
		return fmt.Errorf("neo4j connectivity: %w", err)
	}
	m.neo4jDriver = driver
	return nil
}

func (m *manager) initHANA() error {
	pool, err := hanapool.NewPoolFromEnv()
	if err != nil {
		return err
	}
	m.hanaPool = pool
	return nil
}

// ---- backend persistence implementations ----

func (m *manager) persistSQLite(ctx context.Context, records []Record) error {
	m.sqliteOnce.Do(func() {
		const ddl = `
CREATE TABLE IF NOT EXISTS extract_entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path TEXT,
  entity_class TEXT,
  entity_text TEXT,
  attributes_json TEXT,
  source TEXT,
  extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);`
		_, m.sqliteErr = m.sqlite.ExecContext(ctx, ddl)
	})
	if m.sqliteErr != nil {
		return m.sqliteErr
	}

	stmt, err := m.sqlite.PrepareContext(ctx, `
INSERT INTO extract_entities (file_path, entity_class, entity_text, attributes_json, source, extracted_at)
VALUES (?, ?, ?, ?, ?, ?)
`)
	if err != nil {
		return fmt.Errorf("sqlite prepare: %w", err)
	}
	defer stmt.Close()

	for _, rec := range records {
		attrJSON, _ := json.Marshal(rec.Attributes)
		extractedAt := rec.ExtractedAt.UTC().Format(time.RFC3339Nano)
		if _, err := stmt.ExecContext(ctx, rec.FilePath, rec.Class, rec.Text, string(attrJSON), rec.Source, extractedAt); err != nil {
			return fmt.Errorf("sqlite insert: %w", err)
		}
	}

	return nil
}

func (m *manager) persistRedis(ctx context.Context, records []Record) error {
	payload := struct {
		FilePath string                 `json:"file_path"`
		Class    string                 `json:"class"`
		Text     string                 `json:"text"`
		Source   string                 `json:"source"`
		Attrs    map[string]any         `json:"attributes"`
		Time     time.Time              `json:"extracted_at"`
	}{
		Attrs: map[string]any{},
	}

	pipe := m.redis.Pipeline()
	for _, rec := range records {
		payload.FilePath = rec.FilePath
		payload.Class = rec.Class
		payload.Text = rec.Text
		payload.Source = rec.Source
		payload.Attrs = rec.Attributes
		payload.Time = rec.ExtractedAt.UTC()

		data, _ := json.Marshal(payload)
		pipe.LPush(ctx, "extract:entities", data)
	}
	pipe.LTrim(ctx, "extract:entities", 0, 9999)

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("redis pipeline: %w", err)
	}
	return nil
}

func (m *manager) persistNeo4j(ctx context.Context, records []Record) error {
	session := m.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeWrite,
	})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, rec := range records {
			params := map[string]any{
				"filePath":    rec.FilePath,
				"class":       rec.Class,
				"text":        rec.Text,
				"source":      rec.Source,
				"attributes":  rec.Attributes,
				"extractedAt": rec.ExtractedAt.UTC().Format(time.RFC3339Nano),
			}
			cypher := `
MERGE (f:File {path: $filePath})
SET f.updatedAt = datetime()
MERGE (e:ExtractedEntity {filePath: $filePath, class: $class, text: $text, source: $source})
SET e.attributes = $attributes, e.extractedAt = datetime($extractedAt)
MERGE (f)-[:HAS_ENTITY]->(e)
`
			if _, err := tx.Run(ctx, cypher, params); err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
	if err != nil {
		return fmt.Errorf("neo4j write: %w", err)
	}
	return nil
}

func (m *manager) persistHANA(ctx context.Context, records []Record) error {
	db := m.hanaPool.GetDB()
	if db == nil {
		return fmt.Errorf("hana db unavailable")
	}

	schema := os.Getenv("HANA_SCHEMA")
	if schema == "" {
		schema = "AGENTICAI"
	}

	tableName := fmt.Sprintf(`%s.EXTRACT_ENTITIES`, schema)
	ddl := fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS %s (
  ID NVARCHAR(64) PRIMARY KEY,
  FILE_PATH NVARCHAR(1024),
  ENTITY_CLASS NVARCHAR(256),
  ENTITY_TEXT NVARCHAR(4096),
  ATTRIBUTES NCLOB,
  SOURCE NVARCHAR(128),
  EXTRACTED_AT TIMESTAMP,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)`, tableName)

	if _, err := db.ExecContext(ctx, ddl); err != nil {
		return fmt.Errorf("hana ddl: %w", err)
	}

	stmt, err := db.PrepareContext(ctx, fmt.Sprintf(`
UPSERT %s (ID, FILE_PATH, ENTITY_CLASS, ENTITY_TEXT, ATTRIBUTES, SOURCE, EXTRACTED_AT)
VALUES (?, ?, ?, ?, ?, ?, ?)
WITH PRIMARY KEY
`, tableName))
	if err != nil {
		return fmt.Errorf("hana prepare: %w", err)
	}
	defer stmt.Close()

	for _, rec := range records {
		attrJSON, _ := json.Marshal(rec.Attributes)
		id := uuid.New().String()
		extractedAt := rec.ExtractedAt.UTC()
		if _, err := stmt.ExecContext(ctx,
			id,
			rec.FilePath,
			rec.Class,
			rec.Text,
			string(attrJSON),
			rec.Source,
			extractedAt,
		); err != nil {
			return fmt.Errorf("hana upsert: %w", err)
		}
	}

	return nil
}

// ---- helpers ----

func sqlitePath() string {
	if explicit := strings.TrimSpace(os.Getenv("EXTRACT_SQLITE_PATH")); explicit != "" {
		return explicit
	}
	raw := strings.TrimSpace(os.Getenv("CHECKPOINT_STORE_URL"))
	if raw == "" {
		return ""
	}
	if !strings.HasPrefix(raw, "sqlite://") {
		return ""
	}
	path := strings.TrimPrefix(raw, "sqlite://")
	if strings.HasPrefix(path, "/") {
		return path
	}
	if strings.HasPrefix(path, "file:") {
		return strings.TrimPrefix(path, "file:")
	}
	return path
}
