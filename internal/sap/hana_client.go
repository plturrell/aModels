// Package sap provides SAP HANA, Datasphere, and AI Core integration for training layer
package sap

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"

	_ "github.com/SAP/go-hdb/driver"
	"github.com/agenticAiETH/agenticAiETH_layer4/pkg/contracts"
)

// HANAClient provides HANA database connectivity for training data
type HANAClient struct {
	db       *sql.DB
	host     string
	port     string
	user     string
	database string
	schema   string
}

// HANAConfig holds HANA connection configuration
type HANAConfig struct {
	Host     string
	Port     string
	User     string
	Password string
	Database string
	Schema   string
	Encrypt  bool
}

// TrainingDataset represents a training dataset in HANA
type TrainingDataset struct {
	ID          string
	Name        string
	Description string
	Schema      string
	Table       string
	RowCount    int64
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Metadata    map[string]interface{}
}

// TrainingRecord represents a single training record
type TrainingRecord struct {
	ID       string
	Features map[string]interface{}
	Label    interface{}
	Weight   float64
	Metadata map[string]interface{}
}

// NewHANAClient creates a new HANA database client
func NewHANAClient(config HANAConfig) (*HANAClient, error) {
	// Build DSN with proper URL encoding for special characters
	dsn := fmt.Sprintf("hdb://%s:%s@%s:%s?database=%s",
		url.QueryEscape(config.User),
		url.QueryEscape(config.Password),
		config.Host,
		config.Port,
		config.Database,
	)

	if config.Encrypt {
		dsn += "&encrypt=true&validateCertificate=false"
	}

	db, err := sql.Open("hdb", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open HANA connection: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping HANA: %w", err)
	}

	client := &HANAClient{
		db:       db,
		host:     config.Host,
		port:     config.Port,
		user:     config.User,
		database: config.Database,
		schema:   config.Schema,
	}

	return client, nil
}

// Close closes the HANA connection
func (c *HANAClient) Close() error {
	if c.db != nil {
		return c.db.Close()
	}
	return nil
}

// CreateTrainingTable creates a table for training data
func (c *HANAClient) CreateTrainingTable(ctx context.Context, tableName string, columns map[string]string) error {
	var columnDefs string
	for name, dataType := range columns {
		if columnDefs != "" {
			columnDefs += ", "
		}
		columnDefs += fmt.Sprintf("%s %s", name, dataType)
	}

	query := fmt.Sprintf(`
		CREATE COLUMN TABLE "%s"."%s" (
			ID NVARCHAR(36) PRIMARY KEY,
			%s,
			CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, c.schema, tableName, columnDefs)

	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to create training table: %w", err)
	}

	return nil
}

// InsertTrainingData inserts training data into HANA
func (c *HANAClient) InsertTrainingData(ctx context.Context, tableName string, records []TrainingRecord) error {
	if len(records) == 0 {
		return nil
	}

	tx, err := c.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Build column names from first record
	var columns []string
	var placeholders []string
	idx := 1
	for col := range records[0].Features {
		columns = append(columns, col)
		placeholders = append(placeholders, fmt.Sprintf("$%d", idx))
		idx++
	}

	query := fmt.Sprintf(`
		INSERT INTO "%s"."%s" (ID, %s, LABEL, WEIGHT)
		VALUES ($1, %s, $%d, $%d)
	`, c.schema, tableName, joinStrings(columns, ", "), joinStrings(placeholders, ", "), idx, idx+1)

	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, record := range records {
		values := make([]interface{}, 0, len(columns)+3)
		values = append(values, record.ID)
		for _, col := range columns {
			values = append(values, record.Features[col])
		}
		values = append(values, record.Label, record.Weight)

		if _, err := stmt.ExecContext(ctx, values...); err != nil {
			return fmt.Errorf("failed to insert record: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// QueryTrainingData queries training data from HANA
func (c *HANAClient) QueryTrainingData(ctx context.Context, tableName string, limit int, offset int) ([]TrainingRecord, error) {
	query := fmt.Sprintf(`
		SELECT ID, LABEL, WEIGHT
		FROM "%s"."%s"
		ORDER BY CREATED_AT DESC
		LIMIT %d OFFSET %d
	`, c.schema, tableName, limit, offset)

	rows, err := c.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query training data: %w", err)
	}
	defer rows.Close()

	var records []TrainingRecord
	for rows.Next() {
		var record TrainingRecord
		if err := rows.Scan(&record.ID, &record.Label, &record.Weight); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}
		records = append(records, record)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return records, nil
}

// GetDatasetInfo retrieves dataset metadata
func (c *HANAClient) GetDatasetInfo(ctx context.Context, tableName string) (*TrainingDataset, error) {
	query := fmt.Sprintf(`
		SELECT COUNT(*) as row_count
		FROM "%s"."%s"
	`, c.schema, tableName)

	var rowCount int64
	if err := c.db.QueryRowContext(ctx, query).Scan(&rowCount); err != nil {
		return nil, fmt.Errorf("failed to get row count: %w", err)
	}

	dataset := &TrainingDataset{
		Name:      tableName,
		Schema:    c.schema,
		Table:     tableName,
		RowCount:  rowCount,
		UpdatedAt: time.Now(),
	}

	return dataset, nil
}

// ExecuteQuery executes a custom SQL query
func (c *HANAClient) ExecuteQuery(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	return c.db.QueryContext(ctx, query, args...)
}

// ExecuteUpdate executes an update/insert/delete query
func (c *HANAClient) ExecuteUpdate(ctx context.Context, query string, args ...interface{}) (int64, error) {
	result, err := c.db.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, fmt.Errorf("failed to execute update: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected: %w", err)
	}

	return rowsAffected, nil
}

// CreateVectorTable creates a table optimized for vector embeddings
func (c *HANAClient) CreateVectorTable(ctx context.Context, tableName string, dimensions int) error {
	query := fmt.Sprintf(`
		CREATE COLUMN TABLE "%s"."%s" (
			ID NVARCHAR(36) PRIMARY KEY,
			VECTOR REAL ARRAY(%d),
			METADATA NCLOB,
			CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, c.schema, tableName, dimensions)

	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to create vector table: %w", err)
	}

	return nil
}

// InsertVectors inserts vector embeddings into HANA
func (c *HANAClient) InsertVectors(ctx context.Context, tableName string, vectors [][]float64, metadata []string) error {
	if len(vectors) == 0 {
		return nil
	}

	tx, err := c.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	query := fmt.Sprintf(`
		INSERT INTO "%s"."%s" (ID, VECTOR, METADATA)
		VALUES (?, ?, ?)
	`, c.schema, tableName)

	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for i, vector := range vectors {
		id := fmt.Sprintf("vec_%d_%d", time.Now().Unix(), i)
		meta := ""
		if i < len(metadata) {
			meta = metadata[i]
		}

		if _, err := stmt.ExecContext(ctx, id, vector, meta); err != nil {
			return fmt.Errorf("failed to insert vector: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// Helper function to join strings
func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}

// CreateTrainingResultsTable creates a table to store standardized training results.
func (c *HANAClient) CreateTrainingResultsTable(ctx context.Context) error {
	query := fmt.Sprintf(`
		CREATE COLUMN TABLE "%s"."TRAINING_RESULTS" (
			MODEL_ID NVARCHAR(256) PRIMARY KEY,
			MODEL_NAME NVARCHAR(256),
			VERSION NVARCHAR(64),
			TRAINING_DATASET NVARCHAR(512),
			EVALUATION_METRICS NCLOB,
			PARAMETERS NCLOB,
			TAGS NCLOB,
			CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, c.schema)

	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		// Ignore error if table already exists
		if strings.Contains(err.Error(), "table already exists") {
			return nil
		}
		return fmt.Errorf("failed to create TRAINING_RESULTS table: %w", err)
	}
	return nil
}

// InsertTrainingResult persists a standardized training result to HANA.
func (c *HANAClient) InsertTrainingResult(ctx context.Context, result contracts.TrainingResult) error {
	evalMetrics, err := json.Marshal(result.Evaluation)
	if err != nil {
		return fmt.Errorf("failed to marshal evaluation metrics: %w", err)
	}

	params, err := json.Marshal(result.Parameters)
	if err != nil {
		return fmt.Errorf("failed to marshal parameters: %w", err)
	}

	tags, err := json.Marshal(result.Tags)
	if err != nil {
		return fmt.Errorf("failed to marshal tags: %w", err)
	}

	query := fmt.Sprintf(`
		UPSERT "%s"."TRAINING_RESULTS" (
			MODEL_ID, MODEL_NAME, VERSION, TRAINING_DATASET, EVALUATION_METRICS, PARAMETERS, TAGS, CREATED_AT
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		WITH PRIMARY KEY
	`, c.schema)

	_, err = c.db.ExecContext(ctx, query,
		result.ModelID,
		result.ModelName,
		result.Version,
		result.TrainingDataset,
		string(evalMetrics),
		string(params),
		string(tags),
		result.CreatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to insert training result: %w", err)
	}

	return nil
}
