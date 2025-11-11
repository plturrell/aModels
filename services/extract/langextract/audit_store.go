package langextract

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

// AuditStore stores audit trail entries.
type AuditStore interface {
	SaveAuditEntry(ctx context.Context, entry *AuditTrail) error
	GetAuditEntry(ctx context.Context, auditID string) (*AuditTrail, error)
	QueryAuditEntries(ctx context.Context, filters AuditFilters) ([]*AuditTrail, error)
}

// AuditFilters defines filters for querying audit entries.
type AuditFilters struct {
	ExtractionID string
	User         string
	Context      string
	Operation    string
	StartTime    *time.Time
	EndTime      *time.Time
	Limit        int
	Offset       int
}

// PostgresAuditStore implements AuditStore using PostgreSQL.
type PostgresAuditStore struct {
	db     *sql.DB
	logger *log.Logger
}

// NewPostgresAuditStore creates a new PostgreSQL audit store.
func NewPostgresAuditStore(dsn string, logger *log.Logger) (*PostgresAuditStore, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &PostgresAuditStore{
		db:     db,
		logger: logger,
	}, nil
}

// Close closes the database connection.
func (pas *PostgresAuditStore) Close() error {
	return pas.db.Close()
}

// SaveAuditEntry saves an audit trail entry to PostgreSQL.
func (pas *PostgresAuditStore) SaveAuditEntry(ctx context.Context, entry *AuditTrail) error {
	requestJSON, err := json.Marshal(entry.Request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	responseJSON, err := json.Marshal(entry.Response)
	if err != nil {
		return fmt.Errorf("failed to marshal response: %w", err)
	}

	metadataJSON, _ := json.Marshal(entry.Metadata)
	qualityMetricsJSON, _ := json.Marshal(entry.QualityMetrics)
	resourceUsageJSON, _ := json.Marshal(entry.ResourceUsage)

	query := `
		INSERT INTO langextract_audit_trail (
			id, extraction_id, timestamp, "user", context, operation,
			request, response, processing_time_ms, confidence, schema_version,
			quality_metrics, resource_usage, metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
	`

	_, err = pas.db.ExecContext(ctx, query,
		entry.ID,
		entry.ExtractionID,
		entry.Timestamp,
		entry.User,
		entry.Context,
		entry.Operation,
		requestJSON,
		responseJSON,
		entry.ProcessingTime.Milliseconds(),
		entry.Confidence,
		entry.SchemaVersion,
		qualityMetricsJSON,
		resourceUsageJSON,
		metadataJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to insert audit entry: %w", err)
	}

	return nil
}

// GetAuditEntry retrieves an audit trail entry by ID.
func (pas *PostgresAuditStore) GetAuditEntry(ctx context.Context, auditID string) (*AuditTrail, error) {
	query := `
		SELECT id, extraction_id, timestamp, "user", context, operation,
		       request, response, processing_time_ms, confidence, schema_version,
		       quality_metrics, resource_usage, metadata
		FROM langextract_audit_trail
		WHERE id = $1
	`

	var at AuditTrail
	var requestJSON, responseJSON, metadataJSON, qualityMetricsJSON, resourceUsageJSON []byte

	err := pas.db.QueryRowContext(ctx, query, auditID).Scan(
		&at.ID,
		&at.ExtractionID,
		&at.Timestamp,
		&at.User,
		&at.Context,
		&at.Operation,
		&requestJSON,
		&responseJSON,
		&at.ProcessingTime,
		&at.Confidence,
		&at.SchemaVersion,
		&qualityMetricsJSON,
		&resourceUsageJSON,
		&metadataJSON,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("audit entry not found: %s", auditID)
		}
		return nil, fmt.Errorf("failed to get audit entry: %w", err)
	}

	// Parse JSON fields
	if err := json.Unmarshal(requestJSON, &at.Request); err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}
	if err := json.Unmarshal(responseJSON, &at.Response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &at.Metadata); err != nil {
			return nil, fmt.Errorf("failed to parse metadata: %w", err)
		}
	}
	if len(qualityMetricsJSON) > 0 {
		if err := json.Unmarshal(qualityMetricsJSON, &at.QualityMetrics); err != nil {
			return nil, fmt.Errorf("failed to parse quality metrics: %w", err)
		}
	}
	if len(resourceUsageJSON) > 0 {
		if err := json.Unmarshal(resourceUsageJSON, &at.ResourceUsage); err != nil {
			return nil, fmt.Errorf("failed to parse resource usage: %w", err)
		}
	}

	// Convert processing time from milliseconds to Duration
	at.ProcessingTime = time.Duration(at.ProcessingTime) * time.Millisecond

	return &at, nil
}

// QueryAuditEntries queries audit trail entries with filters.
func (pas *PostgresAuditStore) QueryAuditEntries(ctx context.Context, filters AuditFilters) ([]*AuditTrail, error) {
	query := `
		SELECT id, extraction_id, timestamp, "user", context, operation,
		       request, response, processing_time_ms, confidence, schema_version,
		       quality_metrics, resource_usage, metadata
		FROM langextract_audit_trail
		WHERE 1=1
	`
	args := []interface{}{}
	argIndex := 1

	if filters.ExtractionID != "" {
		query += fmt.Sprintf(" AND extraction_id = $%d", argIndex)
		args = append(args, filters.ExtractionID)
		argIndex++
	}
	if filters.User != "" {
		query += fmt.Sprintf(" AND \"user\" = $%d", argIndex)
		args = append(args, filters.User)
		argIndex++
	}
	if filters.Context != "" {
		query += fmt.Sprintf(" AND context = $%d", argIndex)
		args = append(args, filters.Context)
		argIndex++
	}
	if filters.Operation != "" {
		query += fmt.Sprintf(" AND operation = $%d", argIndex)
		args = append(args, filters.Operation)
		argIndex++
	}
	if filters.StartTime != nil {
		query += fmt.Sprintf(" AND timestamp >= $%d", argIndex)
		args = append(args, *filters.StartTime)
		argIndex++
	}
	if filters.EndTime != nil {
		query += fmt.Sprintf(" AND timestamp <= $%d", argIndex)
		args = append(args, *filters.EndTime)
		argIndex++
	}

	query += " ORDER BY timestamp DESC"

	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT $%d", argIndex)
		args = append(args, filters.Limit)
		argIndex++
	} else {
		query += " LIMIT 100" // Default limit
	}

	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET $%d", argIndex)
		args = append(args, filters.Offset)
	}

	rows, err := pas.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query audit entries: %w", err)
	}
	defer rows.Close()

	var entries []*AuditTrail
	for rows.Next() {
		var at AuditTrail
		var requestJSON, responseJSON, metadataJSON, qualityMetricsJSON, resourceUsageJSON []byte
		var processingTimeMs int64

		err := rows.Scan(
			&at.ID,
			&at.ExtractionID,
			&at.Timestamp,
			&at.User,
			&at.Context,
			&at.Operation,
			&requestJSON,
			&responseJSON,
			&processingTimeMs,
			&at.Confidence,
			&at.SchemaVersion,
			&qualityMetricsJSON,
			&resourceUsageJSON,
			&metadataJSON,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan audit entry: %w", err)
		}

		// Parse JSON fields
		if err := json.Unmarshal(requestJSON, &at.Request); err != nil {
			return nil, fmt.Errorf("failed to parse request: %w", err)
		}
		if err := json.Unmarshal(responseJSON, &at.Response); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &at.Metadata); err != nil {
				return nil, fmt.Errorf("failed to parse metadata: %w", err)
			}
		}
		if len(qualityMetricsJSON) > 0 {
			if err := json.Unmarshal(qualityMetricsJSON, &at.QualityMetrics); err != nil {
				return nil, fmt.Errorf("failed to parse quality metrics: %w", err)
			}
		}
		if len(resourceUsageJSON) > 0 {
			if err := json.Unmarshal(resourceUsageJSON, &at.ResourceUsage); err != nil {
				return nil, fmt.Errorf("failed to parse resource usage: %w", err)
			}
		}

		at.ProcessingTime = time.Duration(processingTimeMs) * time.Millisecond
		entries = append(entries, &at)
	}

	return entries, nil
}

