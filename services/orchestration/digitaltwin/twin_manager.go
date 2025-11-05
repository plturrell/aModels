package digitaltwin

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
)

// PostgresTwinStore implements TwinStore using PostgreSQL.
type PostgresTwinStore struct {
	db     *sql.DB
	logger *log.Logger
}

// NewPostgresTwinStore creates a new PostgreSQL twin store.
func NewPostgresTwinStore(db *sql.DB, logger *log.Logger) *PostgresTwinStore {
	return &PostgresTwinStore{
		db:     db,
		logger: logger,
	}
}

// SaveTwin saves a twin to the database.
func (pts *PostgresTwinStore) SaveTwin(ctx context.Context, twin *Twin) error {
	stateJSON, _ := json.Marshal(twin.State)
	configJSON, _ := json.Marshal(twin.Configuration)
	metadataJSON, _ := json.Marshal(twin.Metadata)

	query := `
		INSERT INTO digital_twins (
			id, name, type, source_id, version, state, configuration, metadata, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (id) DO UPDATE
		SET name = EXCLUDED.name,
		    type = EXCLUDED.type,
		    source_id = EXCLUDED.source_id,
		    version = EXCLUDED.version,
		    state = EXCLUDED.state,
		    configuration = EXCLUDED.configuration,
		    metadata = EXCLUDED.metadata,
		    updated_at = EXCLUDED.updated_at
	`

	_, err := pts.db.ExecContext(ctx, query,
		twin.ID,
		twin.Name,
		twin.Type,
		twin.SourceID,
		twin.Version,
		stateJSON,
		configJSON,
		metadataJSON,
		twin.CreatedAt,
		twin.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to save twin: %w", err)
	}

	return nil
}

// GetTwin retrieves a twin by ID.
func (pts *PostgresTwinStore) GetTwin(ctx context.Context, id string) (*Twin, error) {
	query := `
		SELECT id, name, type, source_id, version, state, configuration, metadata, created_at, updated_at
		FROM digital_twins
		WHERE id = $1
	`

	var twin Twin
	var stateJSON, configJSON, metadataJSON []byte

	err := pts.db.QueryRowContext(ctx, query, id).Scan(
		&twin.ID,
		&twin.Name,
		&twin.Type,
		&twin.SourceID,
		&twin.Version,
		&stateJSON,
		&configJSON,
		&metadataJSON,
		&twin.CreatedAt,
		&twin.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("twin not found: %s", id)
		}
		return nil, fmt.Errorf("failed to get twin: %w", err)
	}

	// Parse JSON fields
	if err := json.Unmarshal(stateJSON, &twin.State); err != nil {
		return nil, fmt.Errorf("failed to parse state: %w", err)
	}
	if err := json.Unmarshal(configJSON, &twin.Configuration); err != nil {
		return nil, fmt.Errorf("failed to parse configuration: %w", err)
	}
	if err := json.Unmarshal(metadataJSON, &twin.Metadata); err != nil {
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}

	return &twin, nil
}

// ListTwins lists twins matching filters.
func (pts *PostgresTwinStore) ListTwins(ctx context.Context, filters TwinFilters) ([]*Twin, error) {
	query := `
		SELECT id, name, type, source_id, version, state, configuration, metadata, created_at, updated_at
		FROM digital_twins
		WHERE 1=1
	`
	args := []interface{}{}
	argIndex := 1

	if filters.Type != "" {
		query += fmt.Sprintf(" AND type = $%d", argIndex)
		args = append(args, filters.Type)
		argIndex++
	}
	if filters.SourceID != "" {
		query += fmt.Sprintf(" AND source_id = $%d", argIndex)
		args = append(args, filters.SourceID)
		argIndex++
	}

	query += " ORDER BY created_at DESC"

	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT $%d", argIndex)
		args = append(args, filters.Limit)
		argIndex++
	} else {
		query += " LIMIT 100"
	}

	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET $%d", argIndex)
		args = append(args, filters.Offset)
	}

	rows, err := pts.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query twins: %w", err)
	}
	defer rows.Close()

	var twins []*Twin
	for rows.Next() {
		var twin Twin
		var stateJSON, configJSON, metadataJSON []byte

		err := rows.Scan(
			&twin.ID,
			&twin.Name,
			&twin.Type,
			&twin.SourceID,
			&twin.Version,
			&stateJSON,
			&configJSON,
			&metadataJSON,
			&twin.CreatedAt,
			&twin.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan twin: %w", err)
		}

		// Parse JSON fields
		if err := json.Unmarshal(stateJSON, &twin.State); err != nil {
			return nil, fmt.Errorf("failed to parse state: %w", err)
		}
		if err := json.Unmarshal(configJSON, &twin.Configuration); err != nil {
			return nil, fmt.Errorf("failed to parse configuration: %w", err)
		}
		if err := json.Unmarshal(metadataJSON, &twin.Metadata); err != nil {
			return nil, fmt.Errorf("failed to parse metadata: %w", err)
		}

		twins = append(twins, &twin)
	}

	return twins, nil
}

// DeleteTwin deletes a twin.
func (pts *PostgresTwinStore) DeleteTwin(ctx context.Context, id string) error {
	query := `DELETE FROM digital_twins WHERE id = $1`

	_, err := pts.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete twin: %w", err)
	}

	return nil
}

