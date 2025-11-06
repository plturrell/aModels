package regulatory

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// RegulatorySchemaRepository stores and manages regulatory schemas.
type RegulatorySchemaRepository struct {
	db     *sql.DB
	logger *log.Logger
}

// SchemaVersion represents a version of a regulatory schema.
type SchemaVersion struct {
	ID              string
	RegulatoryType  string
	Version         string
	DocumentSource  string
	DocumentVersion string
	Spec            *RegulatorySpec
	CreatedAt       time.Time
	CreatedBy       string
	IsReference     bool // True if this is the official reference schema
	Status          string // "draft", "approved", "deprecated"
}

// SchemaChange represents a change between schema versions.
type SchemaChange struct {
	ID              string
	FromVersion     string
	ToVersion       string
	ChangeType      string // "added", "modified", "removed"
	FieldID         string
	Description     string
	Impact          string // "low", "medium", "high"
	Breaking        bool
	CreatedAt       time.Time
}

// NewRegulatorySchemaRepository creates a new schema repository.
func NewRegulatorySchemaRepository(db *sql.DB, logger *log.Logger) *RegulatorySchemaRepository {
	return &RegulatorySchemaRepository{
		db:     db,
		logger: logger,
	}
}

// SaveSchema saves a regulatory schema.
func (rsr *RegulatorySchemaRepository) SaveSchema(ctx context.Context, spec *RegulatorySpec, user string) error {
	specJSON, err := json.Marshal(spec)
	if err != nil {
		return fmt.Errorf("failed to marshal spec: %w", err)
	}

	query := `
		INSERT INTO regulatory_schemas (
			id, regulatory_type, version, document_source, document_version,
			spec_json, created_at, created_by, is_reference, status
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (id) DO UPDATE
		SET spec_json = EXCLUDED.spec_json,
		    updated_at = NOW(),
		    status = EXCLUDED.status
	`

	_, err = rsr.db.ExecContext(ctx, query,
		spec.ID,
		spec.RegulatoryType,
		spec.Version,
		spec.DocumentSource,
		spec.DocumentVersion,
		specJSON,
		time.Now(),
		user,
		false, // Not reference by default
		"draft",
	)

	if err != nil {
		return fmt.Errorf("failed to save schema: %w", err)
	}

	return nil
}

// GetSchema retrieves a schema by ID.
func (rsr *RegulatorySchemaRepository) GetSchema(ctx context.Context, id string) (*RegulatorySpec, error) {
	query := `
		SELECT spec_json
		FROM regulatory_schemas
		WHERE id = $1
	`

	var specJSON []byte
	err := rsr.db.QueryRowContext(ctx, query, id).Scan(&specJSON)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("schema not found: %s", id)
		}
		return nil, fmt.Errorf("failed to get schema: %w", err)
	}

	var spec RegulatorySpec
	if err := json.Unmarshal(specJSON, &spec); err != nil {
		return nil, fmt.Errorf("failed to unmarshal spec: %w", err)
	}

	return &spec, nil
}

// GetReferenceSchema retrieves the reference schema for a regulatory type.
func (rsr *RegulatorySchemaRepository) GetReferenceSchema(ctx context.Context, regulatoryType string, version string) (*RegulatorySpec, error) {
	query := `
		SELECT spec_json
		FROM regulatory_schemas
		WHERE regulatory_type = $1 AND is_reference = true
	`

	if version != "" {
		query += " AND version = $2"
	} else {
		query += " ORDER BY created_at DESC LIMIT 1"
	}

	var specJSON []byte
	var err error

	if version != "" {
		err = rsr.db.QueryRowContext(ctx, query, regulatoryType, version).Scan(&specJSON)
	} else {
		err = rsr.db.QueryRowContext(ctx, query, regulatoryType).Scan(&specJSON)
	}

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("reference schema not found for %s", regulatoryType)
		}
		return nil, fmt.Errorf("failed to get reference schema: %w", err)
	}

	var spec RegulatorySpec
	if err := json.Unmarshal(specJSON, &spec); err != nil {
		return nil, fmt.Errorf("failed to unmarshal spec: %w", err)
	}

	return &spec, nil
}

// ListSchemas lists schemas for a regulatory type.
func (rsr *RegulatorySchemaRepository) ListSchemas(ctx context.Context, regulatoryType string, limit int) ([]*RegulatorySpec, error) {
	if limit == 0 {
		limit = 100
	}

	query := `
		SELECT spec_json
		FROM regulatory_schemas
		WHERE regulatory_type = $1
		ORDER BY created_at DESC
		LIMIT $2
	`

	rows, err := rsr.db.QueryContext(ctx, query, regulatoryType, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query schemas: %w", err)
	}
	defer rows.Close()

	var specs []*RegulatorySpec
	for rows.Next() {
		var specJSON []byte
		if err := rows.Scan(&specJSON); err != nil {
			continue
		}

		var spec RegulatorySpec
		if err := json.Unmarshal(specJSON, &spec); err != nil {
			continue
		}

		specs = append(specs, &spec)
	}

	return specs, nil
}

// CompareSchemas compares two schema versions and identifies changes.
func (rsr *RegulatorySchemaRepository) CompareSchemas(ctx context.Context, fromID string, toID string) ([]SchemaChange, error) {
	fromSpec, err := rsr.GetSchema(ctx, fromID)
	if err != nil {
		return nil, fmt.Errorf("failed to get from schema: %w", err)
	}

	toSpec, err := rsr.GetSchema(ctx, toID)
	if err != nil {
		return nil, fmt.Errorf("failed to get to schema: %w", err)
	}

	var changes []SchemaChange

	// Compare field definitions
	fromFields := make(map[string]*FieldDefinition)
	for i := range fromSpec.FieldDefinitions {
		field := &fromSpec.FieldDefinitions[i]
		fromFields[field.FieldID] = field
	}

	toFields := make(map[string]*FieldDefinition)
	for i := range toSpec.FieldDefinitions {
		field := &toSpec.FieldDefinitions[i]
		toFields[field.FieldID] = field
	}

	// Find added fields
	for fieldID, field := range toFields {
		if _, exists := fromFields[fieldID]; !exists {
			changes = append(changes, SchemaChange{
				ID:          fmt.Sprintf("change-%d", len(changes)),
				FromVersion: fromSpec.Version,
				ToVersion:   toSpec.Version,
				ChangeType:  "added",
				FieldID:     fieldID,
				Description: fmt.Sprintf("Field %s added", field.FieldName),
				Impact:      "medium",
				Breaking:    field.Required,
				CreatedAt:   time.Now(),
			})
		}
	}

	// Find removed fields
	for fieldID, field := range fromFields {
		if _, exists := toFields[fieldID]; !exists {
			changes = append(changes, SchemaChange{
				ID:          fmt.Sprintf("change-%d", len(changes)),
				FromVersion: fromSpec.Version,
				ToVersion:   toSpec.Version,
				ChangeType:  "removed",
				FieldID:     fieldID,
				Description: fmt.Sprintf("Field %s removed", field.FieldName),
				Impact:      "high",
				Breaking:    true,
				CreatedAt:   time.Now(),
			})
		}
	}

	// Find modified fields
	for fieldID, toField := range toFields {
		if fromField, exists := fromFields[fieldID]; exists {
			if toField.FieldType != fromField.FieldType || toField.Required != fromField.Required {
				changes = append(changes, SchemaChange{
					ID:          fmt.Sprintf("change-%d", len(changes)),
					FromVersion: fromSpec.Version,
					ToVersion:   toSpec.Version,
					ChangeType:  "modified",
					FieldID:     fieldID,
					Description: fmt.Sprintf("Field %s modified", toField.FieldName),
					Impact:      "medium",
					Breaking:    toField.Required != fromField.Required,
					CreatedAt:   time.Now(),
				})
			}
		}
	}

	return changes, nil
}

// SetReferenceSchema sets a schema as the reference schema.
func (rsr *RegulatorySchemaRepository) SetReferenceSchema(ctx context.Context, id string) error {
	// First, unset all reference schemas for this regulatory type
	query := `
		UPDATE regulatory_schemas
		SET is_reference = false
		WHERE id IN (
			SELECT id FROM regulatory_schemas s1
			WHERE s1.regulatory_type = (
				SELECT regulatory_type FROM regulatory_schemas WHERE id = $1
			)
		)
	`
	rsr.db.ExecContext(ctx, query, id)

	// Set the specified schema as reference
	query = `
		UPDATE regulatory_schemas
		SET is_reference = true, status = 'approved'
		WHERE id = $1
	`

	_, err := rsr.db.ExecContext(ctx, query, id)
	return err
}

