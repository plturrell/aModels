package workflows

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// DataProductVersion represents a version of a data product.
type DataProductVersion struct {
	ID                string                 `json:"id"`
	ProductID         string                 `json:"product_id"`
	Version           string                 `json:"version"` // Semantic version: v1.0.0
	Major             int                    `json:"major"`
	Minor             int                    `json:"minor"`
	Patch             int                    `json:"patch"`
	PreRelease        string                 `json:"pre_release,omitempty"`
	BuildMetadata     string                 `json:"build_metadata,omitempty"`
	ProductSnapshot   json.RawMessage        `json:"product_snapshot"` // CompleteDataProduct as JSON
	CreatedAt         time.Time              `json:"created_at"`
	CreatedBy         string                 `json:"created_by,omitempty"`
	Deprecated        bool                   `json:"deprecated"`
	DeprecatedAt      *time.Time             `json:"deprecated_at,omitempty"`
	DeprecationReason string                 `json:"deprecation_reason,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// VersionManager manages data product versions.
type VersionManager struct {
	db     *sql.DB
	logger *log.Logger
}

// NewVersionManager creates a new version manager.
func NewVersionManager(db *sql.DB, logger *log.Logger) *VersionManager {
	return &VersionManager{
		db:     db,
		logger: logger,
	}
}

// CreateVersion creates a new version of a data product.
func (vm *VersionManager) CreateVersion(ctx context.Context, productID string, version string, product *CompleteDataProduct, createdBy string) (*DataProductVersion, error) {
	// Parse semantic version
	major, minor, patch, preRelease, buildMetadata, err := parseSemanticVersion(version)
	if err != nil {
		return nil, fmt.Errorf("invalid semantic version: %w", err)
	}

	// Serialize product snapshot
	productJSON, err := json.Marshal(product)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal product: %w", err)
	}

	// Create version ID
	versionID := fmt.Sprintf("%s-%s", productID, version)

	// Insert into database
	query := `
		INSERT INTO data_product_versions (
			id, product_id, version, major, minor, patch, pre_release, build_metadata,
			product_snapshot, created_at, created_by, deprecated
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`

	now := time.Now()
	_, err = vm.db.ExecContext(ctx, query,
		versionID,
		productID,
		version,
		major,
		minor,
		patch,
		preRelease,
		buildMetadata,
		productJSON,
		now,
		createdBy,
		false,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to insert version: %w", err)
	}

	if vm.logger != nil {
		vm.logger.Printf("Created version %s for product %s", version, productID)
	}

	return &DataProductVersion{
		ID:              versionID,
		ProductID:       productID,
		Version:         version,
		Major:           major,
		Minor:           minor,
		Patch:           patch,
		PreRelease:      preRelease,
		BuildMetadata:   buildMetadata,
		ProductSnapshot: productJSON,
		CreatedAt:       now,
		CreatedBy:        createdBy,
		Deprecated:      false,
	}, nil
}

// GetVersion retrieves a specific version of a data product.
func (vm *VersionManager) GetVersion(ctx context.Context, productID string, version string) (*DataProductVersion, error) {
	versionID := fmt.Sprintf("%s-%s", productID, version)
	query := `
		SELECT id, product_id, version, major, minor, patch, pre_release, build_metadata,
		       product_snapshot, created_at, created_by, deprecated, deprecated_at, deprecation_reason, metadata
		FROM data_product_versions
		WHERE id = $1
	`

	var v DataProductVersion
	var productJSON []byte
	var deprecatedAt sql.NullTime
	var deprecationReason sql.NullString
	var metadataJSON sql.NullString

	err := vm.db.QueryRowContext(ctx, query, versionID).Scan(
		&v.ID,
		&v.ProductID,
		&v.Version,
		&v.Major,
		&v.Minor,
		&v.Patch,
		&v.PreRelease,
		&v.BuildMetadata,
		&productJSON,
		&v.CreatedAt,
		&v.CreatedBy,
		&v.Deprecated,
		&deprecatedAt,
		&deprecationReason,
		&metadataJSON,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("version %s not found for product %s", version, productID)
		}
		return nil, fmt.Errorf("failed to get version: %w", err)
	}

	v.ProductSnapshot = productJSON
	if deprecatedAt.Valid {
		v.DeprecatedAt = &deprecatedAt.Time
	}
	if deprecationReason.Valid {
		v.DeprecationReason = deprecationReason.String
	}
	if metadataJSON.Valid && metadataJSON.String != "" {
		if err := json.Unmarshal([]byte(metadataJSON.String), &v.Metadata); err != nil {
			if vm.logger != nil {
				vm.logger.Printf("Warning: Failed to parse metadata JSON: %v", err)
			}
		}
	}

	return &v, nil
}

// GetLatestVersion retrieves the latest non-deprecated version of a data product.
func (vm *VersionManager) GetLatestVersion(ctx context.Context, productID string) (*DataProductVersion, error) {
	query := `
		SELECT id, product_id, version, major, minor, patch, pre_release, build_metadata,
		       product_snapshot, created_at, created_by, deprecated, deprecated_at, deprecation_reason, metadata
		FROM data_product_versions
		WHERE product_id = $1 AND deprecated = false
		ORDER BY major DESC, minor DESC, patch DESC, created_at DESC
		LIMIT 1
	`

	var v DataProductVersion
	var productJSON []byte
	var deprecatedAt sql.NullTime
	var deprecationReason sql.NullString
	var metadataJSON sql.NullString

	err := vm.db.QueryRowContext(ctx, query, productID).Scan(
		&v.ID,
		&v.ProductID,
		&v.Version,
		&v.Major,
		&v.Minor,
		&v.Patch,
		&v.PreRelease,
		&v.BuildMetadata,
		&productJSON,
		&v.CreatedAt,
		&v.CreatedBy,
		&v.Deprecated,
		&deprecatedAt,
		&deprecationReason,
		&metadataJSON,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("no versions found for product %s", productID)
		}
		return nil, fmt.Errorf("failed to get latest version: %w", err)
	}

	v.ProductSnapshot = productJSON
	if deprecatedAt.Valid {
		v.DeprecatedAt = &deprecatedAt.Time
	}
	if deprecationReason.Valid {
		v.DeprecationReason = deprecationReason.String
	}
	if metadataJSON.Valid && metadataJSON.String != "" {
		if err := json.Unmarshal([]byte(metadataJSON.String), &v.Metadata); err != nil {
			if vm.logger != nil {
				vm.logger.Printf("Warning: Failed to parse metadata JSON: %v", err)
			}
		}
	}

	return &v, nil
}

// ListVersions lists all versions of a data product.
func (vm *VersionManager) ListVersions(ctx context.Context, productID string) ([]*DataProductVersion, error) {
	query := `
		SELECT id, product_id, version, major, minor, patch, pre_release, build_metadata,
		       product_snapshot, created_at, created_by, deprecated, deprecated_at, deprecation_reason, metadata
		FROM data_product_versions
		WHERE product_id = $1
		ORDER BY major DESC, minor DESC, patch DESC, created_at DESC
	`

	rows, err := vm.db.QueryContext(ctx, query, productID)
	if err != nil {
		return nil, fmt.Errorf("failed to query versions: %w", err)
	}
	defer rows.Close()

	var versions []*DataProductVersion
	for rows.Next() {
		var v DataProductVersion
		var productJSON []byte
		var deprecatedAt sql.NullTime
		var deprecationReason sql.NullString
		var metadataJSON sql.NullString

		err := rows.Scan(
			&v.ID,
			&v.ProductID,
			&v.Version,
			&v.Major,
			&v.Minor,
			&v.Patch,
			&v.PreRelease,
			&v.BuildMetadata,
			&productJSON,
			&v.CreatedAt,
			&v.CreatedBy,
			&v.Deprecated,
			&deprecatedAt,
			&deprecationReason,
			&metadataJSON,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan version: %w", err)
		}

		v.ProductSnapshot = productJSON
		if deprecatedAt.Valid {
			v.DeprecatedAt = &deprecatedAt.Time
		}
		if deprecationReason.Valid {
			v.DeprecationReason = deprecationReason.String
		}
		if metadataJSON.Valid && metadataJSON.String != "" {
			if err := json.Unmarshal([]byte(metadataJSON.String), &v.Metadata); err != nil {
				if vm.logger != nil {
					vm.logger.Printf("Warning: Failed to parse metadata JSON: %v", err)
				}
			}
		}

		versions = append(versions, &v)
	}

	return versions, nil
}

// DeprecateVersion marks a version as deprecated.
func (vm *VersionManager) DeprecateVersion(ctx context.Context, productID string, version string, reason string) error {
	versionID := fmt.Sprintf("%s-%s", productID, version)
	query := `
		UPDATE data_product_versions
		SET deprecated = true, deprecated_at = $1, deprecation_reason = $2
		WHERE id = $3
	`

	now := time.Now()
	_, err := vm.db.ExecContext(ctx, query, now, reason, versionID)
	if err != nil {
		return fmt.Errorf("failed to deprecate version: %w", err)
	}

	if vm.logger != nil {
		vm.logger.Printf("Deprecated version %s for product %s: %s", version, productID, reason)
	}

	return nil
}

// CompareVersions compares two versions and returns differences.
func (vm *VersionManager) CompareVersions(ctx context.Context, productID string, version1 string, version2 string) (*VersionComparison, error) {
	v1, err := vm.GetVersion(ctx, productID, version1)
	if err != nil {
		return nil, fmt.Errorf("failed to get version %s: %w", version1, err)
	}

	v2, err := vm.GetVersion(ctx, productID, version2)
	if err != nil {
		return nil, fmt.Errorf("failed to get version %s: %w", version2, err)
	}

	var product1, product2 CompleteDataProduct
	if err := json.Unmarshal(v1.ProductSnapshot, &product1); err != nil {
		return nil, fmt.Errorf("failed to unmarshal product snapshot for %s: %w", version1, err)
	}
	if err := json.Unmarshal(v2.ProductSnapshot, &product2); err != nil {
		return nil, fmt.Errorf("failed to unmarshal product snapshot for %s: %w", version2, err)
	}

	comparison := &VersionComparison{
		ProductID:   productID,
		Version1:    version1,
		Version2:    version2,
		Differences: []VersionDifference{},
	}

	// Compare quality scores
	if product1.QualityMetrics != nil && product2.QualityMetrics != nil {
		if product1.QualityMetrics.QualityScore != product2.QualityMetrics.QualityScore {
			comparison.Differences = append(comparison.Differences, VersionDifference{
				Field:     "quality_score",
				OldValue:  product1.QualityMetrics.QualityScore,
				NewValue:  product2.QualityMetrics.QualityScore,
				ChangeType: "modified",
			})
		}
	}

	// Compare lifecycle states
	if product1.EnhancedElement != nil && product2.EnhancedElement != nil {
		if product1.EnhancedElement.LifecycleState != product2.EnhancedElement.LifecycleState {
			comparison.Differences = append(comparison.Differences, VersionDifference{
				Field:      "lifecycle_state",
				OldValue:   product1.EnhancedElement.LifecycleState,
				NewValue:   product2.EnhancedElement.LifecycleState,
				ChangeType: "modified",
			})
		}
	}

	// Compare changelog entries (if available)
	if v1.Metadata != nil && v2.Metadata != nil {
		changelog1, _ := v1.Metadata["changelog"].([]interface{})
		changelog2, _ := v2.Metadata["changelog"].([]interface{})
		if len(changelog1) != len(changelog2) {
			comparison.Differences = append(comparison.Differences, VersionDifference{
				Field:      "changelog",
				OldValue:   len(changelog1),
				NewValue:   len(changelog2),
				ChangeType: "modified",
			})
		}
	}

	return comparison, nil
}

// VersionComparison represents a comparison between two versions.
type VersionComparison struct {
	ProductID   string            `json:"product_id"`
	Version1    string            `json:"version1"`
	Version2    string            `json:"version2"`
	Differences []VersionDifference `json:"differences"`
}

// VersionDifference represents a difference between two versions.
type VersionDifference struct {
	Field      string      `json:"field"`
	OldValue   interface{} `json:"old_value"`
	NewValue   interface{} `json:"new_value"`
	ChangeType string      `json:"change_type"` // "added", "removed", "modified"
}

// parseSemanticVersion parses a semantic version string (v1.2.3-alpha+build.1).
func parseSemanticVersion(version string) (major int, minor int, patch int, preRelease string, buildMetadata string, err error) {
	// Remove 'v' prefix if present
	version = strings.TrimPrefix(version, "v")

	// Split on '+' for build metadata
	parts := strings.SplitN(version, "+", 2)
	if len(parts) == 2 {
		buildMetadata = parts[1]
	}
	version = parts[0]

	// Split on '-' for pre-release
	parts = strings.SplitN(version, "-", 2)
	if len(parts) == 2 {
		preRelease = parts[1]
	}
	version = parts[0]

	// Parse major.minor.patch
	versionRegex := regexp.MustCompile(`^(\d+)\.(\d+)\.(\d+)$`)
	matches := versionRegex.FindStringSubmatch(version)
	if len(matches) != 4 {
		return 0, 0, 0, "", "", fmt.Errorf("invalid version format: %s", version)
	}

	major, _ = strconv.Atoi(matches[1])
	minor, _ = strconv.Atoi(matches[2])
	patch, _ = strconv.Atoi(matches[3])

	return major, minor, patch, preRelease, buildMetadata, nil
}

