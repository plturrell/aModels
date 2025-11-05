package workflows

import (
	"context"
	"database/sql"
	"log"
	"os"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	// Create schema
	schema := `
		CREATE TABLE IF NOT EXISTS data_product_versions (
			id TEXT PRIMARY KEY,
			product_id TEXT NOT NULL,
			version TEXT NOT NULL,
			major INTEGER NOT NULL,
			minor INTEGER NOT NULL,
			patch INTEGER NOT NULL,
			pre_release TEXT,
			build_metadata TEXT,
			product_snapshot TEXT NOT NULL,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			created_by TEXT,
			deprecated BOOLEAN NOT NULL DEFAULT 0,
			deprecated_at TIMESTAMP,
			deprecation_reason TEXT,
			metadata TEXT
		);
	`
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("Failed to create schema: %v", err)
	}

	return db
}

func TestVersionManager_CreateVersion(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	vm := NewVersionManager(db, logger)
	ctx := context.Background()

	product := &CompleteDataProduct{
		DataElement: &struct {
			Identifier string
		}{Identifier: "test-product-1"},
	}

	version, err := vm.CreateVersion(ctx, "test-product-1", "v1.0.0", product, "test-user")
	if err != nil {
		t.Fatalf("CreateVersion() error = %v", err)
	}

	if version.ProductID != "test-product-1" {
		t.Errorf("ProductID = %v, want test-product-1", version.ProductID)
	}

	if version.Version != "v1.0.0" {
		t.Errorf("Version = %v, want v1.0.0", version.Version)
	}

	if version.Major != 1 || version.Minor != 0 || version.Patch != 0 {
		t.Errorf("Version parts = %d.%d.%d, want 1.0.0", version.Major, version.Minor, version.Patch)
	}
}

func TestVersionManager_GetLatestVersion(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	vm := NewVersionManager(db, logger)
	ctx := context.Background()

	product := &CompleteDataProduct{
		DataElement: &struct {
			Identifier string
		}{Identifier: "test-product-1"},
	}

	// Create multiple versions
	vm.CreateVersion(ctx, "test-product-1", "v1.0.0", product, "test-user")
	vm.CreateVersion(ctx, "test-product-1", "v1.1.0", product, "test-user")

	latest, err := vm.GetLatestVersion(ctx, "test-product-1")
	if err != nil {
		t.Fatalf("GetLatestVersion() error = %v", err)
	}

	if latest.Version != "v1.1.0" {
		t.Errorf("Latest version = %v, want v1.1.0", latest.Version)
	}
}

func TestParseSemanticVersion(t *testing.T) {
	tests := []struct {
		input    string
		major    int
		minor    int
		patch    int
		preRel   string
		build    string
		wantErr  bool
	}{
		{"v1.0.0", 1, 0, 0, "", "", false},
		{"1.2.3", 1, 2, 3, "", "", false},
		{"v1.2.3-alpha", 1, 2, 3, "alpha", "", false},
		{"v2.0.0+build.1", 2, 0, 0, "", "build.1", false},
		{"invalid", 0, 0, 0, "", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			major, minor, patch, preRel, build, err := parseSemanticVersion(tt.input)

			if tt.wantErr {
				if err == nil {
					t.Errorf("parseSemanticVersion() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("parseSemanticVersion() error = %v", err)
				return
			}

			if major != tt.major || minor != tt.minor || patch != tt.patch {
				t.Errorf("Version = %d.%d.%d, want %d.%d.%d", major, minor, patch, tt.major, tt.minor, tt.patch)
			}
		})
	}
}
