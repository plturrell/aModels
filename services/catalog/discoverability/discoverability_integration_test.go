package discoverability

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

	// Create minimal schema
	schema := `
		CREATE TABLE IF NOT EXISTS tags (
			id TEXT PRIMARY KEY,
			name TEXT UNIQUE,
			category TEXT,
			parent_tag_id TEXT,
			description TEXT,
			usage_count INTEGER DEFAULT 0,
			metadata TEXT,
			created_at TIMESTAMP,
			updated_at TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS product_tags (
			product_id TEXT,
			tag_id TEXT,
			confidence REAL,
			source TEXT,
			created_at TIMESTAMP,
			PRIMARY KEY (product_id, tag_id)
		);
		CREATE TABLE IF NOT EXISTS search_history (
			id TEXT PRIMARY KEY,
			query TEXT,
			result_count INTEGER,
			timestamp TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS product_usage_stats (
			product_id TEXT PRIMARY KEY,
			total_views INTEGER DEFAULT 0,
			access_requests INTEGER DEFAULT 0,
			updated_at TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS access_requests (
			id TEXT PRIMARY KEY,
			product_id TEXT,
			requester_id TEXT,
			status TEXT,
			requested_at TIMESTAMP
		);
	`
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("Failed to create schema: %v", err)
	}

	return db
}

func TestTagManager_CreateTag(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	manager := NewTagManager(db, logger)

	ctx := context.Background()
	tag := &Tag{
		ID:          "tag-1",
		Name:        "test-tag",
		Category:    "domain",
		Description: "Test tag",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	err := manager.CreateTag(ctx, tag)
	if err != nil {
		t.Fatalf("CreateTag() error = %v", err)
	}

	// Retrieve tag
	retrieved, err := manager.GetTag(ctx, tag.ID)
	if err != nil {
		t.Fatalf("GetTag() error = %v", err)
	}

	if retrieved.Name != tag.Name {
		t.Errorf("Tag name = %v, want %v", retrieved.Name, tag.Name)
	}
}

func TestTagManager_AddTagToProduct(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	manager := NewTagManager(db, logger)

	ctx := context.Background()
	
	// Create tag
	tag := &Tag{
		ID:        "tag-1",
		Name:      "test-tag",
		Category:  "domain",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	manager.CreateTag(ctx, tag)

	// Add tag to product
	err := manager.AddTagToProduct(ctx, "product-1", tag.ID, "manual", 1.0, "test-user")
	if err != nil {
		t.Fatalf("AddTagToProduct() error = %v", err)
	}

	// Get product tags
	productTags, err := manager.GetProductTags(ctx, "product-1")
	if err != nil {
		t.Fatalf("GetProductTags() error = %v", err)
	}

	if len(productTags) == 0 {
		t.Error("Expected at least one tag")
	}
}

func TestCrossTeamSearch_Search(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	search := NewCrossTeamSearch(db, logger)

	ctx := context.Background()
	req := SearchRequest{
		Query:  "test",
		Limit:  10,
		Offset: 0,
		SortBy: "relevance",
	}

	results, err := search.Search(ctx, req)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if results == nil {
		t.Error("Expected search results")
	}
}

func TestMarketplace_RequestAccess(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	marketplace := NewMarketplace(db, logger)

	ctx := context.Background()
	req := AccessRequest{
		ProductID:     "product-1",
		RequesterID:   "user-1",
		RequesterTeam: "team-1",
		Reason:        "Test access request",
	}

	err := marketplace.RequestAccess(ctx, req)
	if err != nil {
		t.Fatalf("RequestAccess() error = %v", err)
	}

	if req.ID == "" {
		t.Error("Expected request ID to be set")
	}
}

