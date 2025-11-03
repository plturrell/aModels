package sqlite_test

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	checkpoint "github.com/langchain-ai/langgraph-go/pkg/checkpoint/sqlite"
)

func TestStoreCRUD(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	store, err := checkpoint.NewStore(db)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	if err := store.Save(ctx, "foo", []byte("bar")); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	payload, err := store.Load(ctx, "foo")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if string(payload) != "bar" {
		t.Fatalf("unexpected payload: %s", string(payload))
	}

	if err := store.Save(ctx, "foo", []byte("baz")); err != nil {
		t.Fatalf("Save update failed: %v", err)
	}
	payload, err = store.Load(ctx, "foo")
	if err != nil {
		t.Fatalf("Load after update failed: %v", err)
	}
	if string(payload) != "baz" {
		t.Fatalf("unexpected payload after update: %s", string(payload))
	}

	if err := store.Delete(ctx, "foo"); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if _, err := store.Load(ctx, "foo"); err == nil {
		t.Fatal("expected load error after delete")
	}
}

func TestStoreCustomTableName(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	tableName := "custom_checkpoints"
	store, err := checkpoint.NewStore(db, checkpoint.WithTableName(tableName))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()
	if err := store.Save(ctx, "key", []byte("value")); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	query := "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
	row := db.QueryRowContext(ctx, query, tableName)
	var name string
	if err := row.Scan(&name); err != nil {
		t.Fatalf("expected table to exist: %v", err)
	}
}
