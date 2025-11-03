package main

import (
	"database/sql"
	"path/filepath"
	"testing"
)

func TestSQLitePersistenceOverwritesRows(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	p, err := NewSQLitePersistence(dbPath)
	if err != nil {
		t.Fatalf("NewSQLitePersistence: %v", err)
	}

	first := []map[string]any{{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}}
	if err := p.SaveTable("people", first); err != nil {
		t.Fatalf("SaveTable first: %v", err)
	}

	second := []map[string]any{{"id": 3, "name": "Charlie"}}
	if err := p.SaveTable("people", second); err != nil {
		t.Fatalf("SaveTable second: %v", err)
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	row := db.QueryRow("SELECT COUNT(*) FROM people")
	var count int
	if err := row.Scan(&count); err != nil {
		t.Fatalf("scan count: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected 1 row after overwrite, got %d", count)
	}

	var id int
	var name string
	row = db.QueryRow("SELECT id, name FROM people")
	if err := row.Scan(&id, &name); err != nil {
		t.Fatalf("scan row: %v", err)
	}
	if id != 3 || name != "Charlie" {
		t.Fatalf("unexpected row: id=%d name=%s", id, name)
	}
}
