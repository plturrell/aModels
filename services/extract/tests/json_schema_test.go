package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExtractSchemaFromJSONProfilesColumns(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sample.json")
	payload := `[
        {"id": 1, "name": "alpha", "flag": true},
        {"id": 2, "flag": false},
        {"id": 3, "name": null, "flag": true}
    ]`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatalf("write json: %v", err)
	}

	server := &extractServer{}
	nodes, edges, rows, err := server.extractSchemaFromJSON(path)
	if err != nil {
		t.Fatalf("extractSchemaFromJSON: %v", err)
	}

	if len(rows) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(rows))
	}

	if len(edges) == 0 {
		t.Fatalf("expected edges linking table to columns")
	}

	foundName := false
	foundFlag := false

	for _, node := range nodes {
		if node.Type != "column" {
			continue
		}
		switch node.Label {
		case "name":
			foundName = true
			if node.Props == nil {
				t.Fatalf("expected properties for column 'name'")
			}
			if node.Props["type"].(string) != "string" {
				t.Fatalf("expected name type string, got %v", node.Props["type"])
			}
			if nullable, ok := node.Props["nullable"].(bool); !ok || !nullable {
				t.Fatalf("expected name column to be nullable")
			}
		case "flag":
			foundFlag = true
			if node.Props == nil {
				t.Fatalf("expected properties for column 'flag'")
			}
			if node.Props["type"].(string) != "boolean" {
				t.Fatalf("expected flag type boolean, got %v", node.Props["type"])
			}
			if nullable, ok := node.Props["nullable"].(bool); !ok || nullable {
				t.Fatalf("expected flag column to be non-nullable")
			}
		}
	}

	if !foundName || !foundFlag {
		t.Fatalf("expected columns 'name' and 'flag' to be profiled")
	}
}
