package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNormalizeGraphDeduplicatesAndSetsRoot(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "catalog.json")
	cat, err := NewCatalog(tmpFile)
	if err != nil {
		t.Fatalf("NewCatalog: %v", err)
	}

	input := normalizationInput{
		Nodes: []Node{
			{ID: "doc-1", Type: "document", Label: "Document"},
			{ID: "doc-1", Type: "document", Label: "Duplicate"},
			{ID: "doc-1.col", Type: "column", Label: "Name", Props: map[string]any{"type": "string"}},
			{ID: "doc-1.col", Type: "column", Label: "Name", Props: map[string]any{"type": "string"}},
		},
		Edges: []Edge{
			{SourceID: "doc-1", TargetID: "doc-1.col", Label: "HAS_COLUMN"},
			{SourceID: "doc-1", TargetID: "doc-1.col", Label: "HAS_COLUMN"},
		},
		ProjectID: "proj-1",
		SystemID:  "sys-1",
		Catalog:   cat,
	}

	result := normalizeGraph(input)

	if result.RootNodeID != "doc-1" {
		t.Fatalf("expected root node doc-1, got %q", result.RootNodeID)
	}
	if len(result.Nodes) < 2 {
		t.Fatalf("expected at least 2 nodes, got %d", len(result.Nodes))
	}
	if len(result.Edges) < 1 {
		t.Fatalf("expected at least one edge, got %d", len(result.Edges))
	}
	if got := result.Stats["catalog_nodes_added"]; got.(int) == 0 {
		t.Fatalf("expected catalog nodes added stat to be >0, got %v", got)
	}
	if _, err := os.Stat(tmpFile); err != nil {
		t.Fatalf("expected catalog to be saved: %v", err)
	}
}
