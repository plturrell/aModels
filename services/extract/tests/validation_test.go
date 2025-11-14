package main

import (
	"log"
	"os"
	"testing"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

func TestValidateNodes(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	tests := []struct {
		name     string
		nodes    []Node
		wantValid bool
		wantErrors int
	}{
		{
			name: "valid nodes",
			nodes: []Node{
				{ID: "node1", Type: graph.NodeTypeTable, Label: "Table1", Props: map[string]interface{}{"name": "test"}},
				{ID: "node2", Type: graph.NodeTypeColumn, Label: "Column1", Props: map[string]interface{}{"type": "string"}},
			},
			wantValid: true,
			wantErrors: 0,
		},
		{
			name: "missing ID",
			nodes: []Node{
				{ID: "", Type: graph.NodeTypeTable, Label: "Table1"},
			},
			wantValid: false,
			wantErrors: 1,
		},
		{
			name: "missing type",
			nodes: []Node{
				{ID: "node1", Type: "", Label: "Table1"},
			},
			wantValid: false,
			wantErrors: 1,
		},
		{
			name: "mixed valid and invalid",
			nodes: []Node{
				{ID: "node1", Type: graph.NodeTypeTable, Label: "Table1"},
				{ID: "", Type: graph.NodeTypeColumn, Label: "Column1"},
				{ID: "node3", Type: "", Label: "Column2"},
			},
			wantValid: false,
			wantErrors: 2,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ValidateNodes(tt.nodes, logger)
			if result.Valid != tt.wantValid {
				t.Errorf("ValidateNodes() Valid = %v, want %v", result.Valid, tt.wantValid)
			}
			if len(result.Errors) != tt.wantErrors {
				t.Errorf("ValidateNodes() Errors = %d, want %d", len(result.Errors), tt.wantErrors)
			}
		})
	}
}

func TestValidateEdges(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	nodes := []Node{
		{ID: "node1", Type: graph.NodeTypeTable, Label: "Table1"},
		{ID: "node2", Type: graph.NodeTypeColumn, Label: "Column1"},
	}
	
	tests := []struct {
		name     string
		edges    []Edge
		wantValid bool
		wantErrors int
	}{
		{
			name: "valid edges",
			edges: []Edge{
				{SourceID: "node1", TargetID: "node2", Label: "contains"},
			},
			wantValid: true,
			wantErrors: 0,
		},
		{
			name: "missing source ID",
			edges: []Edge{
				{SourceID: "", TargetID: "node2", Label: "contains"},
			},
			wantValid: false,
			wantErrors: 1,
		},
		{
			name: "missing target ID",
			edges: []Edge{
				{SourceID: "node1", TargetID: "", Label: "contains"},
			},
			wantValid: false,
			wantErrors: 1,
		},
		{
			name: "self-loop (warning only)",
			edges: []Edge{
				{SourceID: "node1", TargetID: "node1", Label: "self_ref"},
			},
			wantValid: true,
			wantErrors: 0,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ValidateEdges(tt.edges, nodes, logger)
			if result.Valid != tt.wantValid {
				t.Errorf("ValidateEdges() Valid = %v, want %v", result.Valid, tt.wantValid)
			}
			if len(result.Errors) != tt.wantErrors {
				t.Errorf("ValidateEdges() Errors = %d, want %d", len(result.Errors), tt.wantErrors)
			}
		})
	}
}

func TestFilterValidNodes(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	
	nodes := []Node{
		{ID: "node1", Type: graph.NodeTypeTable, Label: "Table1"},
		{ID: "", Type: graph.NodeTypeColumn, Label: "Column1"}, // Invalid
		{ID: "node3", Type: graph.NodeTypeTable, Label: "Table2"},
		{ID: "node4", Type: "", Label: "Column2"}, // Invalid
	}
	
	result := ValidateNodes(nodes, logger)
	filtered := FilterValidNodes(nodes, result)
	
	if len(filtered) != 2 {
		t.Errorf("FilterValidNodes() returned %d nodes, want 2", len(filtered))
	}
	
	// Check that only valid nodes remain
	for _, node := range filtered {
		if node.ID == "" || node.Type == "" {
			t.Errorf("FilterValidNodes() returned invalid node: %+v", node)
		}
	}
}

