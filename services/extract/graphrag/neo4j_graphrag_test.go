package graphrag

import (
	"context"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

func setupTestNeo4j(t *testing.T) neo4j.DriverWithContext {
	uri := os.Getenv("NEO4J_URI")
	if uri == "" {
		t.Skip("Skipping Neo4j integration test: NEO4J_URI not set")
		return nil
	}
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(
		os.Getenv("NEO4J_USERNAME"),
		os.Getenv("NEO4J_PASSWORD"),
		""))
	if err != nil {
		t.Skipf("Skipping Neo4j test: %v", err)
		return nil
	}
	return driver
}

func TestBreadthFirstStrategy_BuildQuery(t *testing.T) {
	strategy := NewBreadthFirstStrategy()
	cypher, params := strategy.BuildQuery("test", 3, 10)
	if cypher == "" {
		t.Error("BuildQuery() returned empty Cypher")
	}
	if params["query"] != "test" {
		t.Error("BuildQuery() params missing query")
	}
}

func TestNLToCypherTranslator_ValidateCypher(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	translator := NewNLToCypherTranslator("http://localhost:8080", logger)

	tests := []struct {
		name      string
		cypher    string
		wantError bool
	}{
		{"valid", "MATCH (n) RETURN n", false},
		{"empty", "", true},
		{"invalid", "SELECT * FROM table", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := translator.validateCypher(tt.cypher)
			if tt.wantError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}
