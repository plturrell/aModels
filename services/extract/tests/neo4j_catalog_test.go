package main

import (
	"testing"
)

func TestGenerateResourceURI(t *testing.T) {
	tests := []struct {
		name           string
		baseURI        string
		nodeID         string
		expectedResult string
	}{
		{
			name:           "default base URI",
			baseURI:        "",
			nodeID:         "test-node-123",
			expectedResult: "http://amodels.org/catalog/data-element/test-node-123",
		},
		{
			name:           "custom base URI without trailing slash",
			baseURI:        "http://example.com/catalog",
			nodeID:         "test-node-123",
			expectedResult: "http://example.com/catalog/data-element/test-node-123",
		},
		{
			name:           "custom base URI with trailing slash",
			baseURI:        "http://example.com/catalog/",
			nodeID:         "test-node-123",
			expectedResult: "http://example.com/catalog/data-element/test-node-123",
		},
		{
			name:           "node ID with special characters",
			baseURI:        "http://amodels.org/catalog",
			nodeID:         "table:orders:column:amount",
			expectedResult: "http://amodels.org/catalog/data-element/table:orders:column:amount",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Neo4jPersistence{
				catalogResourceBaseURI: tt.baseURI,
			}
			result := p.generateResourceURI(tt.nodeID)
			if result != tt.expectedResult {
				t.Errorf("generateResourceURI() = %v, want %v", result, tt.expectedResult)
			}
		})
	}
}

func TestShouldCreateResourceNode(t *testing.T) {
	tests := []struct {
		name     string
		nodeType string
		expected bool
	}{
		{
			name:     "column node should create resource",
			nodeType: "column",
			expected: true,
		},
		{
			name:     "table node should create resource",
			nodeType: "table",
			expected: true,
		},
		{
			name:     "root node should not create resource",
			nodeType: "root",
			expected: false,
		},
		{
			name:     "project node should not create resource",
			nodeType: "project",
			expected: false,
		},
		{
			name:     "system node should not create resource",
			nodeType: "system",
			expected: false,
		},
		{
			name:     "information-system node should not create resource",
			nodeType: "information-system",
			expected: false,
		},
		{
			name:     "unknown node type should create resource",
			nodeType: "unknown",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shouldCreateResourceNode(tt.nodeType)
			if result != tt.expected {
				t.Errorf("shouldCreateResourceNode(%q) = %v, want %v", tt.nodeType, result, tt.expected)
			}
		})
	}
}

