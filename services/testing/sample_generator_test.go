package testing

import (
	"context"
	"testing"
)

// MockExtractClient is a mock implementation of ExtractClient for testing.
type MockExtractClient struct {
	QueryKnowledgeGraphFunc func(query string, params map[string]any) ([]map[string]any, error)
	GetKnowledgeGraphFunc   func(projectID, systemID string) (*GraphData, error)
}

func (m *MockExtractClient) QueryKnowledgeGraph(query string, params map[string]any) ([]map[string]any, error) {
	if m.QueryKnowledgeGraphFunc != nil {
		return m.QueryKnowledgeGraphFunc(query, params)
	}
	return []map[string]any{}, nil
}

func (m *MockExtractClient) GetKnowledgeGraph(projectID, systemID string) (*GraphData, error) {
	if m.GetKnowledgeGraphFunc != nil {
		return m.GetKnowledgeGraphFunc(projectID, systemID)
	}
	return &GraphData{}, nil
}

func TestSampleGenerator_GenerateSampleData(t *testing.T) {
	// This is a basic test structure - would need actual database for full test
	// For now, just test that the function structure is correct
	
	mockClient := &MockExtractClient{}
	// Note: Would need actual database connection for full test
	// generator := NewSampleGenerator(nil, mockClient, nil)
	
	_ = mockClient // Suppress unused variable warning
	
	// TODO: Add full test with test database
}

func TestConfig_LoadConfig(t *testing.T) {
	cfg := LoadConfig()
	
	if cfg == nil {
		t.Fatal("LoadConfig returned nil")
	}
	
	// Test that defaults are set
	if cfg.Port == "" {
		t.Error("Port should have a default value")
	}
}

func TestConfig_Validate(t *testing.T) {
	cfg := &Config{
		DatabaseDSN:      "",
		ExtractServiceURL: "",
	}
	
	err := cfg.Validate()
	if err == nil {
		t.Error("Validate should return error for empty DatabaseDSN")
	}
	
	cfg.DatabaseDSN = "postgres://test"
	cfg.ExtractServiceURL = "http://test"
	
	err = cfg.Validate()
	if err != nil {
		t.Errorf("Validate should not return error for valid config: %v", err)
	}
}

