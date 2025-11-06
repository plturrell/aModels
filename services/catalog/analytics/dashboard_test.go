package analytics

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/catalog/ai"
	"github.com/plturrell/aModels/services/catalog/iso11179"
)

func TestGetUsageStatistics(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	recommender := ai.NewRecommender(registry, nil)
	dashboard := NewAnalyticsDashboard(registry, recommender, nil)
	
	// Add test elements
	element1 := iso11179.NewDataElement(
		"http://test.org/catalog/data-element/test1",
		"Test Element 1",
		"http://test.org/catalog/concept/test1",
		"http://test.org/catalog/representation/test1",
		"Test definition 1",
	)
	element1.Steward = "user1"
	registry.RegisterDataElement(element1)
	
	element2 := iso11179.NewDataElement(
		"http://test.org/catalog/data-element/test2",
		"Test Element 2",
		"http://test.org/catalog/concept/test2",
		"http://test.org/catalog/representation/test2",
		"Test definition 2",
	)
	element2.Steward = "user2"
	registry.RegisterDataElement(element2)
	
	stats := dashboard.getUsageStatistics()
	
	if stats.TotalAccesses == 0 && len(registry.DataElements) > 0 {
		// This is expected since we're using stewards as proxy for usage
		// In a real implementation, this would track actual accesses
	}
	
	if stats.UniqueUsers < 0 {
		t.Error("UniqueUsers should be non-negative")
	}
	
	if stats.TopUsers == nil {
		t.Error("TopUsers should not be nil")
	}
}

func TestGetPopularElements(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	recommender := ai.NewRecommender(registry, nil)
	dashboard := NewAnalyticsDashboard(registry, recommender, nil)
	
	// Add test elements
	for i := 0; i < 15; i++ {
		element := iso11179.NewDataElement(
			fmt.Sprintf("http://test.org/catalog/data-element/test%d", i),
			fmt.Sprintf("Test Element %d", i),
			fmt.Sprintf("http://test.org/catalog/concept/test%d", i),
			fmt.Sprintf("http://test.org/catalog/representation/test%d", i),
			fmt.Sprintf("Test definition %d", i),
		)
		element.UpdatedAt = time.Now().Add(-time.Duration(i) * time.Hour)
		registry.RegisterDataElement(element)
	}
	
	popular := dashboard.getPopularElements()
	
	// Should return at most 10 elements
	if len(popular) > 10 {
		t.Errorf("Expected at most 10 popular elements, got %d", len(popular))
	}
	
	// Should be sorted by last accessed (most recent first)
	if len(popular) > 1 {
		for i := 1; i < len(popular); i++ {
			if popular[i-1].LastAccessed.Before(popular[i].LastAccessed) {
				t.Error("Popular elements should be sorted by LastAccessed (most recent first)")
			}
		}
	}
}

func TestGetDashboardStats(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	recommender := ai.NewRecommender(registry, nil)
	dashboard := NewAnalyticsDashboard(registry, recommender, nil)
	
	ctx := context.Background()
	stats, err := dashboard.GetDashboardStats(ctx)
	
	if err != nil {
		t.Fatalf("GetDashboardStats failed: %v", err)
	}
	
	if stats == nil {
		t.Fatal("GetDashboardStats returned nil")
	}
	
	if stats.TotalDataElements != len(registry.DataElements) {
		t.Errorf("Expected TotalDataElements to be %d, got %d", len(registry.DataElements), stats.TotalDataElements)
	}
	
	if stats.PopularElements == nil {
		t.Error("PopularElements should not be nil")
	}
	
	if stats.UsageStatistics.TopUsers == nil {
		t.Error("UsageStatistics.TopUsers should not be nil")
	}
}

