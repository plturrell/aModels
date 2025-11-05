package workflows

import (
	"context"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/security"
)

// TestBuildCompleteDataProduct tests building a complete data product (thin slice).
func TestBuildCompleteDataProduct(t *testing.T) {
	// Create test registry
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	
	// Create quality monitor (with mock extract service URL)
	qualityMonitor := quality.NewQualityMonitor("http://localhost:9002", nil)
	
	// Create unified workflow integration
	unifiedWorkflow := NewUnifiedWorkflowIntegration(
		"http://localhost:8081",
		"http://localhost:8081",
		"http://localhost:9001",
		"http://localhost:8081",
		registry,
		qualityMonitor,
		nil,
	)
	
	ctx := context.Background()
	
	// Test building a complete data product
	product, err := unifiedWorkflow.BuildCompleteDataProduct(
		ctx,
		"customer_data",
		"I need to analyze customer purchase patterns",
	)
	
	if err != nil {
		t.Fatalf("Failed to build data product: %v", err)
	}
	
	if product == nil {
		t.Fatal("Product is nil")
	}
	
	if product.DataElement == nil {
		t.Fatal("DataElement is nil")
	}
	
	if product.DataElement.Identifier == "" {
		t.Fatal("DataElement identifier is empty")
	}
	
	if product.QualityMetrics == nil {
		t.Fatal("QualityMetrics is nil")
	}
	
	if product.AccessControl == nil {
		t.Fatal("AccessControl is nil")
	}
	
	if product.EnhancedElement.LifecycleState != "published" {
		t.Errorf("Expected lifecycle state 'published', got '%s'", product.EnhancedElement.LifecycleState)
	}
}

// TestAccessControl tests access control functionality.
func TestAccessControl(t *testing.T) {
	ac := security.NewAccessControl("owner", "internal")
	
	// Test granting access
	ac.GrantAccess("user1", "user")
	
	allowed, reason := ac.CheckAccess("user1", "user", "read")
	if !allowed {
		t.Errorf("Expected access granted, got denied: %s", reason)
	}
	
	// Test revoking access
	ac.RevokeAccess("user1", "user")
	
	allowed, _ = ac.CheckAccess("user1", "user", "read")
	if allowed {
		t.Error("Expected access denied after revoke, got granted")
	}
}

// TestQualityMetrics tests quality metrics calculation.
func TestQualityMetrics(t *testing.T) {
	qm := quality.NewQualityMetrics()
	
	// Add SLOs
	qm.AddSLO("freshness", 0.95, "24h")
	qm.AddSLO("completeness", 0.90, "24h")
	
	// Update metrics
	qm.UpdateMetric("freshness", 0.98)
	qm.UpdateMetric("completeness", 0.92)
	
	// Check overall score
	if qm.QualityScore <= 0 {
		t.Error("Quality score should be > 0")
	}
	
	// Check SLOs
	qm.CheckSLOs()
	
	for _, slo := range qm.SLOs {
		if slo.Status == "violated" && slo.Current >= slo.Target {
			t.Errorf("SLO %s should be met, got status: %s", slo.Name, slo.Status)
		}
	}
}

