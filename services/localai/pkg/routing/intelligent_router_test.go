package routing

import (
	"context"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
)

func TestIntelligentRouter(t *testing.T) {
	// Create domain manager
	dm := domain.NewDomainManager()
	
	// Load test domains
	err := dm.LoadDomainConfigs("../config/domains.json")
	if err != nil {
		t.Skip("Skipping test - requires domains.json config file")
	}
	
	// Create intelligent router
	router := NewIntelligentRouter(dm)
	
	// Initialize capabilities
	err = router.InitializeCapabilities()
	if err != nil {
		t.Fatalf("Failed to initialize capabilities: %v", err)
	}
	
	t.Run("RouteSimpleQuery", func(t *testing.T) {
		query := "What is a database?"
		userDomains := []string{"0x5678-SQLAgent", "general"}
		
		decision, err := router.RouteQuery(context.Background(), query, userDomains, nil)
		if err != nil {
			t.Fatalf("Failed to route query: %v", err)
		}
		
		if decision.SelectedDomain == "" {
			t.Error("Expected selected domain to be set")
		}
		
		if decision.Confidence < 0.0 || decision.Confidence > 1.0 {
			t.Error("Expected confidence to be between 0.0 and 1.0")
		}
		
		if len(decision.AlternativeRoutes) == 0 {
			t.Error("Expected alternative routes to be provided")
		}
	})
	
	t.Run("RouteComplexQuery", func(t *testing.T) {
		query := "Analyze the performance implications of using recursive CTEs in HANA for large-scale graph traversal operations with millions of nodes and edges, considering memory usage, query optimization, and alternative approaches for distributed systems."
		userDomains := []string{"0x5678-SQLAgent", "0x3579-VectorProcessingAgent", "0xB10C-BlockchainAgent"}
		
		decision, err := router.RouteQuery(context.Background(), query, userDomains, nil)
		if err != nil {
			t.Fatalf("Failed to route complex query: %v", err)
		}
		
		if decision.SelectedDomain == "" {
			t.Error("Expected selected domain to be set")
		}
		
		// Complex queries should have higher confidence for domain-specific routing
		if decision.Confidence < 0.5 {
			t.Errorf("Expected higher confidence for complex query, got: %f", decision.Confidence)
		}
	})
	
	t.Run("RouteDomainSpecificQuery", func(t *testing.T) {
		query := "How do I create a vector index in HANA for similarity search?"
		userDomains := []string{"0x3579-VectorProcessingAgent", "0x5678-SQLAgent"}
		
		decision, err := router.RouteQuery(context.Background(), query, userDomains, nil)
		if err != nil {
			t.Fatalf("Failed to route domain-specific query: %v", err)
		}
		
		// Should prefer VectorProcessingAgent for vector-related queries
		if decision.SelectedDomain != "0x3579-VectorProcessingAgent" {
			t.Logf("Expected VectorProcessingAgent for vector query, got: %s", decision.SelectedDomain)
		}
	})
	
	t.Run("RouteFinancialQuery", func(t *testing.T) {
		query := "Calculate the net present value of a bond with 5% coupon rate and 10-year maturity"
		userDomains := []string{"0x5D1A-SubledgerAgent", "0x71A2-TreasuryAgent", "general"}
		
		decision, err := router.RouteQuery(context.Background(), query, userDomains, nil)
		if err != nil {
			t.Fatalf("Failed to route financial query: %v", err)
		}
		
		if decision.SelectedDomain == "" {
			t.Error("Expected selected domain to be set")
		}
	})
	
	t.Run("RouteWithContext", func(t *testing.T) {
		query := "Process this data"
		contextData := map[string]interface{}{
			"data_type": "financial_transactions",
			"volume":    "large",
			"format":    "csv",
		}
		userDomains := []string{"0xA1B2-DataProcessAgent", "0x5D1A-SubledgerAgent"}
		
		decision, err := router.RouteQuery(context.Background(), query, userDomains, contextData)
		if err != nil {
			t.Fatalf("Failed to route query with context: %v", err)
		}
		
		if decision.SelectedDomain == "" {
			t.Error("Expected selected domain to be set")
		}
	})
}

func TestQueryComplexityAnalysis(t *testing.T) {
	router := &IntelligentRouter{}
	
	t.Run("SimpleQuery", func(t *testing.T) {
		query := "Hello"
		complexity, err := router.analyzeQueryComplexity(query, nil)
		if err != nil {
			t.Fatalf("Failed to analyze simple query: %v", err)
		}
		
		if complexity.Score > 0.5 {
			t.Errorf("Expected low complexity score for simple query, got: %f", complexity.Score)
		}
		
		if complexity.TechnicalLevel != "beginner" {
			t.Errorf("Expected beginner technical level, got: %s", complexity.TechnicalLevel)
		}
	})
	
	t.Run("ComplexQuery", func(t *testing.T) {
		query := "Implement a distributed consensus algorithm using Byzantine fault tolerance with cryptographic proofs and economic incentives for a blockchain network with 1000+ validators"
		complexity, err := router.analyzeQueryComplexity(query, nil)
		if err != nil {
			t.Fatalf("Failed to analyze complex query: %v", err)
		}
		
		if complexity.Score < 0.7 {
			t.Errorf("Expected high complexity score for complex query, got: %f", complexity.Score)
		}
		
		if complexity.TechnicalLevel != "expert" {
			t.Errorf("Expected expert technical level, got: %s", complexity.TechnicalLevel)
		}
		
		if !complexity.RequiresReasoning {
			t.Error("Expected complex query to require reasoning")
		}
	})
	
	t.Run("DomainSpecificQuery", func(t *testing.T) {
		query := "Write a SQL query to join three tables with complex WHERE conditions"
		complexity, err := router.analyzeQueryComplexity(query, nil)
		if err != nil {
			t.Fatalf("Failed to analyze domain-specific query: %v", err)
		}
		
		if !complexity.DomainSpecific {
			t.Error("Expected domain-specific query to be detected")
		}
	})
}

func TestDomainScoring(t *testing.T) {
	dm := domain.NewDomainManager()
	err := dm.LoadDomainConfigs("../config/domains.json")
	if err != nil {
		t.Skip("Skipping test - requires domains.json config file")
	}
	
	router := NewIntelligentRouter(dm)
	err = router.InitializeCapabilities()
	if err != nil {
		t.Fatalf("Failed to initialize capabilities: %v", err)
	}
	
	t.Run("SQLQueryScoring", func(t *testing.T) {
		query := "SELECT * FROM users WHERE age > 25"
		complexity := &QueryComplexity{
			Score:           0.3,
			TokenCount:      20,
			DomainSpecific:  true,
			TechnicalLevel:  "intermediate",
			RequiresReasoning: false,
			ContextLength:   100,
		}
		
		score := router.calculateDomainScore("0x5678-SQLAgent", query, complexity, nil)
		if score < 0.5 {
			t.Errorf("Expected high score for SQL domain with SQL query, got: %f", score)
		}
	})
	
	t.Run("VectorQueryScoring", func(t *testing.T) {
		query := "Find similar vectors using cosine similarity"
		complexity := &QueryComplexity{
			Score:           0.4,
			TokenCount:      15,
			DomainSpecific:  true,
			TechnicalLevel:  "intermediate",
			RequiresReasoning: false,
			ContextLength:   80,
		}
		
		score := router.calculateDomainScore("0x3579-VectorProcessingAgent", query, complexity, nil)
		if score < 0.5 {
			t.Errorf("Expected high score for vector domain with vector query, got: %f", score)
		}
	})
}

func TestModelCapabilityCalculation(t *testing.T) {
	router := &IntelligentRouter{}
	
	t.Run("ReasoningAbility", func(t *testing.T) {
		config := &domain.DomainConfig{
			ModelPath:   "../agenticAiETH_layer4_Models/granite/granite-4.0-transformers-granite",
			Temperature: 0.2,
		}
		
		ability := router.calculateReasoningAbility(config)
		if ability < 0.7 {
			t.Errorf("Expected high reasoning ability for Granite model, got: %f", ability)
		}
	})
	
	t.Run("TechnicalLevel", func(t *testing.T) {
		config := &domain.DomainConfig{
			Name:  "Expert Blockchain Agent",
			Layer: "layer2",
		}
		
		level := router.determineTechnicalLevel(config)
		if level != "expert" {
			t.Errorf("Expected expert technical level, got: %s", level)
		}
	})
	
	t.Run("ModelSpeed", func(t *testing.T) {
		config := &domain.DomainConfig{
			ModelPath:  "../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1",
			MaxTokens:  1000,
		}
		
		speed := router.estimateModelSpeed(config)
		if speed < 50.0 {
			t.Errorf("Expected reasonable speed estimate, got: %f", speed)
		}
	})
}

func TestPerformanceStats(t *testing.T) {
	router := &IntelligentRouter{
		performanceStats: make(map[string]*PerformanceStats),
	}
	
	t.Run("UpdateStats", func(t *testing.T) {
		router.updatePerformanceStats("test-domain")
		
		stats, exists := router.performanceStats["test-domain"]
		if !exists {
			t.Fatal("Expected performance stats to be created")
		}
		
		if stats.TotalQueries != 1 {
			t.Errorf("Expected 1 total query, got: %d", stats.TotalQueries)
		}
		
		if stats.SuccessfulRoutes != 1 {
			t.Errorf("Expected 1 successful route, got: %d", stats.SuccessfulRoutes)
		}
	})
}

func TestAlternativeRoutes(t *testing.T) {
	dm := domain.NewDomainManager()
	err := dm.LoadDomainConfigs("../config/domains.json")
	if err != nil {
		t.Skip("Skipping test - requires domains.json config file")
	}
	
	router := NewIntelligentRouter(dm)
	err = router.InitializeCapabilities()
	if err != nil {
		t.Fatalf("Failed to initialize capabilities: %v", err)
	}
	
	t.Run("GenerateAlternatives", func(t *testing.T) {
		availableDomains := []string{"0x5678-SQLAgent", "0x3579-VectorProcessingAgent", "0xB10C-BlockchainAgent"}
		domainScores := map[string]float64{
			"0x5678-SQLAgent":           0.8,
			"0x3579-VectorProcessingAgent": 0.6,
			"0xB10C-BlockchainAgent":    0.4,
		}
		complexity := &QueryComplexity{
			Score:           0.5,
			TokenCount:      50,
			DomainSpecific:  true,
			TechnicalLevel:  "intermediate",
			RequiresReasoning: false,
			ContextLength:   200,
		}
		
		alternatives := router.generateAlternativeRoutes(availableDomains, domainScores, complexity)
		
		if len(alternatives) == 0 {
			t.Error("Expected alternative routes to be generated")
		}
		
		// Should be sorted by score
		for i := 1; i < len(alternatives); i++ {
			if alternatives[i-1].Score < alternatives[i].Score {
				t.Error("Expected alternatives to be sorted by score")
			}
		}
	})
}

func BenchmarkIntelligentRouter(b *testing.B) {
	dm := domain.NewDomainManager()
	err := dm.LoadDomainConfigs("../config/domains.json")
	if err != nil {
		b.Skip("Skipping benchmark - requires domains.json config file")
	}
	
	router := NewIntelligentRouter(dm)
	err = router.InitializeCapabilities()
	if err != nil {
		b.Fatalf("Failed to initialize capabilities: %v", err)
	}
	
	query := "Analyze the performance implications of using recursive CTEs in HANA for large-scale graph traversal operations"
	userDomains := []string{"0x5678-SQLAgent", "0x3579-VectorProcessingAgent", "0xB10C-BlockchainAgent"}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, err := router.RouteQuery(context.Background(), query, userDomains, nil)
		if err != nil {
			b.Fatalf("Failed to route query: %v", err)
		}
	}
}

func ExampleIntelligentRouter() {
	// Create domain manager
	dm := domain.NewDomainManager()
	
	// Load domain configurations
	err := dm.LoadDomainConfigs("../config/domains.json")
	if err != nil {
		// Handle error
		return
	}
	
	// Create intelligent router
	router := NewIntelligentRouter(dm)
	
	// Initialize model capabilities
	err = router.InitializeCapabilities()
	if err != nil {
		// Handle error
		return
	}
	
	// Route a query
	query := "How do I optimize a SQL query for better performance?"
	userDomains := []string{"0x5678-SQLAgent", "0x3579-VectorProcessingAgent", "general"}
	
	decision, err := router.RouteQuery(context.Background(), query, userDomains, nil)
	if err != nil {
		// Handle error
		return
	}
	
	// Use the routing decision
	if decision.SelectedDomain != "" {
		// Route to the selected domain
		// Process the query with the selected model
	}
	
	// Check alternative routes if needed
	if decision.Confidence < 0.7 {
		// Consider using alternative routes
		for _, alt := range decision.AlternativeRoutes {
			// Evaluate alternative routes
			_ = alt
		}
	}
}
