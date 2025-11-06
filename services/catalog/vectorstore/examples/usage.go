package examples

// Example usage of HANA Cloud Vector Store integration
// This demonstrates how to integrate with break detection and other systems

/*
import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/catalog/breakdetection"
	"github.com/plturrell/aModels/services/catalog/vectorstore"
)

// Example 1: Store break pattern for Murex
func ExampleStoreMurexBreakPattern() {
	// Initialize HANA Cloud vector store
	config := &vectorstore.HANAConfig{
		ConnectionString: "hdb://user:password@hana-host:30015",
		Schema:           "PUBLIC",
		TableName:        "PUBLIC_VECTORS",
		VectorDimension:  1536,
		EnableIndexing:   true,
	}

	store, err := vectorstore.NewHANACloudVectorStore(
		"hdb://user:password@hana-host:30015",
		config,
		log.Default(),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	// Initialize embedding service
	embeddingService := vectorstore.NewEmbeddingService(
		"http://localhost:8081", // LocalAI URL
		log.Default(),
	)

	// Store Murex break pattern
	breakPatternStore := vectorstore.NewHANABreakPatternStore(store)

	ctx := context.Background()
	pattern := &vectorstore.BreakPattern{
		Description: "Missing journal entries during Murex version migration",
		Frequency:   15,
		Resolution:  "Verify ETL Data Factory pipeline and SAP Fioneer ingestion",
		Prevention:  "Implement pre-migration validation checks",
		Tags:        []string{"murex", "migration", "missing_entry", "finance"},
	}

	// Generate embedding
	embedding, err := embeddingService.GenerateEmbedding(ctx, pattern.Description)
	if err != nil {
		log.Fatal(err)
	}

	// Store pattern
	err = breakPatternStore.StoreBreakPattern(
		ctx,
		breakdetection.SystemSAPFioneer,
		breakdetection.DetectionTypeFinance,
		breakdetection.BreakTypeMissingEntry,
		pattern,
		embedding,
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Break pattern stored successfully")
}

// Example 2: Search for similar breaks across systems
func ExampleSearchSimilarBreaks() {
	store, _ := vectorstore.NewHANACloudVectorStore(
		"hdb://user:password@hana-host:30015",
		vectorstore.DefaultHANAConfig(),
		log.Default(),
	)
	defer store.Close()

	embeddingService := vectorstore.NewEmbeddingService("http://localhost:8081", log.Default())

	ctx := context.Background()

	// Search for similar break patterns
	query := "reconciliation break in finance system during migration"
	queryVector, _ := embeddingService.GenerateEmbedding(ctx, query)

	options := &vectorstore.SearchOptions{
		Type:      "break_pattern",
		System:    "general", // Search across all systems
		Category:  "finance",
		IsPublic:  &[]bool{true}[0],
		Limit:     10,
		Threshold: 0.7,
	}

	results, err := store.SearchPublicInformation(ctx, queryVector, options)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Found %d similar break patterns:\n", len(results))
	for _, result := range results {
		fmt.Printf("- %s (System: %s, Similarity: %.2f)\n", 
			result.Title, result.System, result.Metadata["similarity"])
	}
}

// Example 3: Store regulatory rules
func ExampleStoreRegulatoryRules() {
	store, _ := vectorstore.NewHANACloudVectorStore(
		"hdb://user:password@hana-host:30015",
		vectorstore.DefaultHANAConfig(),
		log.Default(),
	)
	defer store.Close()

	regulatoryStore := vectorstore.NewHANARegulatoryRuleStore(store)
	embeddingService := vectorstore.NewEmbeddingService("http://localhost:8081", log.Default())

	ctx := context.Background()

	// Store Basel III capital requirement
	rule := &vectorstore.RegulatoryRule{
		Regulation:    "Basel III",
		Title:         "Minimum Capital Ratio Requirements",
		Description:   "Banks must maintain minimum Tier 1 capital ratio of 6% and total capital ratio of 8%",
		Requirement:   "Tier 1 capital ratio >= 6%, Total capital ratio >= 8%",
		Compliance:    "Regular monitoring and quarterly reporting",
		EffectiveDate: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		Tags:          []string{"basel", "capital", "regulatory", "compliance"},
	}

	embedding, _ := embeddingService.GenerateEmbedding(ctx, rule.Description)
	err := regulatoryStore.StoreRegulatoryRule(ctx, rule, embedding)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Regulatory rule stored successfully")
}

// Example 4: Store best practice for general use
func ExampleStoreBestPractice() {
	store, _ := vectorstore.NewHANACloudVectorStore(
		"hdb://user:password@hana-host:30015",
		vectorstore.DefaultHANAConfig(),
		log.Default(),
	)
	defer store.Close()

	practiceStore := vectorstore.NewHANABestPracticeStore(store)
	embeddingService := vectorstore.NewEmbeddingService("http://localhost:8081", log.Default())

	ctx := context.Background()

	practice := &vectorstore.BestPractice{
		System:      "general", // Available for all systems
		Category:    "break_detection",
		Title:       "Automated Baseline Comparison",
		Description: "Use automated baseline snapshots before system migrations to enable fast break detection",
		Application: "1. Create baseline before migration\n2. Run break detection after migration\n3. Compare results automatically",
		Benefits:    []string{"Reduces manual work", "Faster detection", "More accurate"},
		Tags:        []string{"baseline", "automation", "migration", "best_practice"},
	}

	embedding, _ := embeddingService.GenerateEmbedding(ctx, practice.Description)
	err := practiceStore.StoreBestPractice(ctx, practice, embedding)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Best practice stored successfully")
}

// Example 5: Integration with break detection service
func ExampleBreakDetectionIntegration() {
	// Initialize services
	store, _ := vectorstore.NewHANACloudVectorStore(
		"hdb://user:password@hana-host:30015",
		vectorstore.DefaultHANAConfig(),
		log.Default(),
	)
	defer store.Close()

	embeddingService := vectorstore.NewEmbeddingService("http://localhost:8081", log.Default())

	// When a break is detected, store it in public knowledge base
	breakRecord := &breakdetection.Break{
		BreakID:       "break-123",
		SystemName:    breakdetection.SystemSAPFioneer,
		DetectionType: breakdetection.DetectionTypeFinance,
		BreakType:     breakdetection.BreakTypeReconciliationBreak,
		Severity:      breakdetection.SeverityCritical,
		AIDescription: "Reconciliation break detected in SAP Fioneer during migration",
		RootCauseAnalysis: "ETL Data Factory pipeline failed to ingest all journal entries",
		Recommendations: []string{
			"Verify ETL pipeline status",
			"Check SAP Fioneer ingestion logs",
			"Re-run failed batch",
		},
		ResolutionNotes: "Re-ran ETL pipeline and verified all entries",
	}

	ctx := context.Background()

	// Generate embedding for break content
	content := vectorstore.BuildBreakContent(breakRecord)
	embedding, _ := embeddingService.GenerateEmbedding(ctx, content)

	// DISABLED FOR SECURITY: Write operations are disabled
	// Store in public knowledge base (anonymized) - NOT AVAILABLE
	// err := vectorstore.StoreBreakForPublicKnowledge(ctx, store, breakRecord, embedding, log.Default())
	// if err != nil {
	//     log.Printf("Note: StoreBreakForPublicKnowledge is disabled for security")
	// }
	
	fmt.Println("Note: Break pattern storage is disabled for security. Service is read-only.")
}

// Example 6: Search knowledge base for Murex
func ExampleSearchMurexKnowledge() {
	store, _ := vectorstore.NewHANACloudVectorStore(
		"hdb://user:password@hana-host:30015",
		vectorstore.DefaultHANAConfig(),
		log.Default(),
	)
	defer store.Close()

	embeddingService := vectorstore.NewEmbeddingService("http://localhost:8081", log.Default())

	ctx := context.Background()

	// Search for Murex-specific knowledge
	query := "How to handle breaks during Murex version migration?"
	queryVector, _ := embeddingService.GenerateEmbedding(ctx, query)

	// Search across all types for Murex
	options := &vectorstore.SearchOptions{
		System:     "murex", // Murex-specific
		IsPublic:   &[]bool{true}[0],
		Limit:      20,
		Threshold:  0.7,
	}

	results, err := store.SearchPublicInformation(ctx, queryVector, options)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Found %d relevant knowledge entries for Murex:\n", len(results))
	for _, result := range results {
		fmt.Printf("- [%s] %s: %s\n", result.Type, result.Title, result.Content[:100])
	}
}
*/

