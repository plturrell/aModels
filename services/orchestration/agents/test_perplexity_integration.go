package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

func main() {
	logger := log.New(os.Stdout, "[TEST] ", log.LstdFlags)
	
	// Test API key from command line or environment
	apiKey := os.Getenv("PERPLEXITY_API_KEY")
	if len(os.Args) > 1 {
		apiKey = os.Args[1]
	}
	
	if apiKey == "" {
		logger.Fatal("PERPLEXITY_API_KEY not provided. Usage: go run test_perplexity_integration.go <api_key>")
	}

	logger.Printf("Testing Perplexity integration with API key: %s...", maskAPIKey(apiKey))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Test 1: Connector
	logger.Println("\n=== Test 1: Perplexity Connector ===")
	if err := testConnector(ctx, apiKey, logger); err != nil {
		logger.Printf("❌ Connector test failed: %v", err)
	} else {
		logger.Println("✅ Connector test passed")
	}

	// Test 2: Pipeline (without Deep Research - optional)
	logger.Println("\n=== Test 2: Perplexity Pipeline ===")
	if err := testPipeline(ctx, apiKey, logger); err != nil {
		logger.Printf("❌ Pipeline test failed: %v", err)
	} else {
		logger.Println("✅ Pipeline test passed")
	}

	logger.Println("\n=== All Tests Complete ===")
}

func maskAPIKey(key string) string {
	if len(key) < 8 {
		return "****"
	}
	return key[:4] + "..." + key[len(key)-4:]
}

func testConnector(ctx context.Context, apiKey string, logger *log.Logger) error {
	config := map[string]interface{}{
		"api_key":  apiKey,
		"base_url": "https://api.perplexity.ai",
	}

	connector := connectors.NewPerplexityConnector(config, logger)

	// Test connection
	if err := connector.Connect(ctx, config); err != nil {
		return fmt.Errorf("connection failed: %w", err)
	}
	logger.Println("  ✓ Connected to Perplexity API")

	// Test schema discovery
	schema, err := connector.DiscoverSchema(ctx)
	if err != nil {
		return fmt.Errorf("schema discovery failed: %w", err)
	}
	logger.Printf("  ✓ Discovered schema: %d tables", len(schema.Tables))

	// Test data extraction
	query := map[string]interface{}{
		"query":         "What is machine learning?",
		"model":         "sonar",
		"limit":         2,
		"include_images": false,
	}

	documents, err := connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("data extraction failed: %w", err)
	}

	logger.Printf("  ✓ Extracted %d documents", len(documents))
	
	if len(documents) > 0 {
		doc := documents[0]
		docID, _ := doc["id"].(string)
		title, _ := doc["title"].(string)
		content, _ := doc["content"].(string)
		
		logger.Printf("  ✓ Sample document:")
		logger.Printf("    ID: %s", docID)
		logger.Printf("    Title: %s", title)
		if len(content) > 100 {
			logger.Printf("    Content preview: %s...", content[:100])
		} else {
			logger.Printf("    Content: %s", content)
		}
	}

	connector.Close()
	return nil
}

func testPipeline(ctx context.Context, apiKey string, logger *log.Logger) error {
	// Create pipeline config (minimal for testing)
	config := agents.PerplexityPipelineConfig{
		PerplexityAPIKey:    apiKey,
		PerplexityBaseURL:   "https://api.perplexity.ai",
		DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"), // Optional
		DeepSeekOCRAPIKey:   os.Getenv("DEEPSEEK_OCR_API_KEY"),   // Optional
		DeepResearchURL:     os.Getenv("DEEP_RESEARCH_URL"),      // Optional
		CatalogURL:          os.Getenv("CATALOG_URL"),             // Optional
		TrainingURL:         os.Getenv("TRAINING_URL"),            // Optional
		LocalAIURL:          os.Getenv("LOCALAI_URL"),              // Optional
		SearchURL:           os.Getenv("SEARCH_URL"),              // Optional
		ExtractURL:          os.Getenv("EXTRACT_URL"),             // Optional
		Logger:              logger,
	}

	pipeline, err := agents.NewPerplexityPipeline(config)
	if err != nil {
		return fmt.Errorf("failed to create pipeline: %w", err)
	}
	logger.Println("  ✓ Pipeline created")

	// Test document processing with a simple query
	query := map[string]interface{}{
		"query":         "Explain neural networks in one paragraph",
		"limit":         1,
		"include_images": false,
	}

	logger.Println("  ✓ Processing documents...")
	err = pipeline.ProcessDocuments(ctx, query)
	if err != nil {
		return fmt.Errorf("document processing failed: %w", err)
	}

	logger.Println("  ✓ Documents processed successfully")
	return nil
}

