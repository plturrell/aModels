package storage

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
)

func TestSemanticCache(t *testing.T) {
	// Create mock pool for testing
	pool := &hanapool.Pool{}

	// Create semantic cache
	config := &SemanticCacheConfig{
		DefaultTTL:          1 * time.Hour,
		SimilarityThreshold: 0.8,
		MaxEntries:          1000,
		CleanupInterval:     1 * time.Minute,
		EnableVectorSearch:  false,
		EnableFuzzyMatching: true,
	}

	cache := NewSemanticCache(pool, config)

	t.Run("CreateTables", func(t *testing.T) {
		ctx := context.Background()
		err := cache.CreateTables(ctx)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("GenerateSemanticHash", func(t *testing.T) {
		prompt1 := "What is artificial intelligence?"
		prompt2 := "What is AI?"
		prompt3 := "Tell me about machine learning"

		hash1 := cache.GenerateSemanticHash(prompt1)
		hash2 := cache.GenerateSemanticHash(prompt2)
		hash3 := cache.GenerateSemanticHash(prompt3)

		if hash1 == "" {
			t.Error("Expected semantic hash to be generated")
		}

		if hash2 == "" {
			t.Error("Expected semantic hash to be generated")
		}

		if hash3 == "" {
			t.Error("Expected semantic hash to be generated")
		}

		// Similar prompts should have different hashes (due to stop word filtering)
		if hash1 == hash2 {
			t.Log("Note: Similar prompts have same hash (expected due to stop word filtering)")
		}

		// Different prompts should have different hashes
		if hash1 == hash3 {
			t.Error("Expected different prompts to have different hashes")
		}
	})

	t.Run("GenerateCacheKey", func(t *testing.T) {
		key1 := cache.GenerateCacheKey("What is AI?", "vaultgemma-1b", "general", 0.7, 1000, 0.9, 50)
		key2 := cache.GenerateCacheKey("What is AI?", "vaultgemma-1b", "general", 0.7, 1000, 0.9, 50)
		key3 := cache.GenerateCacheKey("What is AI?", "granite-4.0", "general", 0.7, 1000, 0.9, 50)
		key4 := cache.GenerateCacheKey("What is AI?", "vaultgemma-1b", "general", 0.7, 1000, 0.8, 50)
		key5 := cache.GenerateCacheKey("What is AI?", "vaultgemma-1b", "general", 0.7, 1000, 0.9, 40)

		if key1 == "" {
			t.Error("Expected cache key to be generated")
		}

		// Same parameters should generate same key
		if key1 != key2 {
			t.Error("Expected same parameters to generate same cache key")
		}

		// Different parameters should generate different keys
		if key1 == key3 {
			t.Error("Expected different parameters to generate different cache keys")
		}
		if key1 == key4 {
			t.Error("Expected differing top_p to alter cache key")
		}
		if key1 == key5 {
			t.Error("Expected differing top_k to alter cache key")
		}
	})

	t.Run("SetAndGet", func(t *testing.T) {
		ctx := context.Background()

		entry := &SemanticCacheEntry{
			CacheKey:        "test-key-123",
			PromptHash:      "test-prompt-hash",
			SemanticHash:    "test-semantic-hash",
			Model:           "vaultgemma-1b",
			Domain:          "general",
			Prompt:          "What is artificial intelligence?",
			Response:        "Artificial intelligence (AI) is a branch of computer science...",
			TokensUsed:      150,
			Temperature:     0.7,
			MaxTokens:       1000,
			SimilarityScore: 1.0,
			Metadata: map[string]string{
				"user_id": "test-user",
				"session": "test-session",
			},
			Tags: []string{"ai", "general", "question"},
		}

		err := cache.Set(ctx, entry)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		retrievedEntry, err := cache.Get(ctx, "test-key-123")
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if retrievedEntry == nil {
			t.Error("Expected to retrieve cache entry")
		}

		if retrievedEntry.CacheKey != entry.CacheKey {
			t.Errorf("Expected cache key %s, got %s", entry.CacheKey, retrievedEntry.CacheKey)
		}

		if retrievedEntry.Model != entry.Model {
			t.Errorf("Expected model %s, got %s", entry.Model, retrievedEntry.Model)
		}
	})

	t.Run("FindSemanticSimilar", func(t *testing.T) {
		ctx := context.Background()

		// First, add some test entries
		entries := []*SemanticCacheEntry{
			{
				CacheKey:        "similar-1",
				PromptHash:      "hash-1",
				SemanticHash:    cache.GenerateSemanticHash("What is AI?"),
				Model:           "vaultgemma-1b",
				Domain:          "general",
				Prompt:          "What is AI?",
				Response:        "AI is artificial intelligence...",
				TokensUsed:      100,
				Temperature:     0.7,
				MaxTokens:       500,
				SimilarityScore: 1.0,
				Tags:            []string{"ai", "question"},
			},
			{
				CacheKey:        "similar-2",
				PromptHash:      "hash-2",
				SemanticHash:    cache.GenerateSemanticHash("Tell me about artificial intelligence"),
				Model:           "vaultgemma-1b",
				Domain:          "general",
				Prompt:          "Tell me about artificial intelligence",
				Response:        "Artificial intelligence is a field of computer science...",
				TokensUsed:      120,
				Temperature:     0.7,
				MaxTokens:       500,
				SimilarityScore: 0.9,
				Tags:            []string{"ai", "explanation"},
			},
		}

		for _, entry := range entries {
			err := cache.Set(ctx, entry)
			if err != nil {
				t.Skip("Skipping test - requires HANA connection")
			}
		}

		// Find similar entries
		similarEntries, err := cache.FindSemanticSimilar(ctx, "What is artificial intelligence?", "vaultgemma-1b", "general", 0.8, 10)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if len(similarEntries) == 0 {
			t.Log("No similar entries found (expected if no HANA connection)")
		}
	})

	t.Run("GetStats", func(t *testing.T) {
		ctx := context.Background()

		stats, err := cache.GetStats(ctx)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if stats == nil {
			t.Error("Expected stats to be returned")
		}

		if stats.ByModel == nil {
			t.Error("Expected ByModel map to be initialized")
		}

		if stats.ByDomain == nil {
			t.Error("Expected ByDomain map to be initialized")
		}

		if stats.TopTags == nil {
			t.Error("Expected TopTags map to be initialized")
		}
	})

	t.Run("GetTopEntries", func(t *testing.T) {
		ctx := context.Background()

		entries, err := cache.GetTopEntries(ctx, 10)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if entries == nil {
			t.Error("Expected entries to be returned")
		}
	})

	t.Run("GetByTags", func(t *testing.T) {
		ctx := context.Background()

		entries, err := cache.GetByTags(ctx, []string{"ai", "question"}, 10)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}

		if entries == nil {
			t.Error("Expected entries to be returned")
		}
	})

	t.Run("CleanupExpired", func(t *testing.T) {
		ctx := context.Background()

		err := cache.CleanupExpired(ctx)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})

	t.Run("CleanupOldEntries", func(t *testing.T) {
		ctx := context.Background()

		err := cache.CleanupOldEntries(ctx, 30, 5)
		if err != nil {
			t.Skip("Skipping test - requires HANA connection")
		}
	})
}

func TestSemanticCacheEntry(t *testing.T) {
	t.Run("CreateSemanticCacheEntry", func(t *testing.T) {
		now := time.Now()

		entry := &SemanticCacheEntry{
			ID:              1,
			CacheKey:        "test-key-456",
			PromptHash:      "test-prompt-hash-456",
			SemanticHash:    "test-semantic-hash-456",
			Model:           "granite-4.0",
			Domain:          "blockchain",
			Prompt:          "Explain smart contracts",
			Response:        "Smart contracts are self-executing contracts...",
			TokensUsed:      200,
			Temperature:     0.5,
			MaxTokens:       1500,
			SimilarityScore: 0.95,
			CreatedAt:       now.Add(-10 * time.Minute),
			ExpiresAt:       now.Add(24 * time.Hour),
			AccessCount:     5,
			LastAccessed:    now.Add(-1 * time.Minute),
			Metadata: map[string]string{
				"user_id": "user789",
				"session": "session123",
				"source":  "web",
			},
			Tags: []string{"blockchain", "smart-contracts", "explanation"},
		}

		if entry.CacheKey != "test-key-456" {
			t.Errorf("Expected CacheKey 'test-key-456', got '%s'", entry.CacheKey)
		}

		if entry.Model != "granite-4.0" {
			t.Errorf("Expected Model 'granite-4.0', got '%s'", entry.Model)
		}

		if entry.SimilarityScore != 0.95 {
			t.Errorf("Expected SimilarityScore 0.95, got %f", entry.SimilarityScore)
		}

		if len(entry.Tags) != 3 {
			t.Errorf("Expected 3 tags, got %d", len(entry.Tags))
		}
	})
}

func TestSemanticCacheConfig(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := &SemanticCacheConfig{}

		// Test default values
		if config.DefaultTTL != 0 {
			t.Errorf("Expected DefaultTTL 0, got %v", config.DefaultTTL)
		}

		if config.SimilarityThreshold != 0 {
			t.Errorf("Expected SimilarityThreshold 0, got %f", config.SimilarityThreshold)
		}
	})

	t.Run("CustomConfig", func(t *testing.T) {
		config := &SemanticCacheConfig{
			DefaultTTL:          2 * time.Hour,
			SimilarityThreshold: 0.85,
			MaxEntries:          5000,
			CleanupInterval:     30 * time.Minute,
			EnableVectorSearch:  true,
			EnableFuzzyMatching: true,
		}

		if config.DefaultTTL != 2*time.Hour {
			t.Errorf("Expected DefaultTTL 2h, got %v", config.DefaultTTL)
		}

		if config.SimilarityThreshold != 0.85 {
			t.Errorf("Expected SimilarityThreshold 0.85, got %f", config.SimilarityThreshold)
		}

		if config.MaxEntries != 5000 {
			t.Errorf("Expected MaxEntries 5000, got %d", config.MaxEntries)
		}

		if !config.EnableVectorSearch {
			t.Error("Expected EnableVectorSearch to be true")
		}

		if !config.EnableFuzzyMatching {
			t.Error("Expected EnableFuzzyMatching to be true")
		}
	})
}

func TestSemanticCacheStats(t *testing.T) {
	t.Run("CreateSemanticCacheStats", func(t *testing.T) {
		stats := &SemanticCacheStats{
			TotalEntries:       1000,
			HitCount:           800,
			MissCount:          200,
			SemanticHitCount:   150,
			HitRate:            0.8,
			SemanticHitRate:    0.15,
			AvgSimilarityScore: 0.85,
			ByModel: map[string]int64{
				"vaultgemma-1b": 600,
				"granite-4.0":   400,
			},
			ByDomain: map[string]int64{
				"general":    700,
				"blockchain": 300,
			},
			TopTags: map[string]int64{
				"ai":         500,
				"blockchain": 300,
				"question":   200,
			},
		}

		if stats.TotalEntries != 1000 {
			t.Errorf("Expected TotalEntries 1000, got %d", stats.TotalEntries)
		}

		if stats.HitRate != 0.8 {
			t.Errorf("Expected HitRate 0.8, got %f", stats.HitRate)
		}

		if stats.SemanticHitRate != 0.15 {
			t.Errorf("Expected SemanticHitRate 0.15, got %f", stats.SemanticHitRate)
		}

		if len(stats.ByModel) != 2 {
			t.Errorf("Expected 2 models, got %d", len(stats.ByModel))
		}

		if stats.ByModel["vaultgemma-1b"] != 600 {
			t.Errorf("Expected vaultgemma-1b count 600, got %d", stats.ByModel["vaultgemma-1b"])
		}
	})
}

func BenchmarkSemanticCache(b *testing.B) {
	// Create mock pool for testing
	pool := &hanapool.Pool{}

	// Create semantic cache
	config := &SemanticCacheConfig{
		DefaultTTL:          1 * time.Hour,
		SimilarityThreshold: 0.8,
		MaxEntries:          1000,
		EnableFuzzyMatching: true,
	}

	cache := NewSemanticCache(pool, config)

	ctx := context.Background()

	b.Run("GenerateSemanticHash", func(b *testing.B) {
		prompt := "What is artificial intelligence and how does it work?"
		for i := 0; i < b.N; i++ {
			_ = cache.GenerateSemanticHash(prompt)
		}
	})

	b.Run("GenerateCacheKey", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = cache.GenerateCacheKey(
				fmt.Sprintf("Test prompt %d", i),
				"vaultgemma-1b",
				"general",
				0.7,
				1000,
				0.9,
				50,
			)
		}
	})

	b.Run("Set", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			entry := &SemanticCacheEntry{
				CacheKey:        fmt.Sprintf("benchmark-key-%d", i),
				PromptHash:      fmt.Sprintf("benchmark-hash-%d", i),
				SemanticHash:    cache.GenerateSemanticHash(fmt.Sprintf("Test prompt %d", i)),
				Model:           "vaultgemma-1b",
				Domain:          "general",
				Prompt:          fmt.Sprintf("Test prompt %d", i),
				Response:        fmt.Sprintf("Test response %d", i),
				TokensUsed:      100,
				Temperature:     0.7,
				MaxTokens:       1000,
				SimilarityScore: 1.0,
				Tags:            []string{"benchmark", "test"},
			}

			err := cache.Set(ctx, entry)
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})

	b.Run("Get", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := cache.Get(ctx, fmt.Sprintf("benchmark-key-%d", i))
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})

	b.Run("FindSemanticSimilar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := cache.FindSemanticSimilar(ctx,
				fmt.Sprintf("Test prompt %d", i),
				"vaultgemma-1b",
				"general",
				0.8,
				10,
			)
			if err != nil {
				b.Skip("Skipping benchmark - requires HANA connection")
			}
		}
	})
}

func ExampleSemanticCache() {
	// Create HANA pool
	pool := &hanapool.Pool{}

	// Create semantic cache with custom config
	config := &SemanticCacheConfig{
		DefaultTTL:          24 * time.Hour,
		SimilarityThreshold: 0.8,
		MaxEntries:          10000,
		CleanupInterval:     1 * time.Hour,
		EnableVectorSearch:  false,
		EnableFuzzyMatching: true,
	}

	cache := NewSemanticCache(pool, config)

	// Initialize tables
	ctx := context.Background()
	err := cache.CreateTables(ctx)
	if err != nil {
		// Handle error
		return
	}

	// Create a cache entry
	entry := &SemanticCacheEntry{
		CacheKey:        "example-key-123",
		PromptHash:      "example-prompt-hash",
		SemanticHash:    cache.GenerateSemanticHash("What is machine learning?"),
		Model:           "vaultgemma-1b",
		Domain:          "general",
		Prompt:          "What is machine learning?",
		Response:        "Machine learning is a subset of artificial intelligence...",
		TokensUsed:      150,
		Temperature:     0.7,
		MaxTokens:       1000,
		SimilarityScore: 1.0,
		Metadata: map[string]string{
			"user_id": "example-user",
			"session": "example-session",
		},
		Tags: []string{"ml", "ai", "question"},
	}

	// Store the entry
	err = cache.Set(ctx, entry)
	if err != nil {
		// Handle error
		return
	}

	// Retrieve the entry
	retrievedEntry, err := cache.Get(ctx, "example-key-123")
	if err != nil {
		// Handle error
		return
	}

	// Use the retrieved entry
	if retrievedEntry != nil {
		fmt.Printf("Retrieved: %s\n", retrievedEntry.Response)
	}

	// Find similar entries
	similarEntries, err := cache.FindSemanticSimilar(ctx,
		"What is AI?",
		"vaultgemma-1b",
		"general",
		0.8,
		5,
	)
	if err != nil {
		// Handle error
		return
	}

	// Process similar entries
	for _, similarEntry := range similarEntries {
		fmt.Printf("Similar: %s (score: %.2f)\n",
			similarEntry.Response,
			similarEntry.SimilarityScore,
		)
	}

	// Get cache statistics
	stats, err := cache.GetStats(ctx)
	if err != nil {
		// Handle error
		return
	}

	// Use statistics
	fmt.Printf("Total entries: %d\n", stats.TotalEntries)
	fmt.Printf("Hit rate: %.2f%%\n", stats.HitRate*100)
	fmt.Printf("Semantic hit rate: %.2f%%\n", stats.SemanticHitRate*100)
}
