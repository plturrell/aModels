package cache

import (
	"context"
	"testing"
	"time"
)

func TestMultiLevelCache_GetSet(t *testing.T) {
	config := DefaultConfig()
	config.CacheEnabled = true
	config.MemorySize = 100

	cache, err := NewMultiLevelCache(config)
	if err != nil {
		t.Fatalf("NewMultiLevelCache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	key := "test-key"
	value := []byte("test-value")

	// Set value
	if err := cache.Set(ctx, key, value, 5*time.Minute); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// Get value
	got, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != string(value) {
		t.Errorf("Expected %s, got %s", string(value), string(got))
	}
}

func TestMultiLevelCache_Miss(t *testing.T) {
	config := DefaultConfig()
	config.CacheEnabled = true

	cache, err := NewMultiLevelCache(config)
	if err != nil {
		t.Fatalf("NewMultiLevelCache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	got, err := cache.Get(ctx, "non-existent-key")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != nil {
		t.Errorf("Expected nil for cache miss, got %v", got)
	}
}

func TestMultiLevelCache_Delete(t *testing.T) {
	config := DefaultConfig()
	config.CacheEnabled = true

	cache, err := NewMultiLevelCache(config)
	if err != nil {
		t.Fatalf("NewMultiLevelCache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	key := "test-key"
	value := []byte("test-value")

	// Set and verify
	cache.Set(ctx, key, value, 5*time.Minute)
	got, _ := cache.Get(ctx, key)
	if got == nil {
		t.Fatal("Expected value before delete")
	}

	// Delete
	if err := cache.Delete(ctx, key); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// Verify deleted
	got, _ = cache.Get(ctx, key)
	if got != nil {
		t.Error("Expected nil after delete")
	}
}

func TestMultiLevelCache_GetSetJSON(t *testing.T) {
	config := DefaultConfig()
	config.CacheEnabled = true

	cache, err := NewMultiLevelCache(config)
	if err != nil {
		t.Fatalf("NewMultiLevelCache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	key := "test-json-key"
	value := map[string]interface{}{
		"name":  "test",
		"value": 42,
	}

	// Set JSON
	if err := cache.SetJSON(ctx, key, value, 5*time.Minute); err != nil {
		t.Fatalf("SetJSON: %v", err)
	}

	// Get JSON
	var got map[string]interface{}
	if err := cache.GetJSON(ctx, key, &got); err != nil {
		t.Fatalf("GetJSON: %v", err)
	}

	if got["name"] != "test" {
		t.Errorf("Expected 'test', got %v", got["name"])
	}
	if got["value"].(float64) != 42 {
		t.Errorf("Expected 42, got %v", got["value"])
	}
}

func TestMultiLevelCache_Stats(t *testing.T) {
	config := DefaultConfig()
	config.CacheEnabled = true

	cache, err := NewMultiLevelCache(config)
	if err != nil {
		t.Fatalf("NewMultiLevelCache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()

	// Generate some hits and misses
	cache.Set(ctx, "key1", []byte("value1"), 5*time.Minute)
	cache.Get(ctx, "key1") // Hit
	cache.Get(ctx, "key2") // Miss

	stats := cache.Stats()
	if stats.MemoryHits < 1 {
		t.Errorf("Expected at least 1 memory hit, got %d", stats.MemoryHits)
	}
	if stats.MemoryMisses < 1 {
		t.Errorf("Expected at least 1 memory miss, got %d", stats.MemoryMisses)
	}
}

func TestMultiLevelCache_Disabled(t *testing.T) {
	config := DefaultConfig()
	config.CacheEnabled = false

	cache, err := NewMultiLevelCache(config)
	if err != nil {
		t.Fatalf("NewMultiLevelCache: %v", err)
	}

	ctx := context.Background()
	key := "test-key"
	value := []byte("test-value")

	// Set should not error but not cache
	if err := cache.Set(ctx, key, value, 5*time.Minute); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// Get should return nil (cache miss)
	got, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != nil {
		t.Error("Expected nil when cache disabled")
	}
}

