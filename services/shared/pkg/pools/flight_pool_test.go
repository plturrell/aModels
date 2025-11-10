package pools

import (
	"context"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/flight"
)

func TestFlightClientPool_GetPut(t *testing.T) {
	pool, err := NewFlightClientPool(&FlightPoolConfig{
		MaxSize: 5,
		Factory: func() (*flight.Client, error) {
			// Mock factory - would create real client in integration tests
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("NewFlightClientPool: %v", err)
	}
	defer pool.Close()

	ctx := context.Background()

	// Test getting and putting clients
	client1, err := pool.Get(ctx)
	if err != nil {
		t.Fatalf("Get client: %v", err)
	}

	pool.Put(client1)

	stats := pool.Stats()
	if stats.TotalCreated != 1 {
		t.Errorf("Expected 1 created, got %d", stats.TotalCreated)
	}
}

func TestFlightClientPool_Exhaustion(t *testing.T) {
	pool, err := NewFlightClientPool(&FlightPoolConfig{
		MaxSize: 2,
		Factory: func() (*flight.Client, error) {
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("NewFlightClientPool: %v", err)
	}
	defer pool.Close()

	ctx := context.Background()

	// Exhaust pool
	client1, _ := pool.Get(ctx)
	client2, _ := pool.Get(ctx)

	// Try to get another (should create new)
	client3, err := pool.Get(ctx)
	if err != nil {
		t.Fatalf("Get client: %v", err)
	}

	// Return all
	pool.Put(client1)
	pool.Put(client2)
	pool.Put(client3) // This should be closed since pool is full

	stats := pool.Stats()
	if stats.CurrentSize > 2 {
		t.Errorf("Pool size should not exceed max, got %d", stats.CurrentSize)
	}
}

func TestFlightClientPool_Close(t *testing.T) {
	pool, err := NewFlightClientPool(&FlightPoolConfig{
		MaxSize: 5,
		Factory: func() (*flight.Client, error) {
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("NewFlightClientPool: %v", err)
	}

	ctx := context.Background()
	client, _ := pool.Get(ctx)
	pool.Put(client)

	// Close pool
	if err := pool.Close(); err != nil {
		t.Fatalf("Close pool: %v", err)
	}

	// Should not be able to get clients after close
	_, err = pool.Get(ctx)
	if err == nil {
		t.Error("Expected error getting client from closed pool")
	}
}

func TestFlightClientPool_Stats(t *testing.T) {
	pool, err := NewFlightClientPool(&FlightPoolConfig{
		MaxSize: 10,
		Factory: func() (*flight.Client, error) {
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("NewFlightClientPool: %v", err)
	}
	defer pool.Close()

	ctx := context.Background()

	// Create and reuse some clients
	for i := 0; i < 5; i++ {
		client, _ := pool.Get(ctx)
		pool.Put(client)
	}

	stats := pool.Stats()
	if stats.TotalCreated != 1 {
		t.Errorf("Expected 1 created (reused), got %d", stats.TotalCreated)
	}
	if stats.TotalReused < 4 {
		t.Errorf("Expected at least 4 reuses, got %d", stats.TotalReused)
	}
}

