package pools

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/apache/arrow-go/v18/arrow/flight"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// FlightClientPool manages a pool of Arrow Flight client connections.
// It provides connection reuse, health checks, and graceful shutdown.
type FlightClientPool struct {
	factory    func() (*flight.Client, error)
	clients    chan *flight.Client
	maxSize    int
	mu         sync.RWMutex
	stats      PoolStats
	closed     int32
	healthCheck func(context.Context, *flight.Client) error
}

// PoolStats tracks connection pool statistics.
type PoolStats struct {
	TotalCreated int64
	TotalReused  int64
	CurrentSize  int64
	TotalClosed  int64
}

// FlightPoolConfig configures the Flight client pool.
type FlightPoolConfig struct {
	MaxSize     int           // Maximum number of connections in pool
	Factory     func() (*flight.Client, error) // Factory function to create new clients
	HealthCheck func(context.Context, *flight.Client) error // Optional health check function
}

// DefaultFlightPoolConfig returns a default pool configuration.
func DefaultFlightPoolConfig() *FlightPoolConfig {
	return &FlightPoolConfig{
		MaxSize: 10,
		Factory: func() (*flight.Client, error) {
			return nil, nil // Must be overridden
		},
	}
}

// NewFlightClientPool creates a new Flight client pool.
func NewFlightClientPool(config *FlightPoolConfig) (*FlightClientPool, error) {
	if config == nil {
		config = DefaultFlightPoolConfig()
	}
	if config.MaxSize <= 0 {
		config.MaxSize = 10
	}
	if config.Factory == nil {
		return nil, nil // Pool disabled if no factory
	}

	pool := &FlightClientPool{
		factory:     config.Factory,
		clients:     make(chan *flight.Client, config.MaxSize),
		maxSize:     config.MaxSize,
		healthCheck: config.HealthCheck,
	}

	return pool, nil
}

// NewFlightClientPoolFromAddr creates a pool with a factory that connects to the given address.
func NewFlightClientPoolFromAddr(addr string, maxSize int) (*FlightClientPool, error) {
	if maxSize <= 0 {
		maxSize = 10
	}

	factory := func() (*flight.Client, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		client, err := flight.NewClientWithMiddlewareCtx(
			ctx,
			addr,
			nil,
			nil,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		return client, err
	}

	return NewFlightClientPool(&FlightPoolConfig{
		MaxSize: maxSize,
		Factory: factory,
	})
}

// Get retrieves a client from the pool or creates a new one if the pool is empty.
func (p *FlightClientPool) Get(ctx context.Context) (*flight.Client, error) {
	if atomic.LoadInt32(&p.closed) == 1 {
		return nil, nil // Pool closed
	}

	select {
	case client := <-p.clients:
		// Check health before returning
		if p.healthCheck != nil {
			if err := p.healthCheck(ctx, client); err != nil {
				// Client unhealthy, close it and create new one
				client.Close()
				atomic.AddInt64(&p.stats.TotalClosed, 1)
				atomic.AddInt64(&p.stats.CurrentSize, -1)
				return p.createNewClient()
			}
		}
		atomic.AddInt64(&p.stats.TotalReused, 1)
		return client, nil
	default:
		// Pool empty, create new client
		return p.createNewClient()
	}
}

// Put returns a client to the pool. If the pool is full, the client is closed.
func (p *FlightClientPool) Put(client *flight.Client) {
	if client == nil || atomic.LoadInt32(&p.closed) == 1 {
		if client != nil {
			client.Close()
			atomic.AddInt64(&p.stats.TotalClosed, 1)
		}
		return
	}

	select {
	case p.clients <- client:
		// Client returned to pool
	default:
		// Pool full, close the client
		client.Close()
		atomic.AddInt64(&p.stats.TotalClosed, 1)
		atomic.AddInt64(&p.stats.CurrentSize, -1)
	}
}

// createNewClient creates a new client and updates statistics.
func (p *FlightClientPool) createNewClient() (*flight.Client, error) {
	client, err := p.factory()
	if err != nil {
		return nil, err
	}
	atomic.AddInt64(&p.stats.TotalCreated, 1)
	atomic.AddInt64(&p.stats.CurrentSize, 1)
	return client, nil
}

// Stats returns the current pool statistics.
func (p *FlightClientPool) Stats() PoolStats {
	return PoolStats{
		TotalCreated: atomic.LoadInt64(&p.stats.TotalCreated),
		TotalReused:  atomic.LoadInt64(&p.stats.TotalReused),
		CurrentSize:  atomic.LoadInt64(&p.stats.CurrentSize),
		TotalClosed:  atomic.LoadInt64(&p.stats.TotalClosed),
	}
}

// Close closes all clients in the pool and marks the pool as closed.
func (p *FlightClientPool) Close() error {
	if !atomic.CompareAndSwapInt32(&p.closed, 0, 1) {
		return nil // Already closed
	}

	close(p.clients)
	for client := range p.clients {
		client.Close()
		atomic.AddInt64(&p.stats.TotalClosed, 1)
		atomic.AddInt64(&p.stats.CurrentSize, -1)
	}

	return nil
}

// Size returns the current number of clients in the pool.
func (p *FlightClientPool) Size() int {
	return len(p.clients)
}

