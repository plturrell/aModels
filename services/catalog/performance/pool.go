package performance

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/catalog/observability"
)

// Logger interface for pool logging
type Logger interface {
	Info(msg string, fields map[string]interface{})
	Debug(msg string, fields map[string]interface{})
}

// ConnectionPool manages Neo4j connection pooling.
type ConnectionPool struct {
	driver      neo4j.DriverWithContext
	maxSize     int
	currentSize int
	mu          sync.RWMutex
	logger      Logger
}

// NewConnectionPool creates a new connection pool.
func NewConnectionPool(
	uri, username, password string,
	maxSize int,
	logger Logger,
) (*ConnectionPool, error) {
	driver, err := neo4j.NewDriverWithContext(
		uri,
		neo4j.BasicAuth(username, password, ""),
		func(config *neo4j.Config) {
			// Configure connection pool
			config.MaxConnectionPoolSize = maxSize
			config.ConnectionAcquisitionTimeout = 30 * time.Second
			config.MaxConnectionLifetime = 1 * time.Hour
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	pool := &ConnectionPool{
		driver:      driver,
		maxSize:     maxSize,
		currentSize: 0,
		logger:      logger,
	}

	// Update metrics
	go pool.updateMetrics()

	return pool, nil
}

// GetSession gets a new session from the pool.
func (p *ConnectionPool) GetSession(ctx context.Context, config neo4j.SessionConfig) neo4j.SessionWithContext {
	session := p.driver.NewSession(ctx, config)
	
	p.mu.Lock()
	p.currentSize++
	p.mu.Unlock()

	if p.logger != nil {
		p.logger.Debug("Session acquired", map[string]interface{}{
			"pool_size": p.currentSize,
		})
	}

	return session
}

// Close closes the connection pool.
func (p *ConnectionPool) Close() error {
	return p.driver.Close(context.Background())
}

// updateMetrics updates Prometheus metrics for connection pool.
func (p *ConnectionPool) updateMetrics() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		p.mu.RLock()
		currentSize := p.currentSize
		p.mu.RUnlock()

		// Update metrics (in production, would get actual pool stats from driver)
		observability.UpdateConnectionPool("active", currentSize)
	}
}

// ExecuteQuery executes a query with automatic connection management.
func (p *ConnectionPool) ExecuteQuery(
	ctx context.Context,
	query string,
	params map[string]interface{},
	queryType string,
) ([]map[string]interface{}, error) {
	session := p.GetSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	start := time.Now()
	result, err := session.Run(ctx, query, params)
	if err != nil {
		observability.RecordNeo4jQuery(queryType, "error", time.Since(start))
		return nil, fmt.Errorf("query execution failed: %w", err)
	}

	var records []map[string]interface{}
	for result.Next(ctx) {
		record := result.Record()
		recordMap := make(map[string]interface{})
		for _, key := range record.Keys {
			value, _ := record.Get(key)
			recordMap[key] = value
		}
		records = append(records, recordMap)
	}

	if err := result.Err(); err != nil {
		observability.RecordNeo4jQuery(queryType, "error", time.Since(start))
		return nil, fmt.Errorf("result processing failed: %w", err)
	}

	observability.RecordNeo4jQuery(queryType, "success", time.Since(start))
	return records, nil
}

