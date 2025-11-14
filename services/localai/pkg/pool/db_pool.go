package pool

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// DBPool manages database connection pooling
type DBPool struct {
	db    *sql.DB
	stats *PoolStats
	mu    sync.RWMutex
}

// DBPoolConfig holds database pool configuration
type DBPoolConfig struct {
	DSN             string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

// PoolStats tracks pool performance
type PoolStats struct {
	AcquireCount  int64
	ReleaseCount  int64
	TimeoutCount  int64
	AvgWaitTime   time.Duration
	ActiveConns   int64
	IdleConns     int64
	mu            sync.RWMutex
}

// NewDBPool creates a new database connection pool
func NewDBPool(cfg *DBPoolConfig) (*DBPool, error) {
	if cfg == nil {
		return nil, fmt.Errorf("db pool config cannot be nil")
	}

	// Set defaults
	if cfg.MaxOpenConns == 0 {
		cfg.MaxOpenConns = 25
	}
	if cfg.MaxIdleConns == 0 {
		cfg.MaxIdleConns = 5
	}
	if cfg.ConnMaxLifetime == 0 {
		cfg.ConnMaxLifetime = 30 * time.Minute
	}
	if cfg.ConnMaxIdleTime == 0 {
		cfg.ConnMaxIdleTime = 5 * time.Minute
	}

	db, err := sql.Open("postgres", cfg.DSN)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.MaxOpenConns)
	db.SetMaxIdleConns(cfg.MaxIdleConns)
	db.SetConnMaxLifetime(cfg.ConnMaxLifetime)
	db.SetConnMaxIdleTime(cfg.ConnMaxIdleTime)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &DBPool{
		db:    db,
		stats: &PoolStats{},
	}, nil
}

// Acquire gets a connection from the pool
func (p *DBPool) Acquire(ctx context.Context) (*sql.Conn, error) {
	start := time.Now()
	defer func() {
		p.stats.mu.Lock()
		p.stats.AcquireCount++
		p.stats.AvgWaitTime = (p.stats.AvgWaitTime + time.Since(start)) / 2
		p.stats.mu.Unlock()
	}()

	conn, err := p.db.Conn(ctx)
	if err != nil {
		p.stats.mu.Lock()
		p.stats.TimeoutCount++
		p.stats.mu.Unlock()
		return nil, fmt.Errorf("failed to acquire connection: %w", err)
	}

	p.stats.mu.Lock()
	p.stats.ActiveConns++
	p.stats.mu.Unlock()

	return conn, nil
}

// Release returns a connection to the pool
func (p *DBPool) Release(conn *sql.Conn) error {
	p.stats.mu.Lock()
	p.stats.ReleaseCount++
	p.stats.ActiveConns--
	p.stats.mu.Unlock()

	return conn.Close()
}

// Query executes a query with automatic connection management
func (p *DBPool) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	return p.db.QueryContext(ctx, query, args...)
}

// QueryRow executes a query that returns a single row
func (p *DBPool) QueryRow(ctx context.Context, query string, args ...interface{}) *sql.Row {
	return p.db.QueryRowContext(ctx, query, args...)
}

// Exec executes a query without returning rows
func (p *DBPool) Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	return p.db.ExecContext(ctx, query, args...)
}

// Begin starts a new transaction
func (p *DBPool) Begin(ctx context.Context) (*sql.Tx, error) {
	return p.db.BeginTx(ctx, nil)
}

// GetStats returns pool statistics
func (p *DBPool) GetStats() map[string]interface{} {
	dbStats := p.db.Stats()
	
	p.stats.mu.RLock()
	defer p.stats.mu.RUnlock()

	return map[string]interface{}{
		"max_open_connections":  dbStats.MaxOpenConnections,
		"open_connections":      dbStats.OpenConnections,
		"in_use":                dbStats.InUse,
		"idle":                  dbStats.Idle,
		"wait_count":            dbStats.WaitCount,
		"wait_duration_ms":      dbStats.WaitDuration.Milliseconds(),
		"max_idle_closed":       dbStats.MaxIdleClosed,
		"max_lifetime_closed":   dbStats.MaxLifetimeClosed,
		"acquire_count":         p.stats.AcquireCount,
		"release_count":         p.stats.ReleaseCount,
		"timeout_count":         p.stats.TimeoutCount,
		"avg_wait_time_ms":      p.stats.AvgWaitTime.Milliseconds(),
		"active_conns":          p.stats.ActiveConns,
	}
}

// Close closes all connections in the pool
func (p *DBPool) Close() error {
	return p.db.Close()
}

// Health checks if the pool is healthy
func (p *DBPool) Health(ctx context.Context) error {
	return p.db.PingContext(ctx)
}
