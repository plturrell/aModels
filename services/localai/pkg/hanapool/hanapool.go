//go:build hana

package hanapool

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/SAP/go-hdb/driver"
)

// ErrNotConfigured is returned when the minimum HANA configuration is not present.
var ErrNotConfigured = errors.New("hana pool not configured")

// Config captures connection parameters for a HANA database.
type Config struct {
	Host            string
	Port            int
	User            string
	Password        string
	Database        string
	Encrypt         bool
	Schema          string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
}

// Configured reports whether the configuration has the required fields.
func (c *Config) Configured() bool {
	if c == nil {
		return false
	}
	return c.Host != "" && c.User != "" && c.Password != ""
}

// Pool wraps a sql.DB connection pool with HANA specific helpers.
type Pool struct {
	db     *sql.DB
	config Config
}

// NewPool creates a connection pool using the supplied configuration.
func NewPool(cfg *Config) (*Pool, error) {
	if cfg == nil {
		return nil, fmt.Errorf("hana pool config is nil")
	}
	if !cfg.Configured() {
		return nil, ErrNotConfigured
	}

	port := cfg.Port
	if port == 0 {
		port = 443
	}

	dsn := fmt.Sprintf("hdb://%s:%s@%s:%d",
		url.QueryEscape(cfg.User),
		url.QueryEscape(cfg.Password),
		cfg.Host,
		port,
	)

	params := url.Values{}
	if cfg.Database != "" {
		params.Set("databaseName", cfg.Database)
	}
	if cfg.Encrypt {
		params.Set("encrypt", "true")
	}
	if cfg.Schema != "" {
		params.Set("currentSchema", cfg.Schema)
	}
	if len(params) > 0 {
		dsn += "?" + params.Encode()
	}

	db, err := sql.Open("hdb", dsn)
	if err != nil {
		return nil, fmt.Errorf("open hana connection: %w", err)
	}

	// Apply optional pool tuning.
	if cfg.MaxOpenConns > 0 {
		db.SetMaxOpenConns(cfg.MaxOpenConns)
	}
	if cfg.MaxIdleConns > 0 {
		db.SetMaxIdleConns(cfg.MaxIdleConns)
	}
	if cfg.ConnMaxLifetime > 0 {
		db.SetConnMaxLifetime(cfg.ConnMaxLifetime)
	}

	// Validate the connection eagerly so we can surface configuration issues early.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping hana: %w", err)
	}

	return &Pool{
		db:     db,
		config: *cfg,
	}, nil
}

// NewPoolFromEnv constructs a pool using common environment variables.
// Returns (nil, nil) when the configuration is incomplete.
func NewPoolFromEnv() (*Pool, error) {
	cfg := ConfigFromEnv()
	if !cfg.Configured() {
		return nil, nil
	}

	pool, err := NewPool(&cfg)
	if errors.Is(err, ErrNotConfigured) {
		return nil, nil
	}
	return pool, err
}

// ConfigFromEnv reads configuration from environment variables.
func ConfigFromEnv() Config {
	port := envInt("HANA_PORT", 0)
	maxOpen := envInt("HANA_MAX_OPEN_CONNS", 0)
	maxIdle := envInt("HANA_MAX_IDLE_CONNS", 0)

	connLifetime := time.Duration(0)
	if v := strings.TrimSpace(os.Getenv("HANA_CONN_MAX_LIFETIME")); v != "" {
		if dur, err := time.ParseDuration(v); err == nil {
			connLifetime = dur
		}
	}

	return Config{
		Host:            strings.TrimSpace(os.Getenv("HANA_HOST")),
		Port:            port,
		User:            strings.TrimSpace(os.Getenv("HANA_USER")),
		Password:        strings.TrimSpace(os.Getenv("HANA_PASSWORD")),
		Database:        strings.TrimSpace(os.Getenv("HANA_DATABASE")),
		Encrypt:         strings.EqualFold(strings.TrimSpace(os.Getenv("HANA_ENCRYPT")), "true"),
		Schema:          strings.TrimSpace(os.Getenv("HANA_SCHEMA")),
		MaxOpenConns:    maxOpen,
		MaxIdleConns:    maxIdle,
		ConnMaxLifetime: connLifetime,
	}
}

// GetDB exposes the raw sql.DB handle (primarily for health checks and tests).
func (p *Pool) GetDB() *sql.DB {
	if p == nil {
		return nil
	}
	return p.db
}

// Close closes the underlying database handle.
func (p *Pool) Close() error {
	if p == nil || p.db == nil {
		return nil
	}
	return p.db.Close()
}

// Execute runs a statement that does not return rows.
func (p *Pool) Execute(ctx context.Context, query string, args ...any) (sql.Result, error) {
	if p == nil || p.db == nil {
		return nil, ErrNotConfigured
	}
	return p.db.ExecContext(ctx, query, args...)
}

// Query runs a query that returns rows.
func (p *Pool) Query(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	if p == nil || p.db == nil {
		return nil, ErrNotConfigured
	}
	return p.db.QueryContext(ctx, query, args...)
}

// QueryRow runs a query expected to return at most one row.
func (p *Pool) QueryRow(ctx context.Context, query string, args ...any) *sql.Row {
	if p == nil || p.db == nil {
		return &sql.Row{}
	}
	return p.db.QueryRowContext(ctx, query, args...)
}

func envInt(key string, fallback int) int {
	val := strings.TrimSpace(os.Getenv(key))
	if val == "" {
		return fallback
	}
	if i, err := strconv.Atoi(val); err == nil {
		return i
	}
	return fallback
}
