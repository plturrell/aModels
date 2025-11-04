package domain

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

// PostgresConfigStore stores and retrieves domain configurations from PostgreSQL
type PostgresConfigStore struct {
	db *sql.DB
}

// NewPostgresConfigStore creates a new PostgreSQL-based config store
func NewPostgresConfigStore(dsn string) (*PostgresConfigStore, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("postgres ping failed: %w", err)
	}

	// Create table if it doesn't exist
	if err := createDomainConfigTable(db); err != nil {
		return nil, fmt.Errorf("create domain config table: %w", err)
	}

	return &PostgresConfigStore{db: db}, nil
}

// createDomainConfigTable creates the domain_configs table
func createDomainConfigTable(db *sql.DB) error {
	query := `
	CREATE TABLE IF NOT EXISTS domain_configs (
		id SERIAL PRIMARY KEY,
		domain_name VARCHAR(255) UNIQUE NOT NULL,
		config_json JSONB NOT NULL,
		enabled BOOLEAN DEFAULT true,
		created_at TIMESTAMP DEFAULT NOW(),
		updated_at TIMESTAMP DEFAULT NOW(),
		version INTEGER DEFAULT 1,
		-- Link to training process
		training_run_id VARCHAR(255),
		model_version VARCHAR(255),
		performance_metrics JSONB
	);

	CREATE INDEX IF NOT EXISTS idx_domain_configs_enabled ON domain_configs(enabled);
	CREATE INDEX IF NOT EXISTS idx_domain_configs_training_run ON domain_configs(training_run_id);
	`

	_, err := db.Exec(query)
	return err
}

// GetAllDomainConfigs retrieves all enabled domain configurations
func (p *PostgresConfigStore) GetAllDomainConfigs(ctx context.Context) (map[string]*DomainConfig, error) {
	query := `
		SELECT domain_name, config_json 
		FROM domain_configs 
		WHERE enabled = true
		ORDER BY domain_name
	`

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query domain configs: %w", err)
	}
	defer rows.Close()

	configs := make(map[string]*DomainConfig)
	for rows.Next() {
		var domainName string
		var configJSON string

		if err := rows.Scan(&domainName, &configJSON); err != nil {
			log.Printf("⚠️  Error scanning domain config: %v", err)
			continue
		}

		var cfg DomainConfig
		if err := json.Unmarshal([]byte(configJSON), &cfg); err != nil {
			log.Printf("⚠️  Error unmarshaling domain config for %s: %v", domainName, err)
			continue
		}

		if err := cfg.Validate(); err != nil {
			log.Printf("⚠️  Domain %s invalid: %v", domainName, err)
			continue
		}

		configs[domainName] = &cfg
	}

	return configs, rows.Err()
}

// SaveDomainConfig saves or updates a domain configuration
func (p *PostgresConfigStore) SaveDomainConfig(ctx context.Context, domainName string, config *DomainConfig, trainingRunID, modelVersion string, metrics map[string]interface{}) error {
	configJSON, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	metricsJSON, _ := json.Marshal(metrics)

	query := `
		INSERT INTO domain_configs (domain_name, config_json, training_run_id, model_version, performance_metrics, updated_at)
		VALUES ($1, $2, $3, $4, $5, NOW())
		ON CONFLICT (domain_name) 
		DO UPDATE SET 
			config_json = EXCLUDED.config_json,
			training_run_id = EXCLUDED.training_run_id,
			model_version = EXCLUDED.model_version,
			performance_metrics = EXCLUDED.performance_metrics,
			updated_at = NOW(),
			version = domain_configs.version + 1
	`

	_, err = p.db.ExecContext(ctx, query, domainName, configJSON, trainingRunID, modelVersion, metricsJSON)
	return err
}

// SyncToRedis syncs all enabled domain configs to Redis
func (p *PostgresConfigStore) SyncToRedis(ctx context.Context, redisLoader *RedisConfigLoader) error {
	configs, err := p.GetAllDomainConfigs(ctx)
	if err != nil {
		return fmt.Errorf("get domain configs: %w", err)
	}

	// Determine default domain
	defaultDomain := "general"
	if _, exists := configs["general"]; !exists && len(configs) > 0 {
		for name := range configs {
			defaultDomain = name
			break
		}
	}

	domainsConfig := DomainsConfig{
		Domains:      configs,
		DefaultDomain: defaultDomain,
	}

	configJSON, err := json.Marshal(domainsConfig)
	if err != nil {
		return fmt.Errorf("marshal domains config: %w", err)
	}

	if err := redisLoader.client.Set(ctx, redisLoader.key, configJSON, 0).Err(); err != nil {
		return fmt.Errorf("set redis key: %w", err)
	}

	log.Printf("✅ Synced %d domain configs to Redis", len(configs))
	return nil
}

// GetDomainConfigByTrainingRun retrieves domain configs for a specific training run
func (p *PostgresConfigStore) GetDomainConfigByTrainingRun(ctx context.Context, trainingRunID string) (map[string]*DomainConfig, error) {
	query := `
		SELECT domain_name, config_json 
		FROM domain_configs 
		WHERE training_run_id = $1 AND enabled = true
	`

	rows, err := p.db.QueryContext(ctx, query, trainingRunID)
	if err != nil {
		return nil, fmt.Errorf("query domain configs by training run: %w", err)
	}
	defer rows.Close()

	configs := make(map[string]*DomainConfig)
	for rows.Next() {
		var domainName string
		var configJSON string

		if err := rows.Scan(&domainName, &configJSON); err != nil {
			continue
		}

		var cfg DomainConfig
		if err := json.Unmarshal([]byte(configJSON), &cfg); err != nil {
			continue
		}

		configs[domainName] = &cfg
	}

	return configs, rows.Err()
}

// Close closes the database connection
func (p *PostgresConfigStore) Close() error {
	return p.db.Close()
}

