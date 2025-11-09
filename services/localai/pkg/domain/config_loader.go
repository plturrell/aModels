package domain

import (
	"context"
	"fmt"
	"log"
	"os"
)

// ConfigLoader interface for loading domain configurations
type ConfigLoader interface {
	LoadDomainConfigs(ctx context.Context, dm *DomainManager) error
}

// FileConfigLoader loads domain configurations from a JSON file
type FileConfigLoader struct {
	path string
}

// LoadDomainConfigs loads domain configurations from file (existing implementation)
func (f *FileConfigLoader) LoadDomainConfigs(ctx context.Context, dm *DomainManager) error {
	return dm.LoadDomainConfigs(f.path)
}

// NewFileConfigLoader creates a file-based config loader with the specified path.
func NewFileConfigLoader(path string) ConfigLoader {
	if path == "" {
		path = "config/domains.json"
	}
	return &FileConfigLoader{path: path}
}

// PostgresConfigLoader loads domain configurations from PostgreSQL
type PostgresConfigLoader struct {
	store *PostgresConfigStore
}

// LoadDomainConfigs loads domain configurations from Postgres
func (p *PostgresConfigLoader) LoadDomainConfigs(ctx context.Context, dm *DomainManager) error {
	configs, err := p.store.GetAllDomainConfigs(ctx)
	if err != nil {
		return fmt.Errorf("load domain configs from postgres: %w", err)
	}

	if len(configs) == 0 {
		return fmt.Errorf("no domain configs found in postgres")
	}

	// Determine default domain
	defaultDomain := "general"
	if _, exists := configs["general"]; !exists && len(configs) > 0 {
		for name := range configs {
			defaultDomain = name
			break
		}
	}

	// Set domains in DomainManager
	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.domains = configs
	dm.defaultDomain = defaultDomain

	log.Printf("✅ Loaded %d domains from Postgres", len(configs))
	return nil
}

// NewPostgresConfigLoader creates a Postgres-based config loader
func NewPostgresConfigLoader(dsn string) (ConfigLoader, error) {
	store, err := NewPostgresConfigStore(dsn)
	if err != nil {
		return nil, fmt.Errorf("create postgres config store: %w", err)
	}
	return &PostgresConfigLoader{store: store}, nil
}

// NewConfigLoader creates the appropriate config loader based on environment
func NewConfigLoader() (ConfigLoader, error) {
	// Priority: Postgres > Redis > File
	
	// Check for Postgres configuration (highest priority for Phase 1)
	postgresDSN := os.Getenv("POSTGRES_DSN")
	if postgresDSN != "" {
		loader, err := NewPostgresConfigLoader(postgresDSN)
		if err == nil {
			return loader, nil
		}
		log.Printf("⚠️  Failed to create Postgres config loader: %v, falling back", err)
	}

	// Check for Redis configuration (preferred for production)
	redisURL := os.Getenv("REDIS_URL")
	if redisURL != "" {
		redisKey := os.Getenv("REDIS_DOMAIN_CONFIG_KEY")
		if redisKey == "" {
			redisKey = "localai:domains:config"
		}
		return NewRedisConfigLoader(redisURL, redisKey)
	}

	// Fallback to file-based config
	configPath := os.Getenv("DOMAIN_CONFIG_PATH")
	if configPath == "" {
		configPath = "config/domains.json"
	}
	return &FileConfigLoader{path: configPath}, nil
}

// GetConfigSource returns a human-readable description of the config source
func GetConfigSource() string {
	if postgresDSN := os.Getenv("POSTGRES_DSN"); postgresDSN != "" {
		return fmt.Sprintf("Postgres")
	}
	if redisURL := os.Getenv("REDIS_URL"); redisURL != "" {
		return fmt.Sprintf("Redis (%s)", redisURL)
	}
	configPath := os.Getenv("DOMAIN_CONFIG_PATH")
	if configPath == "" {
		configPath = "config/domains.json"
	}
	return fmt.Sprintf("File (%s)", configPath)
}
