package domain

import (
	"context"
	"fmt"
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

// NewConfigLoader creates the appropriate config loader based on environment
func NewConfigLoader() (ConfigLoader, error) {
	// Check for Redis configuration first (preferred for production)
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
	if redisURL := os.Getenv("REDIS_URL"); redisURL != "" {
		return fmt.Sprintf("Redis (%s)", redisURL)
	}
	configPath := os.Getenv("DOMAIN_CONFIG_PATH")
	if configPath == "" {
		configPath = "config/domains.json"
	}
	return fmt.Sprintf("File (%s)", configPath)
}
