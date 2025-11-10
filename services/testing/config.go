package testing

import (
	"os"
	"strconv"
	"time"
)

// Config holds configuration for the testing service.
type Config struct {
	// Server settings
	Port string

	// Database settings
	DatabaseDSN string

	// Service URLs
	ExtractServiceURL string
	LocalAIURL        string
	SearchServiceURL  string

	// LocalAI settings
	LocalAIModel        string
	LocalAIEnabled      bool
	LocalAITimeout      time.Duration
	LocalAIRetryAttempts int

	// Default row counts per table type
	DefaultReferenceRowCount int
	DefaultTransactionRowCount int
	DefaultStagingRowCount int

	// Performance settings
	BatchInsertSize      int
	MaxConcurrentGenerators int
	ConnectionPoolSize   int
	ConnectionMaxIdle    int
	ConnectionMaxLifetime time.Duration

	// Timeouts
	DatabaseTimeout      time.Duration
	ExtractServiceTimeout time.Duration
	SearchServiceTimeout time.Duration

	// Feature flags
	EnableLocalAI        bool
	EnableSearch         bool
	EnableFKResolution   bool
	EnableBatchInserts   bool
}

// LoadConfig loads configuration from environment variables with defaults.
func LoadConfig() *Config {
	cfg := &Config{
		// Server settings
		Port: getEnv("TEST_SERVICE_PORT", "8082"),

		// Database settings
		DatabaseDSN: getEnv("TEST_DB_DSN", ""),

		// Service URLs
		ExtractServiceURL: getEnv("EXTRACT_SERVICE_URL", "http://localhost:8081"),
		LocalAIURL:        getEnv("LOCALAI_URL", "http://localhost:8080"),
		SearchServiceURL:  getEnv("SEARCH_SERVICE_URL", ""), // Empty means use Extract service

		// LocalAI settings
		LocalAIModel:         getEnv("LOCALAI_MODEL", "phi-3.5-mini"),
		LocalAIEnabled:       getEnvBool("LOCALAI_ENABLED", true),
		LocalAITimeout:        getEnvDuration("LOCALAI_TIMEOUT", 60*time.Second),
		LocalAIRetryAttempts:  getEnvInt("LOCALAI_RETRY_ATTEMPTS", 3),

		// Default row counts per table type
		DefaultReferenceRowCount:    getEnvInt("DEFAULT_REFERENCE_ROW_COUNT", 50),
		DefaultTransactionRowCount:   getEnvInt("DEFAULT_TRANSACTION_ROW_COUNT", 1000),
		DefaultStagingRowCount:       getEnvInt("DEFAULT_STAGING_ROW_COUNT", 500),

		// Performance settings
		BatchInsertSize:          getEnvInt("BATCH_INSERT_SIZE", 100),
		MaxConcurrentGenerators:  getEnvInt("MAX_CONCURRENT_GENERATORS", 5),
		ConnectionPoolSize:       getEnvInt("DB_CONNECTION_POOL_SIZE", 10),
		ConnectionMaxIdle:        getEnvInt("DB_CONNECTION_MAX_IDLE", 5),
		ConnectionMaxLifetime:     getEnvDuration("DB_CONNECTION_MAX_LIFETIME", 5*time.Minute),

		// Timeouts
		DatabaseTimeout:       getEnvDuration("DB_TIMEOUT", 30*time.Second),
		ExtractServiceTimeout: getEnvDuration("EXTRACT_SERVICE_TIMEOUT", 30*time.Second),
		SearchServiceTimeout:  getEnvDuration("SEARCH_SERVICE_TIMEOUT", 30*time.Second),

		// Feature flags
		EnableLocalAI:      getEnvBool("ENABLE_LOCALAI", true),
		EnableSearch:       getEnvBool("ENABLE_SEARCH", true),
		EnableFKResolution: getEnvBool("ENABLE_FK_RESOLUTION", true),
		EnableBatchInserts:  getEnvBool("ENABLE_BATCH_INSERTS", true),
	}

	// If SearchServiceURL is empty, use ExtractServiceURL
	if cfg.SearchServiceURL == "" {
		cfg.SearchServiceURL = cfg.ExtractServiceURL
	}

	return cfg
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	if c.DatabaseDSN == "" {
		return &ConfigError{Field: "TEST_DB_DSN", Message: "database DSN is required"}
	}
	if c.ExtractServiceURL == "" {
		return &ConfigError{Field: "EXTRACT_SERVICE_URL", Message: "extract service URL is required"}
	}
	return nil
}

// ConfigError represents a configuration error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return "config error: " + e.Field + ": " + e.Message
}

// Helper functions for environment variable parsing

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

