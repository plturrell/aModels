package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds configuration for the telemetry exporter service.
type Config struct {
	// Server settings
	Port string

	// Signavio settings
	SignavioAPIURL     string
	SignavioAPIKey     string
	SignavioTenantID   string
	SignavioDataset    string
	SignavioTimeout    time.Duration
	SignavioMaxRetries int

	// Source service URLs
	AgentMetricsBaseURL string
	ExtractServiceURL   string

	// Agent identification
	AgentName string
}

// LoadConfig loads configuration from environment variables with defaults.
func LoadConfig() *Config {
	cfg := &Config{
		// Server settings
		Port: getEnv("TELEMETRY_EXPORTER_PORT", "8085"),

		// Signavio settings
		SignavioAPIURL:     getEnv("SIGNAVIO_API_URL", "https://ingestion-eu.signavio.com"),
		SignavioAPIKey:     getEnv("SIGNAVIO_API_KEY", ""),
		SignavioTenantID:   getEnv("SIGNAVIO_TENANT_ID", ""),
		SignavioDataset:    getEnv("SIGNAVIO_DATASET", "agent-telemetry"),
		SignavioTimeout:    getEnvDuration("SIGNAVIO_TIMEOUT", 30*time.Second),
		SignavioMaxRetries: getEnvInt("SIGNAVIO_MAX_RETRIES", 3),

		// Source service URLs
		AgentMetricsBaseURL: getEnv("AGENT_METRICS_BASE_URL", ""),
		ExtractServiceURL:   getEnv("EXTRACT_SERVICE_URL", "http://localhost:8081"),

		// Agent identification
		AgentName: getAgentName(),
	}

	return cfg
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	if c.SignavioAPIKey == "" {
		return &ConfigError{Field: "SIGNAVIO_API_KEY", Message: "Signavio API key is required"}
	}
	if c.SignavioDataset == "" {
		return &ConfigError{Field: "SIGNAVIO_DATASET", Message: "Signavio dataset name is required"}
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

// getAgentName gets the agent name from environment or system.
func getAgentName() string {
	// Try environment variable first
	if name := os.Getenv("AGENT_NAME"); name != "" {
		return name
	}

	// Try SERVICE_NAME
	if name := os.Getenv("SERVICE_NAME"); name != "" {
		return name
	}

	// Try hostname
	if hostname, err := os.Hostname(); err == nil && hostname != "" {
		return hostname
	}

	// Default fallback
	return "telemetry-exporter"
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

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

