package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Config holds pipeline configuration
type Config struct {
	ProjectConfig *ProjectConfig
	ExtractURL    string
	LocalAIURL    string
	TempDir       string
}

// LoadConfig loads configuration from environment and config file
func LoadConfig(configPath string) (*Config, error) {
	cfg := &Config{
		ExtractURL: getEnvOrDefault("EXTRACT_SERVICE_URL", "http://localhost:8083"),
		LocalAIURL: getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		TempDir:    getEnvOrDefault("TEMP_DIR", "/tmp"),
	}

	// Load project config if path provided
	if configPath != "" {
		projectConfig, err := LoadProjectConfig(configPath)
		if err != nil {
			return nil, fmt.Errorf("load project config: %w", err)
		}
		cfg.ProjectConfig = projectConfig
	}

	return cfg, nil
}

// GetProjectID returns the project ID from config or environment
func (c *Config) GetProjectID() string {
	if c.ProjectConfig != nil && c.ProjectConfig.Project.ID != "" {
		return c.ProjectConfig.Project.ID
	}
	return getEnvOrDefault("PROJECT_ID", "")
}

// GetSystemID returns the system ID from config or environment
func (c *Config) GetSystemID() string {
	if c.ProjectConfig != nil && c.ProjectConfig.Project.SystemID != "" {
		return c.ProjectConfig.Project.SystemID
	}
	return getEnvOrDefault("SYSTEM_ID", "")
}

// IsAIEnabled returns whether AI enhancement is enabled
func (c *Config) IsAIEnabled() bool {
	if c.ProjectConfig != nil {
		return c.ProjectConfig.AI.Enabled
	}
	return strings.ToLower(getEnvOrDefault("AI_ENABLED", "false")) == "true"
}

// GetAIModel returns the AI model to use
func (c *Config) GetAIModel() string {
	if c.ProjectConfig != nil && c.ProjectConfig.AI.Model != "" {
		return c.ProjectConfig.AI.Model
	}
	return getEnvOrDefault("AI_MODEL", "auto")
}

// GetLocalAIURL returns the LocalAI URL
func (c *Config) GetLocalAIURL() string {
	if c.ProjectConfig != nil && c.ProjectConfig.AI.LocalAIURL != "" {
		return c.ProjectConfig.AI.LocalAIURL
	}
	return c.LocalAIURL
}

// EnsureTempDir ensures the temp directory exists
func (c *Config) EnsureTempDir() error {
	return os.MkdirAll(c.TempDir, 0755)
}

// GetTempPath returns a path in the temp directory
func (c *Config) GetTempPath(name string) string {
	return filepath.Join(c.TempDir, name)
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

