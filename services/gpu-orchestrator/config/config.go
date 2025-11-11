package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config represents the GPU orchestrator configuration
type Config struct {
	Server           ServerConfig           `yaml:"server"`
	Services         ServicesConfig         `yaml:"services"`
	Scheduler        SchedulerConfig        `yaml:"scheduler"`
	WorkloadDefaults WorkloadDefaultsConfig `yaml:"workload_defaults"`
	Monitoring       MonitoringConfig       `yaml:"monitoring"`
	Auth             AuthConfig             `yaml:"auth"`
}

// ServerConfig contains HTTP server configuration
type ServerConfig struct {
	Port         string        `yaml:"port"`
	ReadTimeout  time.Duration `yaml:"read_timeout"`
	WriteTimeout time.Duration `yaml:"write_timeout"`
	IdleTimeout  time.Duration `yaml:"idle_timeout"`
}

// ServicesConfig contains external service URLs
type ServicesConfig struct {
	DeepAgentsURL   string `yaml:"deepagents_url"`
	GraphServiceURL string `yaml:"graph_service_url"`
}

// SchedulerConfig contains scheduler configuration
type SchedulerConfig struct {
	QueueCheckInterval   time.Duration `yaml:"queue_check_interval"`
	CleanupInterval      time.Duration `yaml:"cleanup_interval"`
	MaxQueueSize         int           `yaml:"max_queue_size"`
	DefaultTTLMultiplier float64       `yaml:"default_ttl_multiplier"`
}

// WorkloadDefaultsConfig contains default settings for different workload types
type WorkloadDefaultsConfig struct {
	Training        WorkloadThresholds `yaml:"training"`
	Inference       WorkloadThresholds `yaml:"inference"`
	Embedding       WorkloadThresholds `yaml:"embedding"`
	OCR             WorkloadThresholds `yaml:"ocr"`
	GraphProcessing WorkloadThresholds `yaml:"graph_processing"`
	Generic         WorkloadThresholds `yaml:"generic"`
}

// WorkloadThresholds contains thresholds for workload analysis
type WorkloadThresholds struct {
	DefaultGPUs        int     `yaml:"default_gpus"`
	DefaultMemoryMB    int64   `yaml:"default_memory_mb"`
	DefaultPriority    int     `yaml:"default_priority"`
	MaxUtilization     float64 `yaml:"max_utilization"`
	LargeBatchSize     int     `yaml:"large_batch_size"`
	XLargeBatchSize    int     `yaml:"xlarge_batch_size"`
	HighConcurrency    int     `yaml:"high_concurrency"`
	VeryHighConcurrency int    `yaml:"very_high_concurrency"`
}

// MonitoringConfig contains monitoring configuration
type MonitoringConfig struct {
	RefreshInterval time.Duration `yaml:"refresh_interval"`
	MetricsEnabled  bool          `yaml:"metrics_enabled"`
}

// AuthConfig contains authentication configuration
type AuthConfig struct {
	Enabled    bool              `yaml:"enabled"`
	APIKeys    map[string]string `yaml:"api_keys"`    // key -> service name
	HeaderName string            `yaml:"header_name"` // HTTP header name for API key
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Port:         "8086",
			ReadTimeout:  15 * time.Second,
			WriteTimeout: 15 * time.Second,
			IdleTimeout:  60 * time.Second,
		},
		Services: ServicesConfig{
			DeepAgentsURL:   "http://localhost:9004",
			GraphServiceURL: "http://localhost:8081",
		},
		Scheduler: SchedulerConfig{
			QueueCheckInterval:   5 * time.Second,
			CleanupInterval:      5 * time.Second,
			MaxQueueSize:         100,
			DefaultTTLMultiplier: 2.0,
		},
		WorkloadDefaults: WorkloadDefaultsConfig{
			Training: WorkloadThresholds{
				DefaultGPUs:         1,
				DefaultMemoryMB:     8192,
				DefaultPriority:     5,
				MaxUtilization:      90.0,
				LargeBatchSize:      64,
				XLargeBatchSize:     128,
				HighConcurrency:     10,
				VeryHighConcurrency: 20,
			},
			Inference: WorkloadThresholds{
				DefaultGPUs:         1,
				DefaultMemoryMB:     4096,
				DefaultPriority:     7,
				MaxUtilization:      80.0,
				LargeBatchSize:      32,
				XLargeBatchSize:     64,
				HighConcurrency:     10,
				VeryHighConcurrency: 20,
			},
			Embedding: WorkloadThresholds{
				DefaultGPUs:         1,
				DefaultMemoryMB:     2048,
				DefaultPriority:     6,
				MaxUtilization:      85.0,
				LargeBatchSize:      32,
				XLargeBatchSize:     64,
				HighConcurrency:     10,
				VeryHighConcurrency: 20,
			},
			OCR: WorkloadThresholds{
				DefaultGPUs:         1,
				DefaultMemoryMB:     4096,
				DefaultPriority:     6,
				MaxUtilization:      80.0,
				LargeBatchSize:      100,
				XLargeBatchSize:     200,
				HighConcurrency:     50,
				VeryHighConcurrency: 100,
			},
			GraphProcessing: WorkloadThresholds{
				DefaultGPUs:         1,
				DefaultMemoryMB:     4096,
				DefaultPriority:     5,
				MaxUtilization:      75.0,
				LargeBatchSize:      1000000,
				XLargeBatchSize:     2000000,
				HighConcurrency:     5,
				VeryHighConcurrency: 10,
			},
			Generic: WorkloadThresholds{
				DefaultGPUs:         1,
				DefaultMemoryMB:     4096,
				DefaultPriority:     5,
				MaxUtilization:      80.0,
				LargeBatchSize:      50,
				XLargeBatchSize:     100,
				HighConcurrency:     10,
				VeryHighConcurrency: 20,
			},
		},
		Monitoring: MonitoringConfig{
			RefreshInterval: 5 * time.Second,
			MetricsEnabled:  true,
		},
		Auth: AuthConfig{
			Enabled:    false,
			APIKeys:    make(map[string]string),
			HeaderName: "X-API-Key",
		},
	}
}

// LoadConfig loads configuration from a YAML file
func LoadConfig(path string) (*Config, error) {
	// Start with defaults
	config := DefaultConfig()

	// If no config file specified, return defaults
	if path == "" {
		return config, nil
	}

	// Read config file
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return config, nil // Use defaults if file doesn't exist
		}
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// Parse YAML
	if err := yaml.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return config, nil
}

// SaveConfig saves configuration to a YAML file
func SaveConfig(config *Config, path string) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}
