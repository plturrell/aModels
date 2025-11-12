package piqa

import (
	"encoding/json"
	"fmt"
	"os"
)

// PIQAConfig holds all configuration for PIQA benchmark
type PIQAConfig struct {
	// Model configuration
	Model ModelConfig `json:"model"`

	// Retrieval configuration
	Retrieval RetrievalConfig `json:"retrieval"`

	// Storage configuration
	Storage StorageConfig `json:"storage"`

	// Performance configuration
	Performance PerformanceConfig `json:"performance"`

	// Logging configuration
	Logging LoggingConfig `json:"logging"`
}

// ModelConfig holds model-specific parameters
type ModelConfig struct {
	InputSize    int     `json:"input_size"`
	HiddenSize   int     `json:"hidden_size"`
	OutputSize   int     `json:"output_size"`
	LearningRate float64 `json:"learning_rate"`
	Dropout      float64 `json:"dropout"`
	ModelType    string  `json:"model_type"` // "phi-mini", "gemma-vault", "default"
}

// RetrievalConfig holds retrieval parameters
type RetrievalConfig struct {
	TopK           int     `json:"top_k"`
	SimilarityType string  `json:"similarity_type"` // "cosine", "dot", "euclidean"
	UseANN         bool    `json:"use_ann"`         // Use Approximate Nearest Neighbor
	ANNTables      int     `json:"ann_tables"`      // Number of LSH tables
	ANNHashes      int     `json:"ann_hashes"`      // Hashes per table
	CacheEnabled   bool    `json:"cache_enabled"`
	CacheTTL       int64   `json:"cache_ttl"` // seconds
	Threshold      float64 `json:"threshold"` // Minimum similarity threshold
}

// StorageConfig holds storage backend parameters
type StorageConfig struct {
	Backend      string `json:"backend"` // "hana", "memory", "file"
	HANAHost     string `json:"hana_host"`
	HANAPort     int    `json:"hana_port"`
	HANAUser     string `json:"hana_user"`
	HANAPassword string `json:"hana_password"`
	HANAEncrypt  bool   `json:"hana_encrypt"`
	FilePath     string `json:"file_path"` // For file-based storage
}

// PerformanceConfig holds performance tuning parameters
type PerformanceConfig struct {
	BatchSize       int  `json:"batch_size"`
	NumWorkers      int  `json:"num_workers"`
	PrefetchSize    int  `json:"prefetch_size"`
	EnableProfiling bool `json:"enable_profiling"`
	MaxMemoryMB     int  `json:"max_memory_mb"`
	EnableMetrics   bool `json:"enable_metrics"`
	MetricsInterval int  `json:"metrics_interval"` // seconds
}

// LoggingConfig holds logging parameters
type LoggingConfig struct {
	Level      string `json:"level"`       // "debug", "info", "warn", "error"
	Format     string `json:"format"`      // "json", "text"
	Output     string `json:"output"`      // "stdout", "file"
	FilePath   string `json:"file_path"`   // Log file path
	MaxSizeMB  int    `json:"max_size_mb"` // Max log file size
	MaxBackups int    `json:"max_backups"` // Max number of old log files
}

// DefaultConfig returns default PIQA configuration
func DefaultConfig() PIQAConfig {
	return PIQAConfig{
		Model: ModelConfig{
			InputSize:    768,
			HiddenSize:   512,
			OutputSize:   256,
			LearningRate: 0.001,
			Dropout:      0.1,
			ModelType:    "default",
		},
		Retrieval: RetrievalConfig{
			TopK:           10,
			SimilarityType: "cosine",
			UseANN:         true,
			ANNTables:      10,
			ANNHashes:      5,
			CacheEnabled:   true,
			CacheTTL:       3600,
			Threshold:      0.5,
		},
		Storage: StorageConfig{
			Backend:     "hana",
			HANAEncrypt: true,
		},
		Performance: PerformanceConfig{
			BatchSize:       32,
			NumWorkers:      4,
			PrefetchSize:    100,
			EnableProfiling: false,
			MaxMemoryMB:     4096,
			EnableMetrics:   true,
			MetricsInterval: 60,
		},
		Logging: LoggingConfig{
			Level:      "info",
			Format:     "json",
			Output:     "stdout",
			MaxSizeMB:  100,
			MaxBackups: 3,
		},
	}
}

// PhiMiniConfig returns optimized config for Phi-Mini-3.5
func PhiMiniConfig() PIQAConfig {
	cfg := DefaultConfig()
	cfg.Model.InputSize = 512
	cfg.Model.HiddenSize = 256
	cfg.Model.OutputSize = 128
	cfg.Model.LearningRate = 0.002
	cfg.Model.ModelType = "phi-mini"
	cfg.Performance.BatchSize = 16
	cfg.Performance.MaxMemoryMB = 2048
	return cfg
}

// GemmaVaultConfig returns optimized config for GemmaVault
func GemmaVaultConfig() PIQAConfig {
	cfg := DefaultConfig()
	cfg.Model.InputSize = 1024
	cfg.Model.HiddenSize = 768
	cfg.Model.OutputSize = 384
	cfg.Model.LearningRate = 0.0005
	cfg.Model.ModelType = "gemma-vault"
	cfg.Performance.BatchSize = 64
	cfg.Performance.MaxMemoryMB = 8192
	return cfg
}

// LoadConfig loads configuration from JSON file
func LoadConfig(path string) (*PIQAConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	var cfg PIQAConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	return &cfg, nil
}

// SaveConfig saves configuration to JSON file
func (c *PIQAConfig) SaveConfig(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write config file: %w", err)
	}

	return nil
}

// Validate validates the configuration
func (c *PIQAConfig) Validate() error {
	if c.Model.InputSize <= 0 {
		return fmt.Errorf("invalid input_size: %d", c.Model.InputSize)
	}
	if c.Model.HiddenSize <= 0 {
		return fmt.Errorf("invalid hidden_size: %d", c.Model.HiddenSize)
	}
	if c.Model.LearningRate <= 0 || c.Model.LearningRate > 1 {
		return fmt.Errorf("invalid learning_rate: %f", c.Model.LearningRate)
	}
	if c.Retrieval.TopK <= 0 {
		return fmt.Errorf("invalid top_k: %d", c.Retrieval.TopK)
	}
	if c.Performance.BatchSize <= 0 {
		return fmt.Errorf("invalid batch_size: %d", c.Performance.BatchSize)
	}
	if c.Performance.NumWorkers < 0 {
		return fmt.Errorf("invalid num_workers: %d", c.Performance.NumWorkers)
	}
	return nil
}

// String returns a string representation of the config
func (c *PIQAConfig) String() string {
	data, _ := json.MarshalIndent(c, "", "  ")
	return string(data)
}
