package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds all configuration for the Extract service
type Config struct {
	Server      ServerConfig
	Langextract LangextractConfig
	Training    TrainingConfig
	Persistence PersistenceConfig
	Telemetry   TelemetryConfig
	SAPRPT      SAPRPTConfig
	AgentTelemetry AgentTelemetryConfig
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Port      string
	GRPCPort  string
	FlightAddr string
}

// LangextractConfig holds langextract API configuration
type LangextractConfig struct {
	URL    string
	APIKey string
}

// TrainingConfig holds training output configuration
type TrainingConfig struct {
	OutputDir string
}

// PersistenceConfig holds persistence layer configuration
type PersistenceConfig struct {
	SQLitePath                  string
	RedisAddr                   string
	RedisPassword               string
	RedisDB                     int
	Neo4jURI                    string
	Neo4jUsername               string
	Neo4jPassword               string
	DocStorePath                string
	PostgresDSN                 string
	HanaHost                    string
	HanaUser                    string
	HanaPassword                string
	EnableCatalogSchemaIntegration bool
	CatalogResourceBaseURI      string
}

// SAPRPTConfig holds sap-rpt-1-oss configuration
type SAPRPTConfig struct {
	UseEmbeddings    bool
	UseClassification bool
	ZMQPort          int
	EmbeddingModel   string
}

// TelemetryConfig holds telemetry configuration
type TelemetryConfig struct {
	Enabled       bool
	Address       string
	LibraryType   string
	Operation     string
	PrivacyLevel  string
	UserIDHash    string
	DialTimeout   time.Duration
	CallTimeout   time.Duration
}

// AgentTelemetryConfig holds configuration for retrieving agent metrics telemetry data.
type AgentTelemetryConfig struct {
	BaseURL string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	cfg := &Config{
		Server: ServerConfig{
			Port:       getEnv("PORT", defaultPort),
			GRPCPort:   getEnv("GRPC_PORT", defaultGRPCPort),
			FlightAddr: getEnv("FLIGHT_ADDR", defaultFlightAddr),
		},
		Langextract: LangextractConfig{
			URL:    strings.TrimRight(strings.TrimSpace(os.Getenv("LANGEXTRACT_API_URL")), "/"),
			APIKey: os.Getenv("LANGEXTRACT_API_KEY"),
		},
		Training: TrainingConfig{
			OutputDir: getEnv("TRAINING_OUTPUT_DIR", defaultTrainingDir),
		},
		Persistence: PersistenceConfig{
			SQLitePath:                  os.Getenv("SQLITE_PATH"),
			RedisAddr:                   os.Getenv("REDIS_ADDR"),
			RedisPassword:               os.Getenv("REDIS_PASSWORD"),
			RedisDB:                     parseIntEnv("REDIS_DB", 0),
			Neo4jURI:                    os.Getenv("NEO4J_URI"),
			Neo4jUsername:               os.Getenv("NEO4J_USERNAME"),
			Neo4jPassword:               os.Getenv("NEO4J_PASSWORD"),
			DocStorePath:                os.Getenv("DOCUMENT_STORE_PATH"),
			PostgresDSN:                 os.Getenv("POSTGRES_CATALOG_DSN"),
			HanaHost:                    os.Getenv("HANA_HOST"),
			HanaUser:                    os.Getenv("HANA_USER"),
			HanaPassword:                os.Getenv("HANA_PASSWORD"),
			EnableCatalogSchemaIntegration: parseBoolEnv("EXTRACT_ENABLE_CATALOG_SCHEMA", true),
			CatalogResourceBaseURI:      getEnv("EXTRACT_CATALOG_BASE_URI", "http://amodels.org/catalog"),
		},
		Telemetry: TelemetryConfig{
			Enabled:      parseBoolEnv("TELEMETRY_ENABLED", parseBoolEnv("POSTGRES_LANG_SERVICE_ENABLED", false)),
			Address:      strings.TrimSpace(os.Getenv("POSTGRES_LANG_SERVICE_ADDR")),
			LibraryType:  getEnv("POSTGRES_LANG_SERVICE_LIBRARY_TYPE", defaultTelemetryLibrary),
			Operation:    getEnv("POSTGRES_LANG_SERVICE_OPERATION", defaultTelemetryOperation),
			PrivacyLevel: os.Getenv("POSTGRES_LANG_SERVICE_PRIVACY"),
			UserIDHash:   os.Getenv("POSTGRES_LANG_SERVICE_USER_ID"),
			DialTimeout:  defaultDialTimeout,
			CallTimeout:  defaultCallTimeout,
		},
		AgentTelemetry: AgentTelemetryConfig{
			BaseURL: strings.TrimRight(strings.TrimSpace(os.Getenv("AGENT_METRICS_BASE_URL")), "/"),
		},
		SAPRPT: SAPRPTConfig{
			UseEmbeddings:      parseBoolEnv("USE_SAP_RPT_EMBEDDINGS", false),
			UseClassification: parseBoolEnv("USE_SAP_RPT_CLASSIFICATION", false),
			ZMQPort:           parseIntEnv("SAP_RPT_ZMQ_PORT", 5655),
			EmbeddingModel:    getEnv("SAP_RPT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
		},
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return cfg, nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
	var errors []string
	
	// Required: Server port
	if c.Server.Port == "" {
		errors = append(errors, "PORT is required")
	}
	
	// Required: LangExtract configuration (only if URL is set)
	if c.Langextract.URL != "" && c.Langextract.APIKey == "" {
		errors = append(errors, "LANGEXTRACT_API_KEY is required when LANGEXTRACT_API_URL is set")
	}

	// Required: Core persistence
	if strings.TrimSpace(c.Persistence.SQLitePath) == "" {
		errors = append(errors, "SQLITE_PATH is required for table persistence")
	}
	if strings.TrimSpace(c.Persistence.DocStorePath) == "" {
		errors = append(errors, "DOCUMENT_STORE_PATH is required for document persistence")
	}

	// Required: Telemetry configuration when enabled
	if c.Telemetry.Enabled && strings.TrimSpace(c.Telemetry.Address) == "" {
		errors = append(errors, "POSTGRES_LANG_SERVICE_ADDR is required when telemetry is enabled")
	}

	// Required: Neo4j configuration (if Neo4j is being used)
	// Check if any Neo4j fields are set, indicating Neo4j is being used
	usingNeo4j := c.Persistence.Neo4jURI != "" || 
		c.Persistence.Neo4jUsername != "" || 
		c.Persistence.Neo4jPassword != ""
	
	if usingNeo4j {
		if c.Persistence.Neo4jURI == "" {
			errors = append(errors, "NEO4J_URI is required when using Neo4j")
		}
		if c.Persistence.Neo4jUsername == "" {
			errors = append(errors, "NEO4J_USERNAME is required when using Neo4j")
		}
		if c.Persistence.Neo4jPassword == "" {
			errors = append(errors, "NEO4J_PASSWORD is required when using Neo4j")
		}
	}
	
	if len(errors) > 0 {
		return fmt.Errorf("configuration validation failed:\n  %s", strings.Join(errors, "\n  "))
	}
	
	return nil
}

// Constants for default values
const (
	defaultPort               = "8081"
	defaultGRPCPort           = "9090"
	defaultFlightAddr         = ":8815"
	defaultLangextractURL     = "http://langextract-api:5000"
	defaultTrainingDir        = "../../data/training/extracts"
	defaultTelemetryLibrary   = "layer4_extract"
	defaultTelemetryOperation = "run_extract"
	DefaultHTTPClientTimeout   = 45 * time.Second
	defaultDialTimeout        = 5 * time.Second
	defaultCallTimeout        = 3 * time.Second
)

// Helper functions
func getEnv(key, defaultValue string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return defaultValue
}

func parseIntEnv(key string, defaultValue int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return defaultValue
	}
	val, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return val
}

func parseBoolEnv(key string, defaultValue bool) bool {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return defaultValue
	}
	val, err := strconv.ParseBool(value)
	if err != nil {
		return defaultValue
	}
	return val
}

