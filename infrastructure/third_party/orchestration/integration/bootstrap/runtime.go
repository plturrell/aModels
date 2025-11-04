package bootstrap

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
	localai "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	hanaMemory "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/memory/hana"
	hanaVector "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/vectorstores/hana"
)

// Config captures the bootstrap settings for LocalAI and HANA wiring.
type Config struct {
	LocalAIEndpoint    string
	LocalAIModel       string
	LocalAIDomains     []string
	LocalAIAutoRouting bool
	LocalAITemperature float64
	LocalAIMaxTokens   int
	LocalAITimeout     time.Duration

	EnableHANA  bool
	HANAConfig  *hanapool.Config
	HANASchema  string
	SkipVector  bool
	SkipHistory bool
}

// Runtime aggregates the shared infrastructure needed by orchestration chains.
type Runtime struct {
	LocalAI       *localai.LLM
	HANAPool      *hanapool.Pool
	MemoryManager *hanaMemory.HANAChatMessageHistoryManager
	VectorStore   *hanaVector.HANAVectorStore
	httpClient    *http.Client
	hanaSchema    string
}

// DefaultConfig builds a Config populated from environment variables.
func DefaultConfig() *Config {
	return &Config{
		LocalAIEndpoint:    getEnv("LOCALAI_ENDPOINT", "http://localhost:8080"),
		LocalAIModel:       getEnv("LOCALAI_MODEL", "auto"),
		LocalAIDomains:     splitAndTrim(os.Getenv("LOCALAI_DOMAINS")),
		LocalAIAutoRouting: getEnvBool("LOCALAI_AUTO_ROUTING", true),
		LocalAITemperature: getEnvFloat("LOCALAI_TEMPERATURE", 0.7),
		LocalAIMaxTokens:   getEnvInt("LOCALAI_MAX_TOKENS", 500),
		LocalAITimeout:     time.Duration(getEnvInt("LOCALAI_TIMEOUT_SECONDS", 60)) * time.Second,
		EnableHANA:         getEnvBool("HANA_ENABLE", true),
		HANASchema:         getEnv("HANA_SCHEMA", "AGENTICAI"),
	}
}

// NewRuntime constructs the shared LocalAI client and HANA integrations.
func NewRuntime(cfg *Config) (*Runtime, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	httpClient := &http.Client{
		Timeout: cfg.LocalAITimeout,
	}

	opts := []localai.Option{
		localai.WithHTTPClient(httpClient),
	}

	if cfg.LocalAIEndpoint != "" {
		opts = append(opts, localai.WithBaseURL(cfg.LocalAIEndpoint))
	}
	if cfg.LocalAIModel != "" {
		opts = append(opts, localai.WithModel(cfg.LocalAIModel))
	}
	if cfg.LocalAITemperature > 0 {
		opts = append(opts, localai.WithTemperature(cfg.LocalAITemperature))
	}
	if cfg.LocalAIMaxTokens > 0 {
		opts = append(opts, localai.WithMaxTokens(cfg.LocalAIMaxTokens))
	}
	if len(cfg.LocalAIDomains) > 0 {
		opts = append(opts, localai.WithDomains(cfg.LocalAIDomains))
	}
	opts = append(opts, localai.WithAutoRouting(cfg.LocalAIAutoRouting))

	llm, err := localai.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("bootstrap localai: %w", err)
	}

	rt := &Runtime{
		LocalAI:    llm,
		httpClient: httpClient,
		hanaSchema: cfg.HANASchema,
	}

	if !cfg.EnableHANA {
		return rt, nil
	}

	var pool *hanapool.Pool
	if cfg.HANAConfig != nil {
		pool, err = hanapool.NewPool(cfg.HANAConfig)
	} else {
		pool, err = hanapool.NewPoolFromEnv()
	}
	if err != nil {
		return nil, fmt.Errorf("bootstrap hana pool: %w", err)
	}

	rt.HANAPool = pool
	if rt.hanaSchema == "" && cfg.HANAConfig != nil {
		rt.hanaSchema = cfg.HANAConfig.Schema
	}
	if rt.hanaSchema == "" {
		rt.hanaSchema = "AGENTICAI"
	}

	if !cfg.SkipHistory {
		rt.MemoryManager = hanaMemory.NewHANAChatMessageHistoryManager(pool)
	}

	if !cfg.SkipVector {
		vectorStore, err := hanaVector.NewHANAVectorStore(pool)
		if err != nil {
			return nil, fmt.Errorf("bootstrap hana vector store: %w", err)
		}
		rt.VectorStore = vectorStore
	}

	return rt, nil
}

// Close releases managed resources.
func (r *Runtime) Close() error {
	if r == nil {
		return nil
	}

	if r.HANAPool != nil {
		if err := r.HANAPool.Close(); err != nil {
			return err
		}
	}
	return nil
}

// NewChatHistory provides a convenience helper to create or retrieve a chat history.
func (r *Runtime) NewChatHistory(agentID, sessionID string) (*hanaMemory.HANAChatMessageHistory, error) {
	if r == nil || r.MemoryManager == nil {
		return nil, fmt.Errorf("hana-backed memory disabled")
	}
	return r.MemoryManager.GetOrCreateHistory(agentID, sessionID)
}

// HTTPClient exposes the LocalAI HTTP client for advanced integrations.
func (r *Runtime) HTTPClient() *http.Client {
	if r == nil {
		return nil
	}
	return r.httpClient
}

func getEnv(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		if n, err := strconv.Atoi(val); err == nil {
			return n
		}
	}
	return fallback
}

func getEnvFloat(key string, fallback float64) float64 {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		if n, err := strconv.ParseFloat(val, 64); err == nil {
			return n
		}
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		switch strings.ToLower(val) {
		case "1", "true", "t", "yes", "y", "on":
			return true
		case "0", "false", "f", "no", "n", "off":
			return false
		}
	}
	return fallback
}

func splitAndTrim(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}
