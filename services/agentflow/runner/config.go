package runner

import (
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/internal/langflow"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/pkg/catalog"
)

// Config captures the settings required to construct a Runner without exposing
// the internal Langflow client outside this module.
type Config struct {
	// BaseURL points to the Langflow HTTP endpoint.
	BaseURL string
	// APIKey configures the X-API-Key header for Langflow.
	APIKey string
	// AuthToken configures a bearer token for authenticated deployments.
	AuthToken string
	// FlowDir sets the on-disk catalog location used for flow discovery.
	FlowDir string
	// Timeout controls the HTTP client timeout when talking to Langflow.
	Timeout time.Duration
}

const (
	defaultBaseURL = "http://localhost:7860"
	defaultFlowDir = "./flows"
	defaultTimeout = 60 * time.Second
)

// FromEnv initialises a Config from environment variables documented in the
// AgentFlow README. Missing values fall back to sane defaults.
func FromEnv() Config {
	cfg := Config{
		BaseURL: envOr("LANGFLOW_URL", defaultBaseURL),
		FlowDir: envOr("LANGFLOW_FLOWS_DIR", defaultFlowDir),
	}
	if v := strings.TrimSpace(os.Getenv("LANGFLOW_API_KEY")); v != "" {
		cfg.APIKey = v
	}
	if v := strings.TrimSpace(os.Getenv("LANGFLOW_AUTH_TOKEN")); v != "" {
		cfg.AuthToken = v
	}
	if v := strings.TrimSpace(os.Getenv("LANGFLOW_TIMEOUT")); v != "" {
		if parsed, err := time.ParseDuration(v); err == nil {
			cfg.Timeout = parsed
		}
	}
	return cfg
}

// NewFromConfig builds a Runner using the provided configuration.
func NewFromConfig(cfg Config) (*Runner, error) {
	clientTimeout := cfg.Timeout
	if clientTimeout <= 0 {
		clientTimeout = defaultTimeout
	}
	httpClient := &http.Client{
		Timeout: clientTimeout,
	}

	opts := []langflow.Option{
		langflow.WithHTTPClient(httpClient),
	}
	if cfg.APIKey != "" {
		opts = append(opts, langflow.WithAPIKey(cfg.APIKey))
	}
	if cfg.AuthToken != "" {
		opts = append(opts, langflow.WithAuthToken(cfg.AuthToken))
	}

	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	client, err := langflow.NewClient(baseURL, opts...)
	if err != nil {
		return nil, err
	}

	flowDir := cfg.FlowDir
	if strings.TrimSpace(flowDir) == "" {
		flowDir = defaultFlowDir
	}
	loader := catalog.NewLoader(flowDir)

	return New(client, loader), nil
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
