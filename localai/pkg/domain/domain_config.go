package domain

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
)

// DomainConfig holds configuration for a domain-specific model (one agent per domain)
type DomainConfig struct {
	Name               string              `json:"name"`
	Layer              string              `json:"layer"`        // Layer: layer1, layer2, layer3
	Team               string              `json:"team"`         // Team: DataTeam, FoundationTeam, etc.
	BackendType        string              `json:"backend_type"` // Backend: vaultgemma, openai, ollama, mock
	ModelPath          string              `json:"model_path"`   // For local models
	ModelName          string              `json:"model_name"`   // For API models (e.g., gpt-4, llama2)
	APIKey             string              `json:"api_key"`      // For API backends
	BaseURL            string              `json:"base_url"`     // For custom API endpoints
	AgentID            string              `json:"agent_id"`     // Single agent hex ID (e.g., "0x3579")
	AttentionWeights   map[string]float32  `json:"attention_weights"`
	MaxTokens          int                 `json:"max_tokens"`
	Temperature        float32             `json:"temperature"`
	TopP               float32             `json:"top_p,omitempty"`
	TopK               int                 `json:"top_k,omitempty"`
	DomainTags         []string            `json:"tags"`
	Keywords           []string            `json:"keywords"` // For domain detection
	FallbackModel      string              `json:"fallback_model"`
	EnabledEnvVar      string              `json:"enabled_env_var,omitempty"`
	VisionConfig       *VisionConfig       `json:"vision_config,omitempty"`
	TransformersConfig *TransformersConfig `json:"transformers_config,omitempty"`
}

// VisionConfig captures configuration for vision-backed domains (e.g., DeepSeek OCR).
type VisionConfig struct {
	Endpoint       string `json:"endpoint,omitempty"`        // REST endpoint for OCR service
	APIKey         string `json:"api_key,omitempty"`         // Optional API key
	PythonExec     string `json:"python_exec,omitempty"`     // Path to python executable for local runner
	ScriptPath     string `json:"script_path,omitempty"`     // Path to local runner script
	ModelVariant   string `json:"model_variant,omitempty"`   // Specific model variant
	DefaultPrompt  string `json:"default_prompt,omitempty"`  // Prompt to prefix OCR requests
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"` // Request timeout in seconds
}

// TransformersConfig captures configuration for local HuggingFace transformers backends.
type TransformersConfig struct {
	Endpoint       string `json:"endpoint"`                  // REST endpoint for the Python service
	ModelName      string `json:"model_name"`                // Model identifier understood by the service
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"` // Optional HTTP timeout override
}

// Validate ensures the domain configuration contains required fields and sane values.
func (dc *DomainConfig) Validate() error {
	if dc == nil {
		return fmt.Errorf("domain config is nil")
	}
	if strings.TrimSpace(dc.Name) == "" {
		return fmt.Errorf("domain name cannot be empty")
	}
	backend := strings.ToLower(strings.TrimSpace(dc.BackendType))
	if backend != "deepseek-ocr" && strings.TrimSpace(dc.ModelPath) == "" && strings.TrimSpace(dc.ModelName) == "" {
		return fmt.Errorf("either model_path or model_name must be provided")
	}

	if backend == "deepseek-ocr" {
		if dc.VisionConfig == nil {
			return fmt.Errorf("vision_config must be provided for deepseek-ocr backend")
		}
		if strings.TrimSpace(dc.VisionConfig.Endpoint) == "" && strings.TrimSpace(dc.VisionConfig.ScriptPath) == "" {
			return fmt.Errorf("either endpoint or script_path must be provided for deepseek-ocr backend")
		}
		if dc.VisionConfig.TimeoutSeconds < 0 {
			return fmt.Errorf("vision_config.timeout_seconds must be positive")
		}
	}
	if backend == "hf-transformers" {
		if dc.TransformersConfig == nil {
			return fmt.Errorf("transformers_config must be provided for hf-transformers backend")
		}
		if strings.TrimSpace(dc.TransformersConfig.Endpoint) == "" {
			return fmt.Errorf("transformers_config.endpoint must be provided")
		}
		if strings.TrimSpace(dc.TransformersConfig.ModelName) == "" {
			return fmt.Errorf("transformers_config.model_name must be provided")
		}
		if dc.TransformersConfig.TimeoutSeconds < 0 {
			return fmt.Errorf("transformers_config.timeout_seconds must be positive")
		}
	}
	if dc.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be positive")
	}
	if dc.Temperature < 0 || dc.Temperature > 2.0 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0")
	}
	if dc.TopP < 0 || dc.TopP > 1.0 {
		return fmt.Errorf("top_p must be between 0.0 and 1.0")
	}
	if dc.TopK < 0 {
		return fmt.Errorf("top_k must be non-negative")
	}

	return nil
}

// DomainManager manages multiple domain configurations
type DomainManager struct {
	domains       map[string]*DomainConfig
	defaultDomain string
	mu            sync.RWMutex
}

// LayerTeamMapping represents the hierarchical structure of agents
type LayerTeamMapping map[string]map[string][]string // layer -> team -> agent domains

// DomainsConfig represents the full configuration file structure
type DomainsConfig struct {
	Domains       map[string]*DomainConfig `json:"domains"`
	DefaultDomain string                   `json:"default_domain"`
	LayerMapping  LayerTeamMapping         `json:"layer_mapping"` // Hierarchical: layer -> team -> agents
}

// NewDomainManager creates a new domain manager
func NewDomainManager() *DomainManager {
	return &DomainManager{
		domains:       make(map[string]*DomainConfig),
		defaultDomain: "general",
	}
}

// LoadDomainConfigs loads domain configurations from a JSON file
func (dm *DomainManager) LoadDomainConfigs(configPath string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read config file: %w", err)
	}

	var config DomainsConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	filtered := make(map[string]*DomainConfig)
	for name, cfg := range config.Domains {
		if cfg == nil {
			continue
		}
		if !isDomainEnabled(cfg.EnabledEnvVar) {
			continue
		}
		if err := cfg.Validate(); err != nil {
			return fmt.Errorf("domain %s invalid: %w", name, err)
		}
		filtered[name] = cfg
	}

	if len(filtered) == 0 {
		return fmt.Errorf("no domains enabled after applying configuration toggles")
	}

	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.domains = filtered

	newDefault := dm.defaultDomain
	if config.DefaultDomain != "" {
		newDefault = config.DefaultDomain
	}
	if _, exists := filtered[newDefault]; !exists {
		newDefault = ""
	}
	if newDefault == "" {
		if _, exists := filtered["general"]; exists {
			newDefault = "general"
		} else {
			for name := range filtered {
				newDefault = name
				break
			}
		}
	}
	if newDefault == "" {
		return fmt.Errorf("could not determine default domain after applying configuration toggles")
	}
	dm.defaultDomain = newDefault

	return nil
}

// GetDomainConfig retrieves a domain configuration
func (dm *DomainManager) GetDomainConfig(domain string) (*DomainConfig, bool) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	config, exists := dm.domains[domain]
	return config, exists
}

// ListDomains returns all available domain names
func (dm *DomainManager) ListDomains() []string {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	domains := make([]string, 0, len(dm.domains))
	for name := range dm.domains {
		domains = append(domains, name)
	}
	return domains
}

// ListDomainConfigs returns a copy of all domain configurations indexed by domain name
func (dm *DomainManager) ListDomainConfigs() map[string]*DomainConfig {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	configs := make(map[string]*DomainConfig, len(dm.domains))
	for name, cfg := range dm.domains {
		configs[name] = cfg
	}
	return configs
}

// DetectDomain attempts to detect the appropriate domain from prompt content
func (dm *DomainManager) DetectDomain(prompt string, userDomains []string) string {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	promptLower := strings.ToLower(prompt)

	// Create a map of user's available domains for quick lookup
	userDomainMap := make(map[string]bool)
	for _, d := range userDomains {
		userDomainMap[d] = true
	}

	// Score each domain based on keyword matches
	bestScore := 0
	bestDomain := dm.defaultDomain

	for domainName, config := range dm.domains {
		// Skip if user doesn't have access to this domain
		if len(userDomains) > 0 && !userDomainMap[domainName] {
			continue
		}

		score := 0
		for _, keyword := range config.Keywords {
			if strings.Contains(promptLower, strings.ToLower(keyword)) {
				score++
			}
		}

		if score > bestScore {
			bestScore = score
			bestDomain = domainName
		}
	}

	// If no matches found and user has domains, use first available
	if bestScore == 0 && len(userDomains) > 0 {
		for _, userDomain := range userDomains {
			if _, exists := dm.domains[userDomain]; exists {
				return userDomain
			}
		}
	}

	return bestDomain
}

// GetDefaultDomain returns the default domain name
func (dm *DomainManager) GetDefaultDomain() string {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	return dm.defaultDomain
}

// AddDomain adds or updates a domain configuration
func (dm *DomainManager) AddDomain(name string, config *DomainConfig) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	dm.domains[name] = config
}

// RemoveDomain removes a domain configuration
func (dm *DomainManager) RemoveDomain(name string) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	delete(dm.domains, name)
}

func isDomainEnabled(envVar string) bool {
	if strings.TrimSpace(envVar) == "" {
		return true
	}

	value, exists := os.LookupEnv(envVar)
	if !exists {
		return false
	}

	normalized := strings.TrimSpace(strings.ToLower(value))
	if normalized == "" {
		return false
	}

	switch normalized {
	case "0", "false", "no", "off":
		return false
	}

	return true
}
