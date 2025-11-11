package domain

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeTempConfig(t *testing.T, cfg DomainsConfig) string {
	t.Helper()

	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("failed to marshal config: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "domains.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	return path
}

func TestDomainConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		config  *DomainConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: &DomainConfig{
				Name:        "General",
				ModelPath:   "models/general",
				MaxTokens:   1024,
				Temperature: 0.7,
			},
			wantErr: false,
		},
		{
			name: "missing name",
			config: &DomainConfig{
				ModelPath:   "models/general",
				MaxTokens:   1024,
				Temperature: 0.7,
			},
			wantErr: true,
		},
		{
			name: "missing model path and name",
			config: &DomainConfig{
				Name:        "NoModel",
				MaxTokens:   1024,
				Temperature: 0.7,
			},
			wantErr: true,
		},
		{
			name: "invalid max tokens",
			config: &DomainConfig{
				Name:        "InvalidTokens",
				ModelPath:   "models/invalid",
				MaxTokens:   0,
				Temperature: 0.7,
			},
			wantErr: true,
		},
		{
			name: "invalid temperature",
			config: &DomainConfig{
				Name:        "InvalidTemp",
				ModelPath:   "models/temp",
				MaxTokens:   512,
				Temperature: 3.0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDomainManagerLoadAndList(t *testing.T) {
	cfg := DomainsConfig{
		Domains: map[string]*DomainConfig{
			"general": {
				Name:             "General",
				ModelPath:        "models/general",
				MaxTokens:        1024,
				Temperature:      0.7,
				DomainTags:       []string{"general"},
				Keywords:         []string{"general"},
				FallbackModel:    "",
				AttentionWeights: map[string]float32{"default": 1.0},
			},
			"sql": {
				Name:          "SQL",
				ModelPath:     "models/sql",
				MaxTokens:     2048,
				Temperature:   0.6,
				DomainTags:    []string{"sql"},
				Keywords:      []string{"select", "database"},
				FallbackModel: "general",
			},
		},
		DefaultDomain: "general",
	}

	path := writeTempConfig(t, cfg)

	dm := NewDomainManager()
	if err := dm.LoadDomainConfigs(path); err != nil {
		t.Fatalf("LoadDomainConfigs failed: %v", err)
	}

	if got := dm.GetDefaultDomain(); got != "general" {
		t.Fatalf("expected default domain 'general', got %s", got)
	}

	domains := dm.ListDomains()
	if len(domains) != 2 {
		t.Fatalf("expected 2 domains, got %d", len(domains))
	}

	if cfg, ok := dm.GetDomainConfig("sql"); !ok || cfg.FallbackModel != "general" {
		t.Fatalf("expected sql domain with fallback 'general', got %+v", cfg)
	}

	all := dm.ListDomainConfigs()
	if len(all) != 2 {
		t.Fatalf("expected ListDomainConfigs to return 2 entries, got %d", len(all))
	}
}

func TestDomainManagerDetectDomain(t *testing.T) {
	dm := NewDomainManager()
	dm.AddDomain("general", &DomainConfig{
		Name:        "General",
		ModelPath:   "models/general",
		MaxTokens:   1024,
		Temperature: 0.7,
	})
	dm.AddDomain("sql", &DomainConfig{
		Name:        "SQL",
		ModelPath:   "models/sql",
		MaxTokens:   2048,
		Temperature: 0.8,
		Keywords:    []string{"select", "database"},
	})

	prompt := "How do I SELECT columns from database?"
	if got := dm.DetectDomain(prompt, nil); got != "sql" {
		t.Fatalf("expected sql domain, got %s", got)
	}

	prompt = "Tell me a joke"
	if got := dm.DetectDomain(prompt, nil); got != "general" {
		t.Fatalf("expected general domain, got %s", got)
	}

	// When user has limited domains, ensure detection respects them
	prompt = "Explain SQL joins"
	if got := dm.DetectDomain(prompt, []string{"general"}); got != "general" {
		t.Fatalf("expected general domain due to access restrictions, got %s", got)
	}
}

func TestDomainManagerAddRemove(t *testing.T) {
	dm := NewDomainManager()
	dm.AddDomain("general", &DomainConfig{Name: "General", ModelPath: "models/general", MaxTokens: 512, Temperature: 0.5})

	if _, ok := dm.GetDomainConfig("general"); !ok {
		t.Fatalf("expected general domain to be present")
	}

	dm.RemoveDomain("general")
	if _, ok := dm.GetDomainConfig("general"); ok {
		t.Fatalf("expected general domain to be removed")
	}
}

func TestDomainManagerEnvToggle(t *testing.T) {
	cfg := DomainsConfig{
		Domains: map[string]*DomainConfig{
			"general": {
				Name:        "General",
				ModelPath:   "models/general",
				MaxTokens:   256,
				Temperature: 0.7,
			},
			"optional": {
				Name:          "Optional",
				ModelPath:     "models/optional",
				MaxTokens:     128,
				Temperature:   0.5,
				FallbackModel: "general",
				EnabledEnvVar: "ENABLE_OPTIONAL_DOMAIN",
			},
		},
		DefaultDomain: "general",
	}

	path := writeTempConfig(t, cfg)
	dm := NewDomainManager()

	if err := os.Unsetenv("ENABLE_OPTIONAL_DOMAIN"); err != nil {
		t.Fatalf("failed to unset env: %v", err)
	}

	if err := dm.LoadDomainConfigs(path); err != nil {
		t.Fatalf("LoadDomainConfigs failed: %v", err)
	}

	if _, ok := dm.GetDomainConfig("optional"); ok {
		t.Fatalf("optional domain should be disabled without env toggle")
	}

	t.Setenv("ENABLE_OPTIONAL_DOMAIN", "1")
	if err := dm.LoadDomainConfigs(path); err != nil {
		t.Fatalf("LoadDomainConfigs failed with toggle set: %v", err)
	}
	if _, ok := dm.GetDomainConfig("optional"); !ok {
		t.Fatalf("expected optional domain to be enabled when env toggle present")
	}
}
