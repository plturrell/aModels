package testing

import (
	"os"
	"testing"
	"time"
)

func TestLoadConfig(t *testing.T) {
	cfg := LoadConfig()
	
	if cfg == nil {
		t.Fatal("LoadConfig returned nil")
	}
	
	// Test defaults
	if cfg.Port != "8082" {
		t.Errorf("Expected default port 8082, got %s", cfg.Port)
	}
	
	if cfg.DefaultReferenceRowCount != 50 {
		t.Errorf("Expected default reference row count 50, got %d", cfg.DefaultReferenceRowCount)
	}
}

func TestLoadConfig_EnvironmentVariables(t *testing.T) {
	// Set environment variables
	os.Setenv("TEST_SERVICE_PORT", "9090")
	os.Setenv("DEFAULT_REFERENCE_ROW_COUNT", "100")
	defer func() {
		os.Unsetenv("TEST_SERVICE_PORT")
		os.Unsetenv("DEFAULT_REFERENCE_ROW_COUNT")
	}()
	
	cfg := LoadConfig()
	
	if cfg.Port != "9090" {
		t.Errorf("Expected port 9090 from env, got %s", cfg.Port)
	}
	
	if cfg.DefaultReferenceRowCount != 100 {
		t.Errorf("Expected reference row count 100 from env, got %d", cfg.DefaultReferenceRowCount)
	}
}

func TestConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				DatabaseDSN:      "postgres://test",
				ExtractServiceURL: "http://test",
			},
			wantErr: false,
		},
		{
			name: "missing database DSN",
			config: &Config{
				DatabaseDSN:      "",
				ExtractServiceURL: "http://test",
			},
			wantErr: true,
		},
		{
			name: "missing extract URL",
			config: &Config{
				DatabaseDSN:      "postgres://test",
				ExtractServiceURL: "",
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetEnvDuration(t *testing.T) {
	os.Setenv("TEST_DURATION", "30s")
	defer os.Unsetenv("TEST_DURATION")
	
	duration := getEnvDuration("TEST_DURATION", 60*time.Second)
	if duration != 30*time.Second {
		t.Errorf("Expected 30s, got %v", duration)
	}
	
	// Test default
	duration = getEnvDuration("NONEXISTENT", 60*time.Second)
	if duration != 60*time.Second {
		t.Errorf("Expected default 60s, got %v", duration)
	}
}

