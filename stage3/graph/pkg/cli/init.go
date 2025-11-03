package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// InitProject scaffolds a new LangGraph-Go project directory with a sample
// configuration file referencing the CLI demo runtime.
func InitProject(dir string, cfg ProjectConfig, logger Logger) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create project dir: %w", err)
	}

	cfg.Name = strings.TrimSpace(cfg.Name)
	if cfg.Name == "" {
		cfg.Name = "LangGraph Go Project"
	}
	if cfg.Description == "" {
		cfg.Description = "LangGraph-Go demo project"
	}
	if cfg.Checkpoint == "" {
		cfg.Checkpoint = DefaultDevCheckpoint
	}
	if cfg.InitialInput == 0 {
		cfg.InitialInput = 1
	}
	if cfg.Metadata == nil {
		cfg.Metadata = map[string]string{}
	}
	cfg.Metadata["created_at"] = time.Now().Format(time.RFC3339)
	EnsureProjectDefaults(&cfg)

	if err := ValidateProjectConfig(&cfg); err != nil {
		return fmt.Errorf("invalid project config: %w", err)
	}

	// Write JSON config for portability; users can convert to YAML if desired.
	configPath := filepath.Join(dir, "langgraph.project.json")
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal project config: %w", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		return fmt.Errorf("write project config: %w", err)
	}

	// Provide a README stub.
	readmePath := filepath.Join(dir, "README.md")
	readme := fmt.Sprintf(`# %s\n\nThis project was scaffolded using the LangGraph-Go CLI demo.\n\nRun:\n\n\	go run ./cmd/langgraph demo -checkpoint %s -resume\n`, cfg.Name, cfg.Checkpoint)
	if err := os.WriteFile(readmePath, []byte(readme), 0o644); err != nil {
		return fmt.Errorf("write README: %w", err)
	}

	logger.Println("Project initialized at", dir)
	return nil
}
