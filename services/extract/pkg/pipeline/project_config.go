package pipeline

import (
	"encoding/json"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// ProjectConfig represents the configuration for a code-to-knowledge graph project
type ProjectConfig struct {
	Project ProjectInfo     `yaml:"project" json:"project"`
	Sources SourceConfig   `yaml:"sources" json:"sources"`
	Parsers []ParserConfig `yaml:"parsers" json:"parsers"`
	AI      AIConfig       `yaml:"ai" json:"ai"`
}

// ProjectInfo contains project metadata
type ProjectInfo struct {
	ID       string `yaml:"id" json:"id"`
	Name     string `yaml:"name" json:"name"`
	SystemID string `yaml:"system_id" json:"system_id"`
}

// SourceConfig defines where code comes from
type SourceConfig struct {
	Files          []string         `yaml:"files,omitempty" json:"files,omitempty"`
	GitRepositories []GitRepository `yaml:"git_repositories,omitempty" json:"git_repositories,omitempty"`
}

// GitRepository defines a Git repository source
type GitRepository struct {
	URL         string            `yaml:"url" json:"url"`
	Type        string            `yaml:"type,omitempty" json:"type,omitempty"` // gitea, github, gitlab, generic
	Branch      string            `yaml:"branch,omitempty" json:"branch,omitempty"`
	Tag         string            `yaml:"tag,omitempty" json:"tag,omitempty"`
	Commit      string            `yaml:"commit,omitempty" json:"commit,omitempty"`
	Auth        GitAuth           `yaml:"auth,omitempty" json:"auth,omitempty"`
	FilePatterns []string         `yaml:"file_patterns,omitempty" json:"file_patterns,omitempty"`
}

// GitAuth defines authentication for Git repositories
type GitAuth struct {
	Type     string `yaml:"type" json:"type"` // token, ssh, basic
	Token    string `yaml:"token,omitempty" json:"token,omitempty"`
	KeyPath  string `yaml:"key_path,omitempty" json:"key_path,omitempty"`
	Username string `yaml:"username,omitempty" json:"username,omitempty"`
	Password string `yaml:"password,omitempty" json:"password,omitempty"`
}

// ParserConfig defines parser settings
type ParserConfig struct {
	Type    string                 `yaml:"type" json:"type"` // ddl, sql, json, python, go, etc.
	Enabled bool                   `yaml:"enabled" json:"enabled"`
	Options map[string]interface{} `yaml:"options,omitempty" json:"options,omitempty"`
}

// AIConfig defines AI enhancement settings
type AIConfig struct {
	Enabled   bool     `yaml:"enabled" json:"enabled"`
	Model     string   `yaml:"model,omitempty" json:"model,omitempty"` // cwm, phi-3.5-mini, auto
	LocalAIURL string  `yaml:"localai_url,omitempty" json:"localai_url,omitempty"`
	Tasks     []string `yaml:"tasks,omitempty" json:"tasks,omitempty"` // semantic_analysis, relationship_discovery, etc.
}

// LoadProjectConfig loads a project configuration from a file
func LoadProjectConfig(path string) (*ProjectConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	var config ProjectConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		// Try JSON if YAML fails
		if err := json.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("parse config file: %w", err)
		}
	}

	return &config, nil
}

