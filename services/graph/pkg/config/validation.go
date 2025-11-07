package config

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

// ValidationError represents a configuration validation error.
type ValidationError struct {
	Variable string
	Message  string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("configuration error: %s: %s", e.Variable, e.Message)
}

// Validator validates environment variables.
type Validator struct {
	errors []*ValidationError
}

// NewValidator creates a new validator.
func NewValidator() *Validator {
	return &Validator{
		errors: make([]*ValidationError, 0),
	}
}

// Require checks that a required environment variable is set.
func (v *Validator) Require(key string) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		v.errors = append(v.errors, &ValidationError{
			Variable: key,
			Message:  "required but not set",
		})
	}
}

// RequireOneOf checks that at least one of the given environment variables is set.
func (v *Validator) RequireOneOf(keys ...string) {
	found := false
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			found = true
			break
		}
	}
	if !found {
		v.errors = append(v.errors, &ValidationError{
			Variable: strings.Join(keys, " or "),
			Message:  "at least one must be set",
		})
	}
}

// RequireURL checks that a required environment variable is set and is a valid URL.
func (v *Validator) RequireURL(key string) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		v.errors = append(v.errors, &ValidationError{
			Variable: key,
			Message:  "required but not set",
		})
		return
	}
	if _, err := url.Parse(value); err != nil {
		v.errors = append(v.errors, &ValidationError{
			Variable: key,
			Message:  fmt.Sprintf("invalid URL: %v", err),
		})
	}
}

// OptionalURL checks that if an environment variable is set, it's a valid URL.
func (v *Validator) OptionalURL(key string) {
	value := strings.TrimSpace(os.Getenv(key))
	if value != "" {
		if _, err := url.Parse(value); err != nil {
			v.errors = append(v.errors, &ValidationError{
				Variable: key,
				Message:  fmt.Sprintf("invalid URL: %v", err),
			})
		}
	}
}

// Validate returns all validation errors.
func (v *Validator) Validate() error {
	if len(v.errors) == 0 {
		return nil
	}
	
	var messages []string
	for _, err := range v.errors {
		messages = append(messages, err.Error())
	}
	return fmt.Errorf("configuration validation failed:\n  %s", strings.Join(messages, "\n  "))
}

// ValidateGraphService validates Graph service configuration.
func ValidateGraphService() error {
	v := NewValidator()
	
	// Required service URLs
	v.RequireURL("EXTRACT_SERVICE_URL")
	v.RequireURL("AGENTFLOW_SERVICE_URL")
	v.RequireURL("LOCALAI_URL")
	
	// Optional service URLs
	v.OptionalURL("DEEPAGENTS_SERVICE_URL")
	v.OptionalURL("GPU_ORCHESTRATOR_URL")
	
	return v.Validate()
}

