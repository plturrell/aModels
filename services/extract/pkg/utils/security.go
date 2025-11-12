package utils

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	// Valid identifier pattern: alphanumeric, underscore, and hyphen
	// Must start with letter or underscore
	validIdentifierRegex = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_-]*$`)
)

// SanitizeIdentifier validates and sanitizes SQL identifiers (table names, schema names, etc.)
// Returns an error if the identifier is invalid
func SanitizeIdentifier(identifier string) (string, error) {
	if identifier == "" {
		return "", fmt.Errorf("identifier cannot be empty")
	}

	// Remove any quotes that might be present
	identifier = strings.Trim(identifier, `"'`)

	// Check for SQL injection patterns
	if strings.Contains(identifier, ";") ||
		strings.Contains(identifier, "--") ||
		strings.Contains(identifier, "/*") ||
		strings.Contains(identifier, "*/") ||
		strings.Contains(identifier, "xp_") ||
		strings.Contains(identifier, "sp_") {
		return "", fmt.Errorf("invalid identifier: contains potentially dangerous characters")
	}

	// Validate identifier format
	if !validIdentifierRegex.MatchString(identifier) {
		return "", fmt.Errorf("invalid identifier format: %s", identifier)
	}

	return identifier, nil
}

// SanitizeSchemaAndTable validates both schema and table names
func SanitizeSchemaAndTable(schema, table string) (string, string, error) {
	sanitizedSchema, err := SanitizeIdentifier(schema)
	if err != nil {
		return "", "", fmt.Errorf("invalid schema name: %w", err)
	}

	sanitizedTable, err := SanitizeIdentifier(table)
	if err != nil {
		return "", "", fmt.Errorf("invalid table name: %w", err)
	}

	return sanitizedSchema, sanitizedTable, nil
}

// BuildSafeQuery builds a safe SQL query with validated identifiers
func BuildSafeQuery(queryTemplate string, args ...interface{}) (string, error) {
	// Validate all string arguments that are identifiers
	for i, arg := range args {
		if str, ok := arg.(string); ok {
			// Check if this looks like an identifier (not a value)
			if strings.Contains(queryTemplate, "%s") {
				// This is a simple check - in production, use parameterized queries
				if _, err := SanitizeIdentifier(str); err != nil {
					return "", fmt.Errorf("unsafe identifier at position %d: %w", i, err)
				}
			}
		}
	}
	return fmt.Sprintf(queryTemplate, args...), nil
}

