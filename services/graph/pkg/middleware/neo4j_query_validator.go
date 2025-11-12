package middleware

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

// QueryValidator validates Neo4j Cypher queries for safety and performance.
type QueryValidator struct {
	maxPathExpansions     int
	maxResultLimit        int
	requireLimit          bool
	allowedReadOperations []string
	blockedOperations     []string
	queryTimeout          time.Duration
}

// QueryValidatorConfig holds configuration for the query validator.
type QueryValidatorConfig struct {
	MaxPathExpansions     int           // Max number of variable-length path expansions (*)
	MaxResultLimit        int           // Maximum LIMIT value allowed
	RequireLimit          bool          // Whether LIMIT is required
	AllowWriteOperations  bool          // Allow write operations (CREATE, DELETE, SET, etc.)
	QueryTimeout          time.Duration // Maximum query execution time
}

// DefaultQueryValidatorConfig returns safe default configuration.
func DefaultQueryValidatorConfig() QueryValidatorConfig {
	return QueryValidatorConfig{
		MaxPathExpansions:    3,              // Limit path traversals
		MaxResultLimit:       10000,          // Max 10k results
		RequireLimit:         true,           // Always require LIMIT
		AllowWriteOperations: false,          // Read-only by default
		QueryTimeout:         30 * time.Second,
	}
}

// NewQueryValidator creates a new query validator with the given config.
func NewQueryValidator(config QueryValidatorConfig) *QueryValidator {
	validator := &QueryValidator{
		maxPathExpansions:     config.MaxPathExpansions,
		maxResultLimit:        config.MaxResultLimit,
		requireLimit:          config.RequireLimit,
		queryTimeout:          config.QueryTimeout,
		allowedReadOperations: []string{"MATCH", "RETURN", "WHERE", "WITH", "UNWIND", "ORDER", "SKIP", "LIMIT", "OPTIONAL"},
		blockedOperations:     []string{},
	}

	if !config.AllowWriteOperations {
		validator.blockedOperations = []string{
			"CREATE", "DELETE", "REMOVE", "SET", "MERGE", "DETACH",
			"DROP", "CALL", "FOREACH", "LOAD CSV",
		}
	}

	return validator
}

// ValidateQuery validates a Cypher query and returns an error if invalid.
func (v *QueryValidator) ValidateQuery(query string) error {
	if query == "" {
		return fmt.Errorf("query cannot be empty")
	}

	queryUpper := strings.ToUpper(query)

	// Check for blocked operations
	if err := v.checkBlockedOperations(queryUpper); err != nil {
		return err
	}

	// Check path expansion complexity
	if err := v.checkPathExpansions(query); err != nil {
		return err
	}

	// Check for LIMIT clause
	if err := v.checkLimitClause(queryUpper); err != nil {
		return err
	}

	// Check LIMIT value
	if err := v.checkLimitValue(queryUpper); err != nil {
		return err
	}

	// Check for potential performance issues
	if err := v.checkPerformanceIssues(queryUpper); err != nil {
		return err
	}

	return nil
}

// checkBlockedOperations checks if the query contains blocked operations.
func (v *QueryValidator) checkBlockedOperations(queryUpper string) error {
	for _, op := range v.blockedOperations {
		if strings.Contains(queryUpper, op) {
			return fmt.Errorf("operation %s is not allowed", op)
		}
	}
	return nil
}

// checkPathExpansions checks the number of variable-length path expansions.
func (v *QueryValidator) checkPathExpansions(query string) error {
	// Pattern for variable-length paths: [*], [*2], [*..5], [*2..5]
	pathPattern := regexp.MustCompile(`\[\*[0-9]*\.\.[0-9]*\]|\[\*[0-9]*\]|\[\*\]`)
	matches := pathPattern.FindAllString(query, -1)

	if len(matches) > v.maxPathExpansions {
		return fmt.Errorf("query has %d variable-length path expansions, maximum allowed is %d", 
			len(matches), v.maxPathExpansions)
	}

	// Check for unbounded expansions [*]
	for _, match := range matches {
		if match == "[*]" {
			return fmt.Errorf("unbounded path expansion [*] is not allowed, specify a range like [*1..3]")
		}
		
		// Check for very large ranges
		if strings.Contains(match, "..") {
			rangePattern := regexp.MustCompile(`\[?\*?([0-9]+)\.\.([0-9]+)\]?`)
			rangeMatches := rangePattern.FindStringSubmatch(match)
			if len(rangeMatches) == 3 {
				start := 0
				end := 0
				fmt.Sscanf(rangeMatches[1], "%d", &start)
				fmt.Sscanf(rangeMatches[2], "%d", &end)
				
				if end-start > 5 {
					return fmt.Errorf("path expansion range [*%d..%d] is too large, maximum range is 5", start, end)
				}
			}
		}
	}

	return nil
}

// checkLimitClause checks if LIMIT is present when required.
func (v *QueryValidator) checkLimitClause(queryUpper string) error {
	if !v.requireLimit {
		return nil
	}

	// Skip check for queries that don't return data
	if !strings.Contains(queryUpper, "RETURN") {
		return nil
	}

	// Check for LIMIT clause
	if !strings.Contains(queryUpper, "LIMIT") {
		return fmt.Errorf("LIMIT clause is required for queries that RETURN data")
	}

	return nil
}

// checkLimitValue validates the LIMIT value is within bounds.
func (v *QueryValidator) checkLimitValue(queryUpper string) error {
	limitPattern := regexp.MustCompile(`LIMIT\s+(\d+)`)
	matches := limitPattern.FindStringSubmatch(queryUpper)

	if len(matches) > 1 {
		limit := 0
		fmt.Sscanf(matches[1], "%d", &limit)

		if limit > v.maxResultLimit {
			return fmt.Errorf("LIMIT value %d exceeds maximum allowed limit of %d", limit, v.maxResultLimit)
		}

		if limit == 0 {
			return fmt.Errorf("LIMIT value cannot be 0")
		}
	}

	return nil
}

// checkPerformanceIssues checks for common performance anti-patterns.
func (v *QueryValidator) checkPerformanceIssues(queryUpper string) error {
	// Check for Cartesian products (multiple MATCH without relationships)
	matchCount := strings.Count(queryUpper, "MATCH")
	if matchCount > 3 && !strings.Contains(queryUpper, "WHERE") {
		return fmt.Errorf("query may produce Cartesian product: multiple MATCH clauses without WHERE filtering")
	}

	// Check for missing direction in relationships (performance warning)
	undirectedPattern := regexp.MustCompile(`-\[.+?\]-`)
	if matches := undirectedPattern.FindAllString(queryUpper, -1); len(matches) > 2 {
		// This is a warning, not an error - but worth noting
		// Could log this for monitoring
	}

	return nil
}

// QueryComplexity calculates a complexity score for the query.
func (v *QueryValidator) QueryComplexity(query string) int {
	complexity := 0
	queryUpper := strings.ToUpper(query)

	// Base complexity for operations
	complexity += strings.Count(queryUpper, "MATCH") * 10
	complexity += strings.Count(queryUpper, "OPTIONAL") * 15
	complexity += strings.Count(queryUpper, "WITH") * 5

	// Path expansions add significant complexity
	pathPattern := regexp.MustCompile(`\[\*[0-9]*\.\.[0-9]*\]|\[\*[0-9]*\]`)
	matches := pathPattern.FindAllString(query, -1)
	complexity += len(matches) * 50

	// Aggregations add complexity
	complexity += strings.Count(queryUpper, "COUNT(") * 5
	complexity += strings.Count(queryUpper, "COLLECT(") * 10

	// Subqueries add complexity
	complexity += strings.Count(queryUpper, "CALL {") * 20

	return complexity
}

// GetQueryTimeout returns the configured query timeout.
func (v *QueryValidator) GetQueryTimeout() time.Duration {
	return v.queryTimeout
}

// ValidationResult contains the result of query validation.
type ValidationResult struct {
	Valid      bool
	Error      error
	Complexity int
	Warnings   []string
}

// ValidateWithDetails performs comprehensive validation and returns detailed results.
func (v *QueryValidator) ValidateWithDetails(query string) ValidationResult {
	result := ValidationResult{
		Valid:      true,
		Complexity: v.QueryComplexity(query),
		Warnings:   []string{},
	}

	if err := v.ValidateQuery(query); err != nil {
		result.Valid = false
		result.Error = err
		return result
	}

	// Add warnings for high complexity
	if result.Complexity > 100 {
		result.Warnings = append(result.Warnings, 
			fmt.Sprintf("Query complexity is high (%d). Consider simplifying.", result.Complexity))
	}

	// Check for undirected relationships (warning only)
	undirectedPattern := regexp.MustCompile(`-\[.+?\]-`)
	if matches := undirectedPattern.FindAllString(query, -1); len(matches) > 2 {
		result.Warnings = append(result.Warnings, 
			"Query uses undirected relationships which may impact performance. Consider using directed relationships.")
	}

	return result
}
