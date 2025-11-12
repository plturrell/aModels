package middleware

import (
	"testing"
	"time"
)

func TestQueryValidator_ValidateQuery(t *testing.T) {
	tests := []struct {
		name    string
		config  QueryValidatorConfig
		query   string
		wantErr bool
		errMsg  string
	}{
		{
			name:   "valid simple query",
			config: DefaultQueryValidatorConfig(),
			query:  "MATCH (n:Node) RETURN n LIMIT 10",
			wantErr: false,
		},
		{
			name:    "missing LIMIT clause",
			config:  DefaultQueryValidatorConfig(),
			query:   "MATCH (n:Node) RETURN n",
			wantErr: true,
			errMsg:  "LIMIT clause is required",
		},
		{
			name:   "LIMIT too high",
			config: DefaultQueryValidatorConfig(),
			query:  "MATCH (n:Node) RETURN n LIMIT 99999",
			wantErr: true,
			errMsg:  "exceeds maximum allowed limit",
		},
		{
			name:    "blocked write operation",
			config:  DefaultQueryValidatorConfig(),
			query:   "CREATE (n:Node {name: 'test'}) RETURN n",
			wantErr: true,
			errMsg:  "not allowed",
		},
		{
			name:    "too many path expansions",
			config:  DefaultQueryValidatorConfig(),
			query:   "MATCH (a)-[*1..3]-(b)-[*1..3]-(c)-[*1..3]-(d)-[*1..3]-(e) RETURN a LIMIT 10",
			wantErr: true,
			errMsg:  "variable-length path expansions",
		},
		{
			name:    "unbounded path expansion",
			config:  DefaultQueryValidatorConfig(),
			query:   "MATCH (a)-[*]-(b) RETURN a LIMIT 10",
			wantErr: true,
			errMsg:  "unbounded path expansion",
		},
		{
			name:    "path range too large",
			config:  DefaultQueryValidatorConfig(),
			query:   "MATCH (a)-[*1..10]-(b) RETURN a LIMIT 10",
			wantErr: true,
			errMsg:  "range [*1..10] is too large",
		},
		{
			name:   "valid path expansion",
			config: DefaultQueryValidatorConfig(),
			query:  "MATCH path = (a:Node)-[*1..3]-(b:Node) RETURN path LIMIT 10",
			wantErr: false,
		},
		{
			name:   "write operations allowed",
			config: QueryValidatorConfig{
				MaxPathExpansions:    3,
				MaxResultLimit:       10000,
				RequireLimit:         false,
				AllowWriteOperations: true,
				QueryTimeout:         30 * time.Second,
			},
			query:   "CREATE (n:Node {name: 'test'}) RETURN n",
			wantErr: false,
		},
		{
			name:    "empty query",
			config:  DefaultQueryValidatorConfig(),
			query:   "",
			wantErr: true,
			errMsg:  "query cannot be empty",
		},
		{
			name:   "query without RETURN (no LIMIT needed)",
			config: DefaultQueryValidatorConfig(),
			query:  "MATCH (n:Node) WHERE n.type = 'test'",
			wantErr: false,
		},
		{
			name:   "complex valid query",
			config: DefaultQueryValidatorConfig(),
			query:  "MATCH (t:Node)-[r:RELATIONSHIP]->(c:Node) WHERE t.type = 'table' AND r.label = 'HAS_COLUMN' RETURN t.label, c.label LIMIT 100",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewQueryValidator(tt.config)
			err := validator.ValidateQuery(tt.query)

			if tt.wantErr {
				if err == nil {
					t.Errorf("ValidateQuery() expected error but got none")
				} else if tt.errMsg != "" && !contains(err.Error(), tt.errMsg) {
					t.Errorf("ValidateQuery() error = %v, want error containing %v", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("ValidateQuery() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestQueryValidator_QueryComplexity(t *testing.T) {
	tests := []struct {
		name       string
		query      string
		minScore   int
		maxScore   int
	}{
		{
			name:     "simple query",
			query:    "MATCH (n:Node) RETURN n LIMIT 10",
			minScore: 5,
			maxScore: 20,
		},
		{
			name:     "complex path query",
			query:    "MATCH path = (a)-[*1..3]-(b)-[*1..3]-(c) RETURN path LIMIT 10",
			minScore: 100,
			maxScore: 200,
		},
		{
			name:     "aggregation query",
			query:    "MATCH (n:Node) WITH n.type as type, count(n) as cnt RETURN type, cnt ORDER BY cnt DESC LIMIT 10",
			minScore: 10,
			maxScore: 50,
		},
		{
			name:     "optional match",
			query:    "MATCH (n:Node) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10",
			minScore: 20,
			maxScore: 50,
		},
	}

	validator := NewQueryValidator(DefaultQueryValidatorConfig())

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			complexity := validator.QueryComplexity(tt.query)

			if complexity < tt.minScore || complexity > tt.maxScore {
				t.Errorf("QueryComplexity() = %v, want between %v and %v", complexity, tt.minScore, tt.maxScore)
			}
		})
	}
}

func TestQueryValidator_ValidateWithDetails(t *testing.T) {
	validator := NewQueryValidator(DefaultQueryValidatorConfig())

	tests := []struct {
		name             string
		query            string
		wantValid        bool
		wantWarnings     bool
	}{
		{
			name:         "valid simple query",
			query:        "MATCH (n:Node) RETURN n LIMIT 10",
			wantValid:    true,
			wantWarnings: false,
		},
		{
			name:         "high complexity query",
			query:        "MATCH (a)-[r1*1..3]-(b)-[r2*1..3]-(c) WITH a, b, c, count(*) as cnt RETURN a, b, c, cnt ORDER BY cnt DESC LIMIT 10",
			wantValid:    true,
			wantWarnings: true,
		},
		{
			name:         "invalid query",
			query:        "MATCH (n:Node) RETURN n",
			wantValid:    false,
			wantWarnings: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := validator.ValidateWithDetails(tt.query)

			if result.Valid != tt.wantValid {
				t.Errorf("ValidateWithDetails() Valid = %v, want %v", result.Valid, tt.wantValid)
			}

			if tt.wantWarnings && len(result.Warnings) == 0 {
				t.Errorf("ValidateWithDetails() expected warnings but got none")
			}

			if !tt.wantWarnings && len(result.Warnings) > 0 {
				t.Errorf("ValidateWithDetails() unexpected warnings: %v", result.Warnings)
			}
		})
	}
}

func TestQueryValidator_GetQueryTimeout(t *testing.T) {
	config := DefaultQueryValidatorConfig()
	config.QueryTimeout = 45 * time.Second

	validator := NewQueryValidator(config)
	timeout := validator.GetQueryTimeout()

	if timeout != 45*time.Second {
		t.Errorf("GetQueryTimeout() = %v, want %v", timeout, 45*time.Second)
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || 
		(len(s) > 0 && len(substr) > 0 && containsHelper(s, substr)))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func BenchmarkQueryValidator_ValidateQuery(b *testing.B) {
	validator := NewQueryValidator(DefaultQueryValidatorConfig())
	query := "MATCH (t:Node)-[r:RELATIONSHIP]->(c:Node) WHERE t.type = 'table' AND r.label = 'HAS_COLUMN' RETURN t.label, c.label LIMIT 100"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validator.ValidateQuery(query)
	}
}

func BenchmarkQueryValidator_QueryComplexity(b *testing.B) {
	validator := NewQueryValidator(DefaultQueryValidatorConfig())
	query := "MATCH path = (a)-[*1..3]-(b)-[*1..3]-(c) WITH a, b, c, count(*) as cnt RETURN a, b, c, cnt ORDER BY cnt DESC LIMIT 10"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validator.QueryComplexity(query)
	}
}
