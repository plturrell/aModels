package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/redis/go-redis/v9"
)

// ConsistencyResult represents the result of consistency validation
type ConsistencyResult struct {
	Consistent bool
	Issues     []ConsistencyIssue
	Metrics    ConsistencyMetrics
}

// ConsistencyIssue represents a specific consistency problem
type ConsistencyIssue struct {
	Type        string // "node_count_mismatch", "edge_count_mismatch", "missing_data"
	Severity    string // "high", "medium", "low"
	Message     string
	System      string
	Details     map[string]interface{}
}

// ConsistencyMetrics tracks consistency statistics
type ConsistencyMetrics struct {
	PostgresNodes int
	RedisNodes    int
	Neo4jNodes    int
	PostgresEdges int
	RedisEdges    int
	Neo4jEdges    int
	NodeVariance  int
	EdgeVariance  int
	ValidationTime time.Duration
}

// ValidateConsistency checks data consistency across all storage systems
func ValidateConsistency(ctx context.Context, projectID string, logger *log.Logger) ConsistencyResult {
	startTime := time.Now()
	result := ConsistencyResult{
		Consistent: true,
		Issues:     []ConsistencyIssue{},
		Metrics: ConsistencyMetrics{},
	}

	// Get counts from each system
	pgNodes, pgEdges := getPostgresCounts(ctx, projectID, logger)
	redisNodes, redisEdges := getRedisCounts(ctx, logger)
	neo4jNodes, neo4jEdges := getNeo4jCounts(ctx, projectID, logger)

	result.Metrics.PostgresNodes = pgNodes
	result.Metrics.RedisNodes = redisNodes
	result.Metrics.Neo4jNodes = neo4jNodes
	result.Metrics.PostgresEdges = pgEdges
	result.Metrics.RedisEdges = redisEdges
	result.Metrics.Neo4jEdges = neo4jEdges

	// Calculate variance
	result.Metrics.NodeVariance = calculateVariance(pgNodes, redisNodes, neo4jNodes)
	result.Metrics.EdgeVariance = calculateVariance(pgEdges, redisEdges, neo4jEdges)

	// Check for issues
	maxNodes := max(pgNodes, redisNodes, neo4jNodes)
	maxEdges := max(pgEdges, redisEdges, neo4jEdges)

	// Node count consistency
	if result.Metrics.NodeVariance > 0 {
		variancePct := float64(result.Metrics.NodeVariance) / float64(maxNodes) * 100
		if variancePct > 5 {
			result.Consistent = false
			severity := "medium"
			if variancePct > 20 {
				severity = "high"
			}
			result.Issues = append(result.Issues, ConsistencyIssue{
				Type:     "node_count_mismatch",
				Severity: severity,
				Message:  fmt.Sprintf("Node count variance: %d (%.1f%%)", result.Metrics.NodeVariance, variancePct),
				Details: map[string]interface{}{
					"postgres": pgNodes,
					"redis":    redisNodes,
					"neo4j":    neo4jNodes,
					"variance": result.Metrics.NodeVariance,
				},
			})
		}
	}

	// Edge count consistency
	if result.Metrics.EdgeVariance > 0 {
		variancePct := float64(result.Metrics.EdgeVariance) / float64(maxEdges) * 100
		if variancePct > 5 {
			result.Consistent = false
			severity := "medium"
			if variancePct > 20 {
				severity = "high"
			}
			result.Issues = append(result.Issues, ConsistencyIssue{
				Type:     "edge_count_mismatch",
				Severity: severity,
				Message:  fmt.Sprintf("Edge count variance: %d (%.1f%%)", result.Metrics.EdgeVariance, variancePct),
				Details: map[string]interface{}{
					"postgres": pgEdges,
					"redis":    redisEdges,
					"neo4j":    neo4jEdges,
					"variance": result.Metrics.EdgeVariance,
				},
			})
		}
	}

	// Check for missing data
	if pgNodes == 0 && pgEdges == 0 {
		result.Consistent = false
		result.Issues = append(result.Issues, ConsistencyIssue{
			Type:     "missing_data",
			Severity: "high",
			Message:  "No data found in Postgres",
			System:   "postgres",
		})
	}

	if neo4jNodes == 0 && neo4jEdges == 0 {
		result.Consistent = false
		result.Issues = append(result.Issues, ConsistencyIssue{
			Type:     "missing_data",
			Severity: "high",
			Message:  "No data found in Neo4j",
			System:   "neo4j",
		})
	}

	result.Metrics.ValidationTime = time.Since(startTime)

	if logger != nil {
		if result.Consistent {
			logger.Printf("Consistency validation passed: nodes variance=%d, edges variance=%d", 
				result.Metrics.NodeVariance, result.Metrics.EdgeVariance)
		} else {
			logger.Printf("Consistency validation found %d issues", len(result.Issues))
			for _, issue := range result.Issues {
				logger.Printf("  [%s] %s: %s", issue.Severity, issue.Type, issue.Message)
			}
		}
	}

	return result
}

// getPostgresCounts retrieves node and edge counts from Postgres
func getPostgresCounts(ctx context.Context, projectID string, logger *log.Logger) (int, int) {
	dsn := os.Getenv("POSTGRES_CATALOG_DSN")
	if dsn == "" {
		return 0, 0
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		if logger != nil {
			logger.Printf("Failed to connect to Postgres for consistency check: %v", err)
		}
		return 0, 0
	}
	defer db.Close()

	var nodeCount, edgeCount int
	err = db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM glean_nodes 
		WHERE properties_json->>'project_id' = $1
	`, projectID).Scan(&nodeCount)
	if err != nil {
		if logger != nil {
			logger.Printf("Failed to query Postgres node count: %v", err)
		}
		return 0, 0
	}

	err = db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM glean_edges e
		JOIN glean_nodes n1 ON e.source_id = n1.id
		WHERE n1.properties_json->>'project_id' = $1
	`, projectID).Scan(&edgeCount)
	if err != nil {
		if logger != nil {
			logger.Printf("Failed to query Postgres edge count: %v", err)
		}
		return nodeCount, 0
	}

	return nodeCount, edgeCount
}

// getRedisCounts retrieves node and edge counts from Redis
func getRedisCounts(ctx context.Context, logger *log.Logger) (int, int) {
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		return 0, 0
	}

	client := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: os.Getenv("REDIS_PASSWORD"),
		DB:       0,
	})
	defer client.Close()

	// Count schema nodes
	nodeKeys, err := client.Keys(ctx, "schema:node:*").Result()
	if err != nil {
		if logger != nil {
			logger.Printf("Failed to query Redis node count: %v", err)
		}
		return 0, 0
	}

	// Count schema edges
	edgeKeys, err := client.Keys(ctx, "schema:edge:*").Result()
	if err != nil {
		if logger != nil {
			logger.Printf("Failed to query Redis edge count: %v", err)
		}
		return len(nodeKeys), 0
	}

	return len(nodeKeys), len(edgeKeys)
}

// getNeo4jCounts retrieves node and edge counts from Neo4j
func getNeo4jCounts(ctx context.Context, projectID string, logger *log.Logger) (int, int) {
	uri := os.Getenv("NEO4J_URI")
	if uri == "" {
		return 0, 0
	}

	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(
		os.Getenv("NEO4J_USERNAME"),
		os.Getenv("NEO4J_PASSWORD"),
		"",
	))
	if err != nil {
		if logger != nil {
			logger.Printf("Failed to connect to Neo4j for consistency check: %v", err)
		}
		return 0, 0
	}
	defer driver.Close(ctx)

	session := driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	var nodeCount, edgeCount int64

	result, err := session.Run(ctx, "MATCH (n:Node) RETURN COUNT(n) as count", nil)
	if err == nil {
		if record, err := result.Single(ctx); err == nil {
			if count, ok := record.Get("count"); ok {
				nodeCount = count.(int64)
			}
		}
	}

	result, err = session.Run(ctx, "MATCH ()-[r:RELATIONSHIP]->() RETURN COUNT(r) as count", nil)
	if err == nil {
		if record, err := result.Single(ctx); err == nil {
			if count, ok := record.Get("count"); ok {
				edgeCount = count.(int64)
			}
		}
	}

	return int(nodeCount), int(edgeCount)
}

// calculateVariance calculates the variance between three values
func calculateVariance(a, b, c int) int {
	maxVal := max(a, b, c)
	minVal := min(a, b, c)
	return maxVal - minVal
}

// max returns the maximum of three integers
func max(a, b, c int) int {
	if a > b && a > c {
		return a
	}
	if b > c {
		return b
	}
	return c
}

// min returns the minimum of three integers
func min(a, b, c int) int {
	if a < b && a < c {
		return a
	}
	if b < c {
		return b
	}
	return c
}

// RunConsistencyValidationScript runs the Python validation script
func RunConsistencyValidationScript(projectID string, logger *log.Logger) ConsistencyResult {
	scriptPath := filepath.Join("scripts", "validate_sgmi_data_flow.py")
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		// Try alternative path
		scriptPath = filepath.Join("..", "..", "scripts", "validate_sgmi_data_flow.py")
	}

	cmd := exec.Command("python3", scriptPath, "--project-id", projectID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		if logger != nil {
			logger.Printf("Consistency validation script failed: %v, output: %s", err, string(output))
		}
		return ConsistencyResult{
			Consistent: false,
			Issues: []ConsistencyIssue{{
				Type:     "validation_error",
				Severity: "high",
				Message:  fmt.Sprintf("Validation script failed: %v", err),
			}},
		}
	}

	if logger != nil {
		logger.Printf("Consistency validation script output: %s", string(output))
	}

	// Parse output would go here if script returns structured data
	// For now, assume success if script exits without error
	return ConsistencyResult{
		Consistent: true,
		Issues:     []ConsistencyIssue{},
	}
}

