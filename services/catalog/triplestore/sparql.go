package triplestore

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// SPARQLClient provides SPARQL query execution capabilities.
// Note: Neo4j doesn't natively support SPARQL, so we'll translate SPARQL to Cypher.
// For full SPARQL support, consider using Neo4j n10s plugin or a separate triplestore.
type SPARQLClient struct {
	client *TriplestoreClient
	logger *log.Logger
}

// NewSPARQLClient creates a new SPARQL client.
func NewSPARQLClient(client *TriplestoreClient, logger *log.Logger) *SPARQLClient {
	return &SPARQLClient{
		client: client,
		logger: logger,
	}
}

// QueryResult represents the result of a SPARQL query.
type QueryResult struct {
	Variables []string
	Bindings  []map[string]string
}

// ExecuteQuery executes a SPARQL query and returns results.
// This is a simplified implementation that translates basic SPARQL to Cypher.
// For full SPARQL support, use a proper SPARQL engine or Neo4j n10s plugin.
func (c *SPARQLClient) ExecuteQuery(ctx context.Context, sparqlQuery string) (*QueryResult, error) {
	// Basic SPARQL to Cypher translation for simple SELECT queries
	// This is a simplified parser - for production, use a proper SPARQL parser
	
	sparqlQuery = strings.TrimSpace(sparqlQuery)
	
	// Check if it's a SELECT query
	if !strings.HasPrefix(strings.ToUpper(sparqlQuery), "SELECT") {
		return nil, fmt.Errorf("only SELECT queries are currently supported")
	}

	// Extract variables from SELECT clause
	selectPart := extractBetween(sparqlQuery, "SELECT", "WHERE")
	variables := parseVariables(selectPart)

	// Extract WHERE clause
	wherePart := extractWhereClause(sparqlQuery)

	// Translate to Cypher
	cypherQuery, err := translateSPARQLToCypher(wherePart, variables)
	if err != nil {
		return nil, fmt.Errorf("failed to translate SPARQL to Cypher: %w", err)
	}

	// Execute Cypher query
	session := c.client.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, cypherQuery, nil)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return nil, err
	}

	// Convert Cypher results to SPARQL format
	bindings := convertCypherResultsToBindings(result, variables)

	return &QueryResult{
		Variables: variables,
		Bindings:  bindings,
	}, nil
}

// extractBetween extracts text between two strings (case-insensitive).
func extractBetween(text, start, end string) string {
	upperText := strings.ToUpper(text)
	upperStart := strings.ToUpper(start)
	upperEnd := strings.ToUpper(end)

	startIdx := strings.Index(upperText, upperStart)
	if startIdx == -1 {
		return ""
	}
	startIdx += len(start)

	endIdx := strings.Index(upperText[startIdx:], upperEnd)
	if endIdx == -1 {
		return text[startIdx:]
	}

	return strings.TrimSpace(text[startIdx : startIdx+endIdx])
}

// extractWhereClause extracts the WHERE clause from a SPARQL query.
func extractWhereClause(query string) string {
	upperQuery := strings.ToUpper(query)
	whereIdx := strings.Index(upperQuery, "WHERE")
	if whereIdx == -1 {
		return ""
	}

	// Find the opening brace
	startIdx := whereIdx + 5
	for startIdx < len(query) && (query[startIdx] == ' ' || query[startIdx] == '\n' || query[startIdx] == '\t') {
		startIdx++
	}

	if startIdx < len(query) && query[startIdx] == '{' {
		// Find matching closing brace
		braceCount := 0
		endIdx := startIdx
		for endIdx < len(query) {
			if query[endIdx] == '{' {
				braceCount++
			} else if query[endIdx] == '}' {
				braceCount--
				if braceCount == 0 {
					return query[startIdx+1 : endIdx]
				}
			}
			endIdx++
		}
	}

	return query[startIdx:]
}

// parseVariables parses variable names from a SELECT clause.
func parseVariables(selectPart string) []string {
	selectPart = strings.TrimSpace(selectPart)
	if strings.HasPrefix(strings.ToUpper(selectPart), "DISTINCT") {
		selectPart = strings.TrimSpace(selectPart[8:])
	}

	var variables []string
	parts := strings.Fields(selectPart)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if strings.HasPrefix(part, "?") {
			variables = append(variables, part)
		}
	}
	return variables
}

// translateSPARQLToCypher translates a SPARQL WHERE clause to Cypher.
// This is a very basic translation - for production, use a proper SPARQL parser.
func translateSPARQLToCypher(whereClause string, variables []string) (string, error) {
	// Simple triple pattern matching
	// Example: ?s ?p ?o -> MATCH (s)-[:PREDICATE {predicate: p}]->(o)
	
	// For now, return a simple query that matches all triples
	// In production, parse the WHERE clause properly
	var cypherParts []string
	
	// Match all Resource nodes and their relationships
	cypherParts = append(cypherParts, "MATCH (s:Resource)-[r:PREDICATE]->(o:Resource)")
	
	// Build RETURN clause based on variables
	var returnParts []string
	for _, varName := range variables {
		varName = strings.TrimPrefix(varName, "?")
		switch varName {
		case "s", "subject":
			returnParts = append(returnParts, "s.uri AS "+varName)
		case "p", "predicate":
			returnParts = append(returnParts, "r.predicate AS "+varName)
		case "o", "object":
			returnParts = append(returnParts, "o.uri AS "+varName)
		default:
			returnParts = append(returnParts, "s.uri AS "+varName)
		}
	}
	
	if len(returnParts) == 0 {
		returnParts = append(returnParts, "s.uri AS subject", "r.predicate AS predicate", "o.uri AS object")
	}
	
	cypherParts = append(cypherParts, "RETURN "+strings.Join(returnParts, ", "))
	
	return strings.Join(cypherParts, " "), nil
}

// convertCypherResultsToBindings converts Neo4j results to SPARQL binding format.
func convertCypherResultsToBindings(result any, variables []string) []map[string]string {
	var bindings []map[string]string

	records, ok := result.([]*neo4j.Record)
	if !ok {
		return bindings
	}

	for _, record := range records {
		binding := make(map[string]string)
		for _, varName := range variables {
			varName = strings.TrimPrefix(varName, "?")
			val, found := record.Get(varName)
			if found {
				if str, ok := val.(string); ok {
					binding[varName] = str
				}
			}
		}
		if len(binding) > 0 {
			bindings = append(bindings, binding)
		}
	}

	return bindings
}

// SimpleQuery executes a simple SPARQL query (for testing).
func (c *SPARQLClient) SimpleQuery(ctx context.Context, subject, predicate, object string) ([]Triple, error) {
	var triples []Triple

	query := `
		MATCH (s:Resource {uri: $subject})-[r:PREDICATE]->(o:Resource {uri: $object})
		WHERE r.predicate = $predicate
		RETURN s.uri AS subject, r.predicate AS predicate, o.uri AS object
	`

	params := map[string]any{
		"subject":   subject,
		"predicate": predicate,
		"object":    object,
	}

	session := c.client.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return nil, err
	}

	records, ok := result.([]*neo4j.Record)
	if ok {
		for _, record := range records {
			subj, _ := record.Get("subject")
			pred, _ := record.Get("predicate")
			obj, _ := record.Get("object")

			triples = append(triples, Triple{
				Subject:   getStringValue(subj),
				Predicate: getStringValue(pred),
				Object:    getStringValue(obj),
				ObjectType: "uri",
			})
		}
	}

	return triples, nil
}

