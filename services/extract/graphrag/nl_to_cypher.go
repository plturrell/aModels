package graphrag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
)

// NLToCypherTranslator translates natural language queries to Cypher queries.
type NLToCypherTranslator struct {
	localaiURL string
	httpClient *http.Client
	logger     *log.Logger
}

// NewNLToCypherTranslator creates a new NL-to-Cypher translator.
func NewNLToCypherTranslator(localaiURL string, logger *log.Logger) *NLToCypherTranslator {
	return &NLToCypherTranslator{
		localaiURL: localaiURL,
		httpClient: &http.Client{},
		logger:     logger,
	}
}

// Translate translates a natural language query to a Cypher query.
func (nct *NLToCypherTranslator) Translate(ctx context.Context, query string, schemaContext string) (string, map[string]interface{}, error) {
	if nct.logger != nil {
		nct.logger.Printf("Translating NL query to Cypher: %s", query)
	}

	// Build prompt for LLM
	prompt := nct.buildTranslationPrompt(query, schemaContext)

	// Call LocalAI to translate
	cypherQuery, params, err := nct.callLLM(ctx, prompt)
	if err != nil {
		return "", nil, fmt.Errorf("LLM translation failed: %w", err)
	}

	// Validate Cypher syntax (basic check)
	if err := nct.validateCypher(cypherQuery); err != nil {
		return "", nil, fmt.Errorf("invalid Cypher query generated: %w", err)
	}

	if nct.logger != nil {
		nct.logger.Printf("Translated query: %s", cypherQuery)
	}

	return cypherQuery, params, nil
}

// buildTranslationPrompt builds a prompt for translating NL to Cypher.
func (nct *NLToCypherTranslator) buildTranslationPrompt(query string, schemaContext string) string {
	return fmt.Sprintf(`You are a Cypher query translator. Translate the following natural language question into a valid Neo4j Cypher query.

Natural Language Question: %s

Schema Context:
%s

Instructions:
1. Identify the entities mentioned in the question
2. Map them to Neo4j node labels (e.g., "trade" -> :Trade, "journal entry" -> :JournalEntry)
3. Identify relationships between entities
4. Generate a valid Cypher query that answers the question
5. Return ONLY the Cypher query, no explanations
6. Use parameterized queries with $paramName for values

Example:
Question: "How is a trade's cashflow reported for MAS 610?"
Cypher: MATCH (t:Trade)-[:HAS_CASHFLOW]->(cf:CashFlow)-[:REPORTED_IN]->(r:RegulatoryReport {report_type: $reportType})
       WHERE r.report_type = 'MAS_610'
       RETURN t.id, cf.amount, r.report_date
       LIMIT 10

Parameters: {"reportType": "MAS_610"}

Now translate this question:
%[1]s

Cypher Query:`, query, schemaContext)
}

// callLLM calls the LocalAI service to translate the query.
func (nct *NLToCypherTranslator) callLLM(ctx context.Context, prompt string) (string, map[string]interface{}, error) {
	payload := map[string]interface{}{
		"model": "general", // Default domain
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": 0.1, // Low temperature for deterministic queries
		"max_tokens":  500,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, nct.localaiURL+"/v1/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return "", nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := nct.httpClient.Do(req)
	if err != nil {
		return "", nil, fmt.Errorf("failed to call LocalAI: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", nil, fmt.Errorf("LocalAI returned status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Extract Cypher query from response
	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", nil, fmt.Errorf("invalid response format")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", nil, fmt.Errorf("invalid choice format")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "", nil, fmt.Errorf("invalid message format")
	}

	content, ok := message["content"].(string)
	if !ok {
		return "", nil, fmt.Errorf("invalid content format")
	}

	// Parse Cypher query and parameters
	cypherQuery, params := nct.parseCypherResponse(content)

	return cypherQuery, params, nil
}

// parseCypherResponse parses the LLM response to extract Cypher query and parameters.
func (nct *NLToCypherTranslator) parseCypherResponse(content string) (string, map[string]interface{}) {
	// Remove markdown code blocks if present
	content = strings.TrimSpace(content)
	if strings.HasPrefix(content, "```") {
		lines := strings.Split(content, "\n")
		content = strings.Join(lines[1:len(lines)-1], "\n")
	}

	// Extract parameters if present in response
	params := make(map[string]interface{})
	
	// Look for JSON parameters block
	if idx := strings.Index(content, "Parameters:"); idx != -1 {
		paramBlock := content[idx:]
		if jsonIdx := strings.Index(paramBlock, "{"); jsonIdx != -1 {
			jsonStr := paramBlock[jsonIdx:]
			if endIdx := strings.Index(jsonStr, "}"); endIdx != -1 {
				jsonStr = jsonStr[:endIdx+1]
				json.Unmarshal([]byte(jsonStr), &params)
			}
		}
		// Remove parameters block from query
		content = strings.TrimSpace(content[:idx])
	}

	return content, params
}

// validateCypher performs basic Cypher syntax validation.
func (nct *NLToCypherTranslator) validateCypher(cypher string) error {
	cypher = strings.TrimSpace(cypher)
	
	// Basic checks
	if len(cypher) == 0 {
		return fmt.Errorf("empty Cypher query")
	}

	// Check for common Cypher keywords
	upperCypher := strings.ToUpper(cypher)
	if !strings.Contains(upperCypher, "MATCH") && !strings.Contains(upperCypher, "RETURN") {
		return fmt.Errorf("query must contain MATCH or RETURN")
	}

	return nil
}

