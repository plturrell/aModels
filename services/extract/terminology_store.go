package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// TerminologyStore interface for storing and retrieving learned terminology.
type TerminologyStore interface {
	StoreTerminology(ctx context.Context, nodes []Node, edges []Edge, timestamp time.Time) error
	LoadTerminology(ctx context.Context) (*StoredTerminology, error)
	GetTerminologyEvolution(ctx context.Context, startTime, endTime time.Time) (*TerminologyEvolution, error)
}

// StoredTerminology represents stored terminology data.
type StoredTerminology struct {
	Domains map[string][]TerminologyExample `json:"domains"`
	Roles   map[string][]TerminologyExample `json:"roles"`
	Patterns map[string][]TerminologyExample `json:"patterns"`
}

// TerminologyExample represents a single terminology example.
type TerminologyExample struct {
	Text      string    `json:"text"`
	Timestamp time.Time `json:"timestamp"`
	Confidence float64  `json:"confidence"`
	Context   map[string]any `json:"context,omitempty"`
}

// TerminologyEvolution represents terminology changes over time.
type TerminologyEvolution struct {
	Domains []DomainEvolution `json:"domains"`
	Roles   []RoleEvolution   `json:"roles"`
	Drift   float64           `json:"semantic_drift"`
}

// DomainEvolution represents domain terminology evolution.
type DomainEvolution struct {
	Domain      string    `json:"domain"`
	FirstSeen   time.Time `json:"first_seen"`
	LastSeen    time.Time `json:"last_seen"`
	ExampleCount int      `json:"example_count"`
	Confidence   float64  `json:"confidence"`
}

// RoleEvolution represents role terminology evolution.
type RoleEvolution struct {
	Role        string    `json:"role"`
	FirstSeen   time.Time `json:"first_seen"`
	LastSeen    time.Time `json:"last_seen"`
	ExampleCount int      `json:"example_count"`
	Confidence   float64  `json:"confidence"`
}

// Neo4jTerminologyStore implements TerminologyStore using Neo4j.
type Neo4jTerminologyStore struct {
	neo4jPersistence *Neo4jPersistence
	logger           *log.Logger
}

// NewNeo4jTerminologyStore creates a new Neo4j terminology store.
func NewNeo4jTerminologyStore(neo4jPersistence *Neo4jPersistence, logger *log.Logger) *Neo4jTerminologyStore {
	return &Neo4jTerminologyStore{
		neo4jPersistence: neo4jPersistence,
		logger:           logger,
	}
}

// StoreTerminology stores terminology in Neo4j.
func (nts *Neo4jTerminologyStore) StoreTerminology(
	ctx context.Context,
	nodes []Node,
	edges []Edge,
	timestamp time.Time,
) error {
	if nts.neo4jPersistence == nil {
		return fmt.Errorf("neo4j persistence not available")
	}

	// Create terminology nodes in Neo4j
	query := `
		UNWIND $nodes AS node
		MERGE (t:Terminology {text: node.text, type: node.type})
		SET t.last_seen = $timestamp,
		    t.confidence = COALESCE(t.confidence, 0.0) + node.confidence,
		    t.example_count = COALESCE(t.example_count, 0) + 1
		WITH t, node
		WHERE node.domain IS NOT NULL
		MERGE (d:Domain {name: node.domain})
		MERGE (t)-[:BELONGS_TO]->(d)
	`

	params := map[string]any{
		"nodes":    extractTerminologyNodes(nodes, timestamp),
		"timestamp": timestamp.Format(time.RFC3339),
	}

	// Execute query using Neo4j session
	session := nts.neo4jPersistence.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}
		return result.Consume(ctx)
	})
	if err != nil {
		return fmt.Errorf("failed to store terminology: %w", err)
	}

	nts.logger.Printf("Stored terminology for %d nodes", len(nodes))
	return nil
}

// LoadTerminology loads terminology from Neo4j.
func (nts *Neo4jTerminologyStore) LoadTerminology(ctx context.Context) (*StoredTerminology, error) {
	if nts.neo4jPersistence == nil {
		return &StoredTerminology{
			Domains: make(map[string][]TerminologyExample),
			Roles:   make(map[string][]TerminologyExample),
			Patterns: make(map[string][]TerminologyExample),
		}, nil
	}

	// Query terminology from Neo4j
	query := `
		MATCH (t:Terminology)-[:BELONGS_TO]->(d:Domain)
		RETURN t.text AS text, d.name AS domain, t.last_seen AS timestamp, t.confidence AS confidence
		ORDER BY t.last_seen DESC
		LIMIT 10000
	`

	// Execute query using Neo4j session
	session := nts.neo4jPersistence.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load terminology: %w", err)
	}

	terminology := &StoredTerminology{
		Domains: make(map[string][]TerminologyExample),
		Roles:   make(map[string][]TerminologyExample),
		Patterns: make(map[string][]TerminologyExample),
	}

	// Process results
	records, ok := result.([]*neo4j.Record)
	if ok {
		for _, record := range records {
			text, _ := record.Get("text")
			domain, _ := record.Get("domain")
			timestampVal, _ := record.Get("timestamp")
			confidence, _ := record.Get("confidence")

			if textStr, ok := text.(string); ok {
				if domainStr, ok := domain.(string); ok {
					example := TerminologyExample{
						Text:       textStr,
						Timestamp:  time.Now(),
						Confidence: 0.8,
					}
					if tsStr, ok := timestampVal.(string); ok {
						if parsed, err := time.Parse(time.RFC3339, tsStr); err == nil {
							example.Timestamp = parsed
						}
					}
					if conf, ok := confidence.(float64); ok {
						example.Confidence = conf
					}
					terminology.Domains[domainStr] = append(terminology.Domains[domainStr], example)
				}
			}
		}
	}

	return terminology, nil
}

// GetTerminologyEvolution gets terminology evolution over time.
func (nts *Neo4jTerminologyStore) GetTerminologyEvolution(
	ctx context.Context,
	startTime, endTime time.Time,
) (*TerminologyEvolution, error) {
	if nts.neo4jPersistence == nil {
		return &TerminologyEvolution{}, nil
	}

	query := `
		MATCH (t:Terminology)-[:BELONGS_TO]->(d:Domain)
		WHERE t.last_seen >= $startTime AND t.last_seen <= $endTime
		RETURN d.name AS domain, 
		       MIN(t.last_seen) AS first_seen,
		       MAX(t.last_seen) AS last_seen,
		       COUNT(t) AS example_count,
		       AVG(t.confidence) AS confidence
		ORDER BY example_count DESC
	`

	params := map[string]any{
		"startTime": startTime.Format(time.RFC3339),
		"endTime":   endTime.Format(time.RFC3339),
	}

	// Execute query using Neo4j session
	session := nts.neo4jPersistence.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get terminology evolution: %w", err)
	}

	evolution := &TerminologyEvolution{
		Domains: []DomainEvolution{},
		Roles:   []RoleEvolution{},
	}

	if records, ok := result.([]*neo4j.Record); ok {
		for _, record := range records {
			domainVal, _ := record.Get("domain")
			firstSeenVal, _ := record.Get("first_seen")
			lastSeenVal, _ := record.Get("last_seen")
			exampleCountVal, _ := record.Get("example_count")
			confidenceVal, _ := record.Get("confidence")

			domainStr, _ := domainVal.(string)
			firstSeen := parseNeo4jTime(firstSeenVal)
			lastSeen := parseNeo4jTime(lastSeenVal)

			exampleCount := 0
			if count, ok := exampleCountVal.(int64); ok {
				exampleCount = int(count)
			}

			confidence := 0.0
			if conf, ok := confidenceVal.(float64); ok {
				confidence = conf
			}

			evolution.Domains = append(evolution.Domains, DomainEvolution{
				Domain:      domainStr,
				FirstSeen:   firstSeen,
				LastSeen:    lastSeen,
				ExampleCount: exampleCount,
				Confidence:   confidence,
			})
		}
	}

	if len(evolution.Domains) > 1 {
		first := evolution.Domains[0]
		last := evolution.Domains[len(evolution.Domains)-1]
		if first.Confidence > 0 {
			evolution.Drift = absFloat(first.Confidence - last.Confidence)
		}
	}

	return evolution, nil
}

// Helper functions

func extractTerminologyNodes(nodes []Node, timestamp time.Time) []map[string]any {
	terminologyNodes := []map[string]any{}

	for _, node := range nodes {
		termNode := map[string]any{
			"text":      node.Label,
			"type":      node.Type,
			"timestamp": timestamp.Format(time.RFC3339),
			"confidence": 0.8, // Default confidence
		}

		// Extract domain from context
		if node.Props != nil {
			if domain, ok := node.Props["domain"].(string); ok {
				termNode["domain"] = domain
			}
		}

		terminologyNodes = append(terminologyNodes, termNode)
	}

	return terminologyNodes
}

func parseNeo4jTime(value any) time.Time {
	switch v := value.(type) {
	case time.Time:
		return v
	case string:
		if parsed, err := time.Parse(time.RFC3339, v); err == nil {
			return parsed
		}
	}
	return time.Time{}
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
