package triplestore

import (
	"context"
	"fmt"
	"log"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// TriplestoreClient provides access to a triplestore (RDF database).
// This implementation uses Neo4j with RDF/SPARQL support.
type TriplestoreClient struct {
	driver neo4j.DriverWithContext
	logger *log.Logger
}

// NewTriplestoreClient creates a new triplestore client using Neo4j.
func NewTriplestoreClient(uri, username, password string, logger *log.Logger) (*TriplestoreClient, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create neo4j driver: %w", err)
	}

	return &TriplestoreClient{
		driver: driver,
		logger: logger,
	}, nil
}

// Close closes the triplestore connection.
func (c *TriplestoreClient) Close() error {
	return c.driver.Close(context.Background())
}

// Triple represents an RDF triple (subject, predicate, object).
type Triple struct {
	Subject   string
	Predicate string
	Object    string
	ObjectType string // "uri" or "literal"
	DataType   string // For literals, e.g., "xsd:string"
}

// StoreTriple stores a single RDF triple in the triplestore.
func (c *TriplestoreClient) StoreTriple(ctx context.Context, triple Triple) error {
	// Store as RDF triple in Neo4j
	// Neo4j n10s plugin can store RDF, but for now we'll use a simple node-based approach
	query := `
		MERGE (s:Resource {uri: $subject})
		MERGE (o:Resource {uri: $object})
		MERGE (s)-[:PREDICATE {predicate: $predicate}]->(o)
		SET s.updated_at = datetime()
		SET o.updated_at = datetime()
	`
	
	params := map[string]any{
		"subject":   triple.Subject,
		"predicate": triple.Predicate,
		"object":    triple.Object,
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})
	return err
}

// StoreTriples stores multiple RDF triples in batch.
func (c *TriplestoreClient) StoreTriples(ctx context.Context, triples []Triple) error {
	if len(triples) == 0 {
		return nil
	}

	// Batch insert triples
	query := `
		UNWIND $triples AS triple
		MERGE (s:Resource {uri: triple.subject})
		MERGE (o:Resource {uri: triple.object})
		MERGE (s)-[:PREDICATE {predicate: triple.predicate}]->(o)
		SET s.updated_at = datetime()
		SET o.updated_at = datetime()
	`

	var tripleMaps []map[string]any
	for _, t := range triples {
		tripleMaps = append(tripleMaps, map[string]any{
			"subject":   t.Subject,
			"predicate": t.Predicate,
			"object":    t.Object,
		})
	}

	params := map[string]any{
		"triples": tripleMaps,
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})
	return err
}

// DeleteTriple deletes a specific RDF triple.
func (c *TriplestoreClient) DeleteTriple(ctx context.Context, triple Triple) error {
	query := `
		MATCH (s:Resource {uri: $subject})-[r:PREDICATE {predicate: $predicate}]->(o:Resource {uri: $object})
		DELETE r
	`

	params := map[string]any{
		"subject":   triple.Subject,
		"predicate": triple.Predicate,
		"object":    triple.Object,
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})
	return err
}

// GetTriplesBySubject retrieves all triples where the given URI is the subject.
func (c *TriplestoreClient) GetTriplesBySubject(ctx context.Context, subjectURI string) ([]Triple, error) {
	query := `
		MATCH (s:Resource {uri: $subject})-[r:PREDICATE]->(o:Resource)
		RETURN s.uri AS subject, r.predicate AS predicate, o.uri AS object
	`

	params := map[string]any{
		"subject": subjectURI,
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
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

	var triples []Triple
	records, ok := result.([]*neo4j.Record)
	if ok {
		for _, record := range records {
			subject, _ := record.Get("subject")
			predicate, _ := record.Get("predicate")
			object, _ := record.Get("object")

			triples = append(triples, Triple{
				Subject:   getStringValue(subject),
				Predicate: getStringValue(predicate),
				Object:    getStringValue(object),
				ObjectType: "uri",
			})
		}
	}

	return triples, nil
}

// GetTriplesByPredicate retrieves all triples with the given predicate.
func (c *TriplestoreClient) GetTriplesByPredicate(ctx context.Context, predicateURI string) ([]Triple, error) {
	query := `
		MATCH (s:Resource)-[r:PREDICATE {predicate: $predicate}]->(o:Resource)
		RETURN s.uri AS subject, r.predicate AS predicate, o.uri AS object
	`

	params := map[string]any{
		"predicate": predicateURI,
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
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

	var triples []Triple
	records, ok := result.([]*neo4j.Record)
	if ok {
		for _, record := range records {
			subject, _ := record.Get("subject")
			predicate, _ := record.Get("predicate")
			object, _ := record.Get("object")

			triples = append(triples, Triple{
				Subject:   getStringValue(subject),
				Predicate: getStringValue(predicate),
				Object:    getStringValue(object),
				ObjectType: "uri",
			})
		}
	}

	return triples, nil
}

// ClearAll clears all triples from the triplestore.
func (c *TriplestoreClient) ClearAll(ctx context.Context) error {
	query := `
		MATCH (s:Resource)-[r:PREDICATE]->(o:Resource)
		DELETE r, s, o
	`

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, nil)
	})
	return err
}

// getStringValue safely extracts a string value from an interface{}.
func getStringValue(val any) string {
	if str, ok := val.(string); ok {
		return str
	}
	return ""
}

