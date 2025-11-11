package bridge

import (
	"context"
	"fmt"
	"log"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/catalog/owl"
	"github.com/plturrell/aModels/services/catalog/triplestore"
)

// OWLConverter converts Neo4j graph nodes and edges to OWL ontology.
type OWLConverter struct {
	neo4jDriver neo4j.DriverWithContext
	logger      *log.Logger
}

// NewOWLConverter creates a new OWL converter.
func NewOWLConverter(neo4jDriver neo4j.DriverWithContext, logger *log.Logger) *OWLConverter {
	return &OWLConverter{
		neo4jDriver: neo4jDriver,
		logger:      logger,
	}
}

// ConvertGraphToOWL converts Neo4j graph to OWL ontology.
func (c *OWLConverter) ConvertGraphToOWL(ctx context.Context, baseURI string) (*owl.Ontology, error) {
	ontology := owl.NewOntology(baseURI)

	// Query nodes
	nodes, err := c.queryNodes(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to query nodes: %w", err)
	}

	// Query edges
	edges, err := c.queryEdges(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to query edges: %w", err)
	}

	// Convert nodes to OWL individuals
	for _, node := range nodes {
		c.convertNodeToOWL(ontology, node, baseURI)
	}

	// Convert edges to OWL object properties
	for _, edge := range edges {
		c.convertEdgeToOWL(ontology, edge, baseURI)
	}

	return ontology, nil
}

// ConvertGraphToTriples converts Neo4j graph directly to RDF triples.
func (c *OWLConverter) ConvertGraphToTriples(ctx context.Context, baseURI string) ([]triplestore.Triple, error) {
	var triples []triplestore.Triple

	// Query nodes
	nodes, err := c.queryNodes(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to query nodes: %w", err)
	}

	// Query edges
	edges, err := c.queryEdges(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to query edges: %w", err)
	}

	// Convert nodes to triples
	for _, node := range nodes {
		nodeTriples := c.convertNodeToTriples(node, baseURI)
		triples = append(triples, nodeTriples...)
	}

	// Convert edges to triples
	for _, edge := range edges {
		edgeTriples := c.convertEdgeToTriples(edge, baseURI)
		triples = append(triples, edgeTriples...)
	}

	return triples, nil
}

// queryNodes queries Neo4j for nodes (same as GraphMapper).
func (c *OWLConverter) queryNodes(ctx context.Context) ([]Node, error) {
	query := `
		MATCH (n)
		RETURN n.id AS id, n.type AS type, n.label AS label, n.properties_json AS props_json
		LIMIT 10000
	`

	session := c.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return nil, err
	}

	var nodes []Node
	records, ok := result.([]*neo4j.Record)
	if ok {
		for _, record := range records {
			id, _ := record.Get("id")
			type_, _ := record.Get("type")
			label, _ := record.Get("label")
			propsJSON, _ := record.Get("props_json")

			nodes = append(nodes, Node{
				ID:    getStringValue(id),
				Type:  getStringValue(type_),
				Label: getStringValue(label),
				Props: parsePropsJSON(getStringValue(propsJSON)),
			})
		}
	}

	return nodes, nil
}

// queryEdges queries Neo4j for edges (same as GraphMapper).
func (c *OWLConverter) queryEdges(ctx context.Context) ([]Edge, error) {
	query := `
		MATCH (s)-[r]->(t)
		RETURN s.id AS source_id, t.id AS target_id, type(r) AS label, r.id AS edge_id, r.properties_json AS props_json
		LIMIT 10000
	`

	session := c.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}
		return result.Collect(ctx)
	})
	if err != nil {
		return nil, err
	}

	var edges []Edge
	records, ok := result.([]*neo4j.Record)
	if ok {
		for _, record := range records {
			sourceID, _ := record.Get("source_id")
			targetID, _ := record.Get("target_id")
			label, _ := record.Get("label")
			edgeID, _ := record.Get("edge_id")
			propsJSON, _ := record.Get("props_json")

			edges = append(edges, Edge{
				ID:       getStringValue(edgeID),
				SourceID: getStringValue(sourceID),
				TargetID: getStringValue(targetID),
				Label:    getStringValue(label),
				Props:    parsePropsJSON(getStringValue(propsJSON)),
			})
		}
	}

	return edges, nil
}

// convertNodeToOWL converts a node to OWL individual.
func (c *OWLConverter) convertNodeToOWL(ontology *owl.Ontology, node Node, baseURI string) {
	individualURI := fmt.Sprintf("%s/resource/%s", baseURI, node.ID)
	
	individual := &owl.Individual{
		URI:     individualURI,
		Label:   node.Label,
		Comment: fmt.Sprintf("Node from knowledge graph: %s", node.Type),
		Types:   []string{fmt.Sprintf("%s#%s", baseURI, node.Type)},
		Properties: make(map[string][]owl.PropertyValue),
	}

	// Add label
	individual.Properties["http://www.w3.org/2000/01/rdf-schema#label"] = []owl.PropertyValue{
		{Value: node.Label, Type: "http://www.w3.org/2001/XMLSchema#string"},
	}

	// Add properties from node props
	if node.Props != nil {
		for key, value := range node.Props {
			if strValue, ok := value.(string); ok {
				propURI := fmt.Sprintf("%s#%s", baseURI, key)
				individual.Properties[propURI] = append(individual.Properties[propURI], owl.PropertyValue{
					Value: strValue,
					Type:  "http://www.w3.org/2001/XMLSchema#string",
				})
			}
		}
	}

	ontology.AddIndividual(individual)
}

// convertEdgeToOWL converts an edge to OWL object property assertion.
func (c *OWLConverter) convertEdgeToOWL(ontology *owl.Ontology, edge Edge, baseURI string) {
	sourceURI := fmt.Sprintf("%s/resource/%s", baseURI, edge.SourceID)
	targetURI := fmt.Sprintf("%s/resource/%s", baseURI, edge.TargetID)
	predicateURI := fmt.Sprintf("%s#%s", baseURI, edge.Label)

	// Find source individual and add property assertion
	if individual, exists := ontology.Individuals[sourceURI]; exists {
		individual.Properties[predicateURI] = append(individual.Properties[predicateURI], owl.PropertyValue{
			Value: targetURI,
		})
	}
}

// convertNodeToTriples converts a node to RDF triples.
func (c *OWLConverter) convertNodeToTriples(node Node, baseURI string) []triplestore.Triple {
	var triples []triplestore.Triple

	subjectURI := fmt.Sprintf("%s/resource/%s", baseURI, node.ID)

	// Type assertion
	triples = append(triples, triplestore.Triple{
		Subject:   subjectURI,
		Predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
		Object:    fmt.Sprintf("%s#%s", baseURI, node.Type),
		ObjectType: "uri",
	})

	// Label
	if node.Label != "" {
		triples = append(triples, triplestore.Triple{
			Subject:   subjectURI,
			Predicate: "http://www.w3.org/2000/01/rdf-schema#label",
			Object:    node.Label,
			ObjectType: "literal",
			DataType:  "http://www.w3.org/2001/XMLSchema#string",
		})
	}

	return triples
}

// convertEdgeToTriples converts an edge to RDF triples.
func (c *OWLConverter) convertEdgeToTriples(edge Edge, baseURI string) []triplestore.Triple {
	var triples []triplestore.Triple

	sourceURI := fmt.Sprintf("%s/resource/%s", baseURI, edge.SourceID)
	targetURI := fmt.Sprintf("%s/resource/%s", baseURI, edge.TargetID)
	predicateURI := fmt.Sprintf("%s#%s", baseURI, edge.Label)

	triples = append(triples, triplestore.Triple{
		Subject:   sourceURI,
		Predicate: predicateURI,
		Object:    targetURI,
		ObjectType: "uri",
	})

	return triples
}

