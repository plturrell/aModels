package bridge

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/owl"
)

// GraphMapper maps Neo4j knowledge graph nodes and edges to ISO 11179 data elements.
type GraphMapper struct {
	neo4jDriver neo4j.DriverWithContext
	registry    *iso11179.MetadataRegistry
	logger      *log.Logger
}

// NewGraphMapper creates a new graph mapper.
func NewGraphMapper(
	neo4jDriver neo4j.DriverWithContext,
	registry *iso11179.MetadataRegistry,
	logger *log.Logger,
) *GraphMapper {
	return &GraphMapper{
		neo4jDriver: neo4jDriver,
		registry:    registry,
		logger:      logger,
	}
}

// Node represents a node from the knowledge graph.
type Node struct {
	ID    string
	Type  string
	Label string
	Props map[string]any
}

// Edge represents an edge from the knowledge graph.
type Edge struct {
	ID       string
	SourceID string
	TargetID string
	Label    string
	Props    map[string]any
}

// MapGraphToISO11179 maps Neo4j graph nodes and edges to ISO 11179 data elements.
func (m *GraphMapper) MapGraphToISO11179(ctx context.Context, baseURI string) error {
	// Query Neo4j for all nodes
	nodes, err := m.queryNodes(ctx)
	if err != nil {
		return fmt.Errorf("failed to query nodes: %w", err)
	}

	// Query Neo4j for all edges
	edges, err := m.queryEdges(ctx)
	if err != nil {
		return fmt.Errorf("failed to query edges: %w", err)
	}

	// Map nodes to ISO 11179
	for _, node := range nodes {
		if err := m.mapNodeToISO11179(node, baseURI); err != nil {
			if m.logger != nil {
				m.logger.Printf("Warning: Failed to map node %s: %v", node.ID, err)
			}
		}
	}

	// Map edges to ISO 11179 relationships
	for _, edge := range edges {
		if err := m.mapEdgeToISO11179(edge, baseURI); err != nil {
			if m.logger != nil {
				m.logger.Printf("Warning: Failed to map edge %s: %v", edge.ID, err)
			}
		}
	}

	if m.logger != nil {
		m.logger.Printf("Mapped %d nodes and %d edges to ISO 11179", len(nodes), len(edges))
	}

	return nil
}

// queryNodes queries Neo4j for all nodes.
func (m *GraphMapper) queryNodes(ctx context.Context) ([]Node, error) {
	query := `
		MATCH (n)
		RETURN n.id AS id, labels(n) AS labels, n.type AS type, n.label AS label, n.properties_json AS props_json
		LIMIT 10000
	`

	session := m.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
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

			node := Node{
				ID:    getStringValue(id),
				Type:  getStringValue(type_),
				Label: getStringValue(label),
				Props: parsePropsJSON(getStringValue(propsJSON)),
			}
			nodes = append(nodes, node)
		}
	}

	return nodes, nil
}

// queryEdges queries Neo4j for all edges.
func (m *GraphMapper) queryEdges(ctx context.Context) ([]Edge, error) {
	query := `
		MATCH (s)-[r]->(t)
		RETURN s.id AS source_id, t.id AS target_id, type(r) AS label, r.id AS edge_id, r.properties_json AS props_json
		LIMIT 10000
	`

	session := m.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
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

			edge := Edge{
				ID:       getStringValue(edgeID),
				SourceID: getStringValue(sourceID),
				TargetID: getStringValue(targetID),
				Label:    getStringValue(label),
				Props:    parsePropsJSON(getStringValue(propsJSON)),
			}
			edges = append(edges, edge)
		}
	}

	return edges, nil
}

// mapNodeToISO11179 maps a node to ISO 11179 data element.
func (m *GraphMapper) mapNodeToISO11179(node Node, baseURI string) error {
	// Map based on node type
	switch node.Type {
	case "table":
		return m.mapTableNode(node, baseURI)
	case "column":
		return m.mapColumnNode(node, baseURI)
	case "sql":
		return m.mapSQLNode(node, baseURI)
	default:
		// Generic node mapping
		return m.mapGenericNode(node, baseURI)
	}
}

// mapTableNode maps a table node to ISO 11179.
func (m *GraphMapper) mapTableNode(node Node, baseURI string) error {
	// Create concept for table
	conceptID := fmt.Sprintf("%s/concept/%s", baseURI, node.ID)
	concept := iso11179.NewDataElementConcept(
		conceptID,
		node.Label,
		"Table",
		"Schema",
		fmt.Sprintf("Table schema for %s", node.Label),
	)
	m.registry.RegisterDataElementConcept(concept)

	// Create representation
	representationID := fmt.Sprintf("%s/representation/%s", baseURI, node.ID)
	representation := iso11179.NewRepresentation(
		representationID,
		fmt.Sprintf("Representation for %s", node.Label),
		"Table",
		"table",
	)
	m.registry.RegisterRepresentation(representation)

	// Create data element
	elementID := fmt.Sprintf("%s/data-element/%s", baseURI, node.ID)
	element := iso11179.NewDataElement(
		elementID,
		node.Label,
		conceptID,
		representationID,
		fmt.Sprintf("Data element for table %s", node.Label),
	)
	element.SetSource("Neo4j Knowledge Graph")
	element.AddMetadata("graph_node_id", node.ID)
	element.AddMetadata("graph_node_type", node.Type)

	m.registry.RegisterDataElement(element)
	return nil
}

// mapColumnNode maps a column node to ISO 11179.
func (m *GraphMapper) mapColumnNode(node Node, baseURI string) error {
	// Extract table name from properties
	tableName := "unknown"
	if node.Props != nil {
		if tn, ok := node.Props["table_name"].(string); ok {
			tableName = tn
		}
	}

	// Create concept for column
	conceptID := fmt.Sprintf("%s/concept/%s", baseURI, node.ID)
	concept := iso11179.NewDataElementConcept(
		conceptID,
		node.Label,
		tableName,
		node.Label,
		fmt.Sprintf("Column %s in table %s", node.Label, tableName),
	)
	m.registry.RegisterDataElementConcept(concept)

	// Extract column type
	columnType := "string"
	if node.Props != nil {
		if ct, ok := node.Props["type"].(string); ok {
			columnType = ct
		}
	}

	// Create representation
	representationID := fmt.Sprintf("%s/representation/%s", baseURI, node.ID)
	representation := iso11179.NewRepresentation(
		representationID,
		fmt.Sprintf("Representation for %s", node.Label),
		columnType,
		columnType,
	)
	m.registry.RegisterRepresentation(representation)

	// Create data element
	elementID := fmt.Sprintf("%s/data-element/%s", baseURI, node.ID)
	element := iso11179.NewDataElement(
		elementID,
		node.Label,
		conceptID,
		representationID,
		fmt.Sprintf("Data element for column %s in table %s", node.Label, tableName),
	)
	element.SetSource("Neo4j Knowledge Graph")
	element.AddMetadata("graph_node_id", node.ID)
	element.AddMetadata("graph_node_type", node.Type)

	m.registry.RegisterDataElement(element)
	return nil
}

// mapSQLNode maps an SQL query node to ISO 11179.
func (m *GraphMapper) mapSQLNode(node Node, baseURI string) error {
	// Create concept for SQL query
	conceptID := fmt.Sprintf("%s/concept/%s", baseURI, node.ID)
	concept := iso11179.NewDataElementConcept(
		conceptID,
		node.Label,
		"SQL",
		"Query",
		fmt.Sprintf("SQL query: %s", node.Label),
	)
	m.registry.RegisterDataElementConcept(concept)

	// Create representation
	representationID := fmt.Sprintf("%s/representation/%s", baseURI, node.ID)
	representation := iso11179.NewRepresentation(
		representationID,
		fmt.Sprintf("Representation for %s", node.Label),
		"SQL",
		"sql",
	)
	m.registry.RegisterRepresentation(representation)

	// Create data element
	elementID := fmt.Sprintf("%s/data-element/%s", baseURI, node.ID)
	element := iso11179.NewDataElement(
		elementID,
		node.Label,
		conceptID,
		representationID,
		fmt.Sprintf("Data element for SQL query %s", node.Label),
	)
	element.SetSource("Neo4j Knowledge Graph")
	element.AddMetadata("graph_node_id", node.ID)
	element.AddMetadata("graph_node_type", node.Type)

	m.registry.RegisterDataElement(element)
	return nil
}

// mapGenericNode maps a generic node to ISO 11179.
func (m *GraphMapper) mapGenericNode(node Node, baseURI string) error {
	// Create concept
	conceptID := fmt.Sprintf("%s/concept/%s", baseURI, node.ID)
	concept := iso11179.NewDataElementConcept(
		conceptID,
		node.Label,
		node.Type,
		"Entity",
		fmt.Sprintf("Generic entity: %s", node.Label),
	)
	m.registry.RegisterDataElementConcept(concept)

	// Create representation
	representationID := fmt.Sprintf("%s/representation/%s", baseURI, node.ID)
	representation := iso11179.NewRepresentation(
		representationID,
		fmt.Sprintf("Representation for %s", node.Label),
		"Generic",
		"generic",
	)
	m.registry.RegisterRepresentation(representation)

	// Create data element
	elementID := fmt.Sprintf("%s/data-element/%s", baseURI, node.ID)
	element := iso11179.NewDataElement(
		elementID,
		node.Label,
		conceptID,
		representationID,
		fmt.Sprintf("Data element for %s", node.Label),
	)
	element.SetSource("Neo4j Knowledge Graph")
	element.AddMetadata("graph_node_id", node.ID)
	element.AddMetadata("graph_node_type", node.Type)

	m.registry.RegisterDataElement(element)
	return nil
}

// mapEdgeToISO11179 maps an edge to ISO 11179 relationships.
func (m *GraphMapper) mapEdgeToISO11179(edge Edge, baseURI string) error {
	// For now, edges are represented as relationships in OWL
	// This is handled during OWL generation
	return nil
}

// Helper functions

func getStringValue(val any) string {
	if str, ok := val.(string); ok {
		return str
	}
	return ""
}

func parsePropsJSON(jsonStr string) map[string]any {
	if jsonStr == "" {
		return nil
	}
	// In production, would use json.Unmarshal
	// For now, return empty map
	return make(map[string]any)
}

