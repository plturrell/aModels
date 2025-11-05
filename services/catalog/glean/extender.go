package glean

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/owl"
)

// GleanExtender extends Glean Catalog with OWL semantics.
type GleanExtender struct {
	registry *iso11179.MetadataRegistry
	logger   *log.Logger
}

// NewGleanExtender creates a new Glean extender.
func NewGleanExtender(registry *iso11179.MetadataRegistry, logger *log.Logger) *GleanExtender {
	return &GleanExtender{
		registry: registry,
		logger:   logger,
	}
}

// GleanFact represents a fact from Glean Catalog.
type GleanFact struct {
	Predicate string
	Key       map[string]any
	Value     any
}

// ExtendGleanWithOWL extends Glean facts with OWL semantics by converting them to ISO 11179 and then to OWL.
func (e *GleanExtender) ExtendGleanWithOWL(ctx context.Context, facts []GleanFact, baseURI string) (*owl.Ontology, error) {
	// Convert Glean facts to ISO 11179 data elements
	for _, fact := range facts {
		dataElement, err := e.convertGleanFactToDataElement(fact)
		if err != nil {
			if e.logger != nil {
				e.logger.Printf("Warning: Failed to convert Glean fact to data element: %v", err)
			}
			continue
		}
		if dataElement != nil {
			e.registry.RegisterDataElement(dataElement)
		}
	}

	// Generate OWL ontology from ISO 11179 registry
	ontology := owl.GenerateOWLFromISO11179(e.registry, baseURI)

	return ontology, nil
}

// convertGleanFactToDataElement converts a Glean fact to an ISO 11179 data element.
func (e *GleanExtender) convertGleanFactToDataElement(fact GleanFact) (*iso11179.DataElement, error) {
	// Extract information from Glean fact
	// Glean facts have a predicate (e.g., "agenticAiETH.ETL.Node.1") and key-value pairs
	
	// Try to extract node information
	var nodeID, nodeType, nodeLabel string
	var props map[string]any

	if key, ok := fact.Key.(map[string]any); ok {
		if id, ok := key["id"].(string); ok {
			nodeID = id
		}
		if kind, ok := key["kind"].(string); ok {
			nodeType = kind
		}
		if label, ok := key["label"].(string); ok {
			nodeLabel = label
		}
		if propsJSON, ok := key["properties_json"].(string); ok {
			// Parse properties JSON (would need JSON unmarshaling)
			// For now, skip
		}
	}

	// If this is a table node, create a data element concept
	if nodeType == "table" {
		// Create concept for table
		conceptID := fmt.Sprintf("%s/concept/%s", e.registry.Namespace, nodeID)
		concept := iso11179.NewDataElementConcept(
			conceptID,
			nodeLabel,
			"Table", // Object class
			"Schema", // Property
			fmt.Sprintf("Table schema for %s", nodeLabel),
		)
		e.registry.RegisterDataElementConcept(concept)

		// Create representation
		representationID := fmt.Sprintf("%s/representation/%s", e.registry.Namespace, nodeID)
		representation := iso11179.NewRepresentation(
			representationID,
			fmt.Sprintf("Representation for %s", nodeLabel),
			"Table",
			"table",
		)
		e.registry.RegisterRepresentation(representation)

		// Create data element
		elementID := fmt.Sprintf("%s/data-element/%s", e.registry.Namespace, nodeID)
		element := iso11179.NewDataElement(
			elementID,
			nodeLabel,
			conceptID,
			representationID,
			fmt.Sprintf("Data element for table %s", nodeLabel),
		)
		element.SetSource("Glean Catalog")
		element.AddMetadata("glean_predicate", fact.Predicate)
		element.AddMetadata("glean_node_id", nodeID)

		return element, nil
	}

	// If this is a column node, create a data element
	if nodeType == "column" {
		// Extract table name from properties or context
		tableName := "unknown"
		if props != nil {
			if tn, ok := props["table_name"].(string); ok {
				tableName = tn
			}
		}

		// Create concept for column
		conceptID := fmt.Sprintf("%s/concept/%s", e.registry.Namespace, nodeID)
		concept := iso11179.NewDataElementConcept(
			conceptID,
			nodeLabel,
			tableName, // Object class (table)
			nodeLabel, // Property (column name)
			fmt.Sprintf("Column %s in table %s", nodeLabel, tableName),
		)
		e.registry.RegisterDataElementConcept(concept)

		// Extract column type from properties
		columnType := "string"
		if props != nil {
			if ct, ok := props["type"].(string); ok {
				columnType = ct
			}
		}

		// Create representation
		representationID := fmt.Sprintf("%s/representation/%s", e.registry.Namespace, nodeID)
		representation := iso11179.NewRepresentation(
			representationID,
			fmt.Sprintf("Representation for %s", nodeLabel),
			columnType,
			columnType,
		)
		e.registry.RegisterRepresentation(representation)

		// Create data element
		elementID := fmt.Sprintf("%s/data-element/%s", e.registry.Namespace, nodeID)
		element := iso11179.NewDataElement(
			elementID,
			nodeLabel,
			conceptID,
			representationID,
			fmt.Sprintf("Data element for column %s in table %s", nodeLabel, tableName),
		)
		element.SetSource("Glean Catalog")
		element.AddMetadata("glean_predicate", fact.Predicate)
		element.AddMetadata("glean_node_id", nodeID)

		return element, nil
	}

	return nil, nil // Not a mappable node type
}

