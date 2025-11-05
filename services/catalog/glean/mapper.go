package glean

import (
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/triplestore"
)

// GleanMapper maps Glean facts to ISO 11179 and RDF triples.
type GleanMapper struct {
	registry *iso11179.MetadataRegistry
	logger   *log.Logger
}

// NewGleanMapper creates a new Glean mapper.
func NewGleanMapper(registry *iso11179.MetadataRegistry, logger *log.Logger) *GleanMapper {
	return &GleanMapper{
		registry: registry,
		logger:   logger,
	}
}

// MapGleanFactToTriples maps a Glean fact to RDF triples.
func (m *GleanMapper) MapGleanFactToTriples(fact GleanFact, baseURI string) []triplestore.Triple {
	var triples []triplestore.Triple

	// Extract node information
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
	}

	// Create subject URI
	subjectURI := fmt.Sprintf("%s/resource/%s", baseURI, nodeID)

	// Type assertion
	triples = append(triples, triplestore.Triple{
		Subject:   subjectURI,
		Predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
		Object:    fmt.Sprintf("%s#%s", baseURI, nodeType),
		ObjectType: "uri",
	})

	// Label
	if nodeLabel != "" {
		triples = append(triples, triplestore.Triple{
			Subject:   subjectURI,
			Predicate: "http://www.w3.org/2000/01/rdf-schema#label",
			Object:    nodeLabel,
			ObjectType: "literal",
			DataType:  "http://www.w3.org/2001/XMLSchema#string",
		})
	}

	// Map Glean predicate to OWL property
	predicateURI := m.mapGleanPredicateToURI(fact.Predicate, baseURI)
	if predicateURI != "" {
		// Create object URI (if applicable)
		objectURI := fmt.Sprintf("%s/resource/%s", baseURI, nodeID)
		triples = append(triples, triplestore.Triple{
			Subject:   baseURI,
			Predicate: predicateURI,
			Object:    objectURI,
			ObjectType: "uri",
		})
	}

	// Map properties to RDF triples
	if props != nil {
		for key, value := range props {
			if strValue, ok := value.(string); ok {
				triples = append(triples, triplestore.Triple{
					Subject:   subjectURI,
					Predicate: fmt.Sprintf("%s#%s", baseURI, key),
					Object:    strValue,
					ObjectType: "literal",
					DataType:  "http://www.w3.org/2001/XMLSchema#string",
				})
			}
		}
	}

	return triples
}

// mapGleanPredicateToURI maps a Glean predicate to an OWL property URI.
func (m *GleanMapper) mapGleanPredicateToURI(gleanPredicate, baseURI string) string {
	// Glean predicates are like "agenticAiETH.ETL.Node.1"
	// Map to OWL properties
	
	// Extract the type from the predicate
	// For now, create a simple mapping
	if len(gleanPredicate) > 0 {
		// Extract the last part (e.g., "Node" from "agenticAiETH.ETL.Node.1")
		parts := splitGleanPredicate(gleanPredicate)
		if len(parts) > 0 {
			lastPart := parts[len(parts)-2] // Second to last (before version number)
			return fmt.Sprintf("%s#has%s", baseURI, lastPart)
		}
	}
	
	return ""
}

// splitGleanPredicate splits a Glean predicate into parts.
func splitGleanPredicate(predicate string) []string {
	// Split by "."
	var parts []string
	current := ""
	for _, char := range predicate {
		if char == '.' {
			if current != "" {
				parts = append(parts, current)
				current = ""
			}
		} else {
			current += string(char)
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	return parts
}

// MapGleanEdgeToTriples maps a Glean edge (relationship) to RDF triples.
func (m *GleanMapper) MapGleanEdgeToTriples(sourceID, targetID, label string, baseURI string) []triplestore.Triple {
	var triples []triplestore.Triple

	sourceURI := fmt.Sprintf("%s/resource/%s", baseURI, sourceID)
	targetURI := fmt.Sprintf("%s/resource/%s", baseURI, targetID)

	// Create relationship triple
	predicateURI := fmt.Sprintf("%s#%s", baseURI, label)
	triples = append(triples, triplestore.Triple{
		Subject:   sourceURI,
		Predicate: predicateURI,
		Object:    targetURI,
		ObjectType: "uri",
	})

	return triples
}

