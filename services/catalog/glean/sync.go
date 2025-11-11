package glean

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/triplestore"
)

// GleanSync provides bidirectional synchronization between Glean Catalog and the semantic catalog.
type GleanSync struct {
	registry     *iso11179.MetadataRegistry
	triplestore  *triplestore.TriplestoreClient
	extender     *GleanExtender
	mapper       *GleanMapper
	logger       *log.Logger
}

// NewGleanSync creates a new Glean synchronizer.
func NewGleanSync(
	registry *iso11179.MetadataRegistry,
	triplestore *triplestore.TriplestoreClient,
	logger *log.Logger,
) *GleanSync {
	return &GleanSync{
		registry:    registry,
		triplestore: triplestore,
		extender:    NewGleanExtender(registry, logger),
		mapper:      NewGleanMapper(registry, logger),
		logger:      logger,
	}
}

// SyncFromGlean synchronizes data from Glean Catalog to the semantic catalog.
func (s *GleanSync) SyncFromGlean(ctx context.Context, facts []GleanFact, baseURI string) error {
	// Convert Glean facts to ISO 11179
	for _, fact := range facts {
		dataElement, err := s.extender.convertGleanFactToDataElement(fact)
		if err != nil {
			if s.logger != nil {
				s.logger.Printf("Warning: Failed to convert Glean fact: %v", err)
			}
			continue
		}
		if dataElement != nil {
			s.registry.RegisterDataElement(dataElement)
		}

		// Also convert to RDF triples
		triples := s.mapper.MapGleanFactToTriples(fact, baseURI)
		if len(triples) > 0 {
			if err := s.triplestore.StoreTriples(ctx, triples); err != nil {
				if s.logger != nil {
					s.logger.Printf("Warning: Failed to store triples: %v", err)
				}
			}
		}
	}

	if s.logger != nil {
		s.logger.Printf("Synchronized %d Glean facts to semantic catalog", len(facts))
	}

	return nil
}

// SyncToGlean synchronizes data from the semantic catalog to Glean Catalog.
// This creates Glean batch files that can be ingested.
func (s *GleanSync) SyncToGlean(ctx context.Context, baseURI string) ([]GleanFact, error) {
	var facts []GleanFact

	// Convert ISO 11179 data elements to Glean facts
	for _, element := range s.registry.DataElements {
		fact := s.convertDataElementToGleanFact(element, baseURI)
		facts = append(facts, fact)
	}

	if s.logger != nil {
		s.logger.Printf("Synchronized %d data elements to Glean format", len(facts))
	}

	return facts, nil
}

// convertDataElementToGleanFact converts an ISO 11179 data element to a Glean fact.
func (s *GleanSync) convertDataElementToGleanFact(element *iso11179.DataElement, baseURI string) GleanFact {
	// Extract identifier from element URI
	identifier := element.Identifier
	if len(identifier) > 0 {
		// Extract local name from URI
		parts := splitURI(identifier)
		if len(parts) > 0 {
			identifier = parts[len(parts)-1]
		}
	}

	// Create Glean fact key
	key := map[string]any{
		"id":    identifier,
		"kind":  "data_element",
		"label": element.Name,
	}

	// Add metadata
	if element.Metadata != nil {
		for k, v := range element.Metadata {
			key[k] = v
		}
	}

	// Create predicate (Glean format)
	predicate := fmt.Sprintf("agenticAiETH.ETL.DataElement.1")

	return GleanFact{
		Predicate: predicate,
		Key:       key,
		Value:     nil,
	}
}

// splitURI splits a URI into parts.
func splitURI(uri string) []string {
	// Split by "/"
	var parts []string
	current := ""
	for _, char := range uri {
		if char == '/' {
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

// GetSyncStatus returns the synchronization status.
func (s *GleanSync) GetSyncStatus() map[string]any {
	stats := s.registry.GetRegistryStats()
	return map[string]any{
		"data_elements":        stats.DataElementCount,
		"concepts":             stats.DataElementConceptCount,
		"representations":      stats.RepresentationCount,
		"value_domains":        stats.ValueDomainCount,
		"last_updated":         stats.LastUpdated,
	}
}

