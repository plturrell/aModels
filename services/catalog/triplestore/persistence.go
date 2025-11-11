package triplestore

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/catalog/owl"
)

// OWLPersistence provides persistence for OWL ontologies to the triplestore.
type OWLPersistence struct {
	client *TriplestoreClient
	logger *log.Logger
}

// NewOWLPersistence creates a new OWL persistence layer.
func NewOWLPersistence(client *TriplestoreClient, logger *log.Logger) *OWLPersistence {
	return &OWLPersistence{
		client: client,
		logger: logger,
	}
}

// StoreOntology stores an OWL ontology in the triplestore as RDF triples.
func (p *OWLPersistence) StoreOntology(ctx context.Context, ontology *owl.Ontology) error {
	var triples []Triple

	// Store classes
	for _, class := range ontology.Classes {
		// Type assertion: class is a Class
		triples = append(triples, Triple{
			Subject:   class.URI,
			Predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
			Object:    "http://www.w3.org/2002/07/owl#Class",
			ObjectType: "uri",
		})

		// Label
		if class.Label != "" {
			triples = append(triples, Triple{
				Subject:   class.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#label",
				Object:    class.Label,
				ObjectType: "literal",
				DataType:  "http://www.w3.org/2001/XMLSchema#string",
			})
		}

		// Comment
		if class.Comment != "" {
			triples = append(triples, Triple{
				Subject:   class.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#comment",
				Object:    class.Comment,
				ObjectType: "literal",
				DataType:  "http://www.w3.org/2001/XMLSchema#string",
			})
		}

		// SubClassOf
		for _, parent := range class.SubClassOf {
			triples = append(triples, Triple{
				Subject:   class.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#subClassOf",
				Object:    parent,
				ObjectType: "uri",
			})
		}
	}

	// Store properties
	for _, property := range ontology.Properties {
		// Type assertion
		if property.Type == "ObjectProperty" {
			triples = append(triples, Triple{
				Subject:   property.URI,
				Predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
				Object:    "http://www.w3.org/2002/07/owl#ObjectProperty",
				ObjectType: "uri",
			})
		} else if property.Type == "DataProperty" {
			triples = append(triples, Triple{
				Subject:   property.URI,
				Predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
				Object:    "http://www.w3.org/2002/07/owl#DatatypeProperty",
				ObjectType: "uri",
			})
		}

		// Label
		if property.Label != "" {
			triples = append(triples, Triple{
				Subject:   property.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#label",
				Object:    property.Label,
				ObjectType: "literal",
				DataType:  "http://www.w3.org/2001/XMLSchema#string",
			})
		}

		// Domain
		for _, domain := range property.Domain {
			triples = append(triples, Triple{
				Subject:   property.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#domain",
				Object:    domain,
				ObjectType: "uri",
			})
		}

		// Range
		for _, range_ := range property.Range {
			triples = append(triples, Triple{
				Subject:   property.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#range",
				Object:    range_,
				ObjectType: "uri",
			})
		}
	}

	// Store individuals
	for _, individual := range ontology.Individuals {
		// Type assertions
		for _, type_ := range individual.Types {
			triples = append(triples, Triple{
				Subject:   individual.URI,
				Predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
				Object:    type_,
				ObjectType: "uri",
			})
		}

		// Label
		if individual.Label != "" {
			triples = append(triples, Triple{
				Subject:   individual.URI,
				Predicate: "http://www.w3.org/2000/01/rdf-schema#label",
				Object:    individual.Label,
				ObjectType: "literal",
				DataType:  "http://www.w3.org/2001/XMLSchema#string",
			})
		}

		// Property assertions
		for propURI, values := range individual.Properties {
			for _, value := range values {
				if value.Type != "" {
					// Data property
					triples = append(triples, Triple{
						Subject:   individual.URI,
						Predicate: propURI,
						Object:    value.Value,
						ObjectType: "literal",
						DataType:  value.Type,
					})
				} else {
					// Object property
					triples = append(triples, Triple{
						Subject:   individual.URI,
						Predicate: propURI,
						Object:    value.Value,
						ObjectType: "uri",
					})
				}
			}
		}
	}

	// Store triples in batch
	if err := p.client.StoreTriples(ctx, triples); err != nil {
		return fmt.Errorf("failed to store ontology: %w", err)
	}

	if p.logger != nil {
		p.logger.Printf("Stored ontology with %d triples", len(triples))
	}

	return nil
}

// LoadOntology loads an OWL ontology from the triplestore.
// This is a simplified implementation - in production, use a proper RDF parser.
func (p *OWLPersistence) LoadOntology(ctx context.Context, baseURI string) (*owl.Ontology, error) {
	ontology := owl.NewOntology(baseURI)

	// Query for all triples with the base URI
	// This is a simplified approach - in production, use proper SPARQL queries
	
	// For now, return an empty ontology
	// Full implementation would query the triplestore and reconstruct the ontology
	
	return ontology, nil
}

