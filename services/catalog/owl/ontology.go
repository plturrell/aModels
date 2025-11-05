package owl

import (
	"fmt"
	"strings"
)

// OWL Ontology Structures
// Based on OWL 2.0 specification for semantic web representation

// Ontology represents an OWL ontology.
type Ontology struct {
	// BaseURI is the base URI for the ontology
	BaseURI string

	// Prefixes are namespace prefixes (e.g., "rdf", "owl", "iso11179")
	Prefixes map[string]string

	// Classes are OWL classes defined in the ontology
	Classes map[string]*Class

	// Properties are OWL properties (object and data)
	Properties map[string]*Property

	// Individuals are OWL individuals
	Individuals map[string]*Individual

	// Axioms are additional OWL axioms
	Axioms []Axiom
}

// Class represents an OWL class.
type Class struct {
	// URI is the full URI of the class
	URI string

	// Label is the human-readable label
	Label string

	// Comment is a description
	Comment string

	// SubClassOf indicates parent classes
	SubClassOf []string

	// EquivalentTo indicates equivalent classes
	EquivalentTo []string

	// DisjointWith indicates disjoint classes
	DisjointWith []string
}

// Property represents an OWL property (object or data property).
type Property struct {
	// URI is the full URI of the property
	URI string

	// Label is the human-readable label
	Label string

	// Comment is a description
	Comment string

	// Type indicates whether this is an object property or data property
	Type PropertyType

	// Domain indicates the domain of the property
	Domain []string

	// Range indicates the range of the property
	Range []string

	// InverseOf indicates inverse properties
	InverseOf []string

	// Functional indicates if this is a functional property
	Functional bool

	// InverseFunctional indicates if this is an inverse functional property
	InverseFunctional bool
}

// PropertyType indicates the type of OWL property.
type PropertyType string

const (
	// ObjectProperty links individuals to individuals
	ObjectProperty PropertyType = "ObjectProperty"

	// DataProperty links individuals to data values
	DataProperty PropertyType = "DataProperty"
)

// Individual represents an OWL individual (instance).
type Individual struct {
	// URI is the full URI of the individual
	URI string

	// Label is the human-readable label
	Label string

	// Comment is a description
	Comment string

	// Types are the classes this individual belongs to
	Types []string

	// Properties are property assertions for this individual
	Properties map[string][]PropertyValue
}

// PropertyValue represents a property value assertion.
type PropertyValue struct {
	// Value is the value (URI for object properties, literal for data properties)
	Value string

	// Type is the literal type (for data properties)
	Type string
}

// Axiom represents an OWL axiom.
type Axiom struct {
	// Type is the type of axiom
	Type string

	// Subject is the subject of the axiom
	Subject string

	// Predicate is the predicate (for triples)
	Predicate string

	// Object is the object (for triples)
	Object string
}

// NewOntology creates a new OWL ontology.
func NewOntology(baseURI string) *Ontology {
	onto := &Ontology{
		BaseURI:     baseURI,
		Prefixes:     make(map[string]string),
		Classes:      make(map[string]*Class),
		Properties:   make(map[string]*Property),
		Individuals:  make(map[string]*Individual),
		Axioms:       []Axiom{},
	}

	// Add standard prefixes
	onto.AddPrefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
	onto.AddPrefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
	onto.AddPrefix("owl", "http://www.w3.org/2002/07/owl#")
	onto.AddPrefix("xsd", "http://www.w3.org/2001/XMLSchema#")
	onto.AddPrefix("dc", "http://purl.org/dc/elements/1.1/")
	onto.AddPrefix("dcterms", "http://purl.org/dc/terms/")
	onto.AddPrefix("skos", "http://www.w3.org/2004/02/skos/core#")
	onto.AddPrefix("iso11179", baseURI)

	return onto
}

// AddPrefix adds a namespace prefix.
func (o *Ontology) AddPrefix(prefix, namespace string) {
	o.Prefixes[prefix] = namespace
}

// AddClass adds a class to the ontology.
func (o *Ontology) AddClass(class *Class) {
	o.Classes[class.URI] = class
}

// AddProperty adds a property to the ontology.
func (o *Ontology) AddProperty(property *Property) {
	o.Properties[property.URI] = property
}

// AddIndividual adds an individual to the ontology.
func (o *Ontology) AddIndividual(individual *Individual) {
	o.Individuals[individual.URI] = individual
}

// AddAxiom adds an axiom to the ontology.
func (o *Ontology) AddAxiom(axiom Axiom) {
	o.Axioms = append(o.Axioms, axiom)
}

// ExpandURI expands a local name to a full URI using the base URI.
func (o *Ontology) ExpandURI(localName string) string {
	if strings.HasPrefix(localName, "http://") || strings.HasPrefix(localName, "https://") {
		return localName
	}
	return fmt.Sprintf("%s%s", strings.TrimSuffix(o.BaseURI, "/"), localName)
}

// GetPrefixedName returns a prefixed name for a URI if a prefix matches.
func (o *Ontology) GetPrefixedName(uri string) string {
	for prefix, namespace := range o.Prefixes {
		if strings.HasPrefix(uri, namespace) {
			localName := strings.TrimPrefix(uri, namespace)
			return fmt.Sprintf("%s:%s", prefix, localName)
		}
	}
	return fmt.Sprintf("<%s>", uri)
}

