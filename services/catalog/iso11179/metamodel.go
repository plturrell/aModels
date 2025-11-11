package iso11179

import (
	"time"
)

// ISO 11179 Core Metamodel Structures
// Based on ISO/IEC 11179 standard for metadata registries

// DataElementConcept represents the "what" of the data - the semantic meaning.
// It combines an ObjectClass (the thing being described) with a Property (the attribute).
type DataElementConcept struct {
	// Identifier is a unique identifier for the concept (URI)
	Identifier string

	// Name is the human-readable name
	Name string

	// ObjectClass is the class of objects being described (e.g., "Person", "Order")
	ObjectClass string

	// Property is the attribute being described (e.g., "Age", "Amount")
	Property string

	// Definition is a textual definition of the concept
	Definition string

	// Examples are example values for this concept
	Examples []string

	// Version is the version of this concept
	Version string

	// CreatedAt is when this concept was created
	CreatedAt time.Time

	// UpdatedAt is when this concept was last updated
	UpdatedAt time.Time

	// RegistrationStatus indicates the status (e.g., "Submitted", "Recorded", "Qualified")
	RegistrationStatus string
}

// Representation represents the "how" of the data - the format and constraints.
type Representation struct {
	// Identifier is a unique identifier for the representation (URI)
	Identifier string

	// Name is the human-readable name
	Name string

	// Format is the data format (e.g., "Integer", "String", "Date")
	Format string

	// DataType is the specific data type (e.g., "int", "varchar", "timestamp")
	DataType string

	// Length is the maximum length (for strings)
	Length *int

	// Precision is the precision (for decimals)
	Precision *int

	// Scale is the scale (for decimals)
	Scale *int

	// ValueDomainID references the Value Domain
	ValueDomainID string

	// Version is the version of this representation
	Version string

	// CreatedAt is when this representation was created
	CreatedAt time.Time

	// UpdatedAt is when this representation was last updated
	UpdatedAt time.Time
}

// ValueDomain represents the set of permissible values for a data element.
type ValueDomain struct {
	// Identifier is a unique identifier for the value domain (URI)
	Identifier string

	// Name is the human-readable name
	Name string

	// Type indicates whether this is an enumerated or non-enumerated domain
	Type ValueDomainType

	// PermissibleValues are the allowed values (for enumerated domains)
	PermissibleValues []PermissibleValue

	// Constraints are additional constraints (e.g., ranges, patterns)
	Constraints []Constraint

	// Definition is a textual definition of the value domain
	Definition string

	// Version is the version of this value domain
	Version string

	// CreatedAt is when this value domain was created
	CreatedAt time.Time

	// UpdatedAt is when this value domain was last updated
	UpdatedAt time.Time
}

// ValueDomainType indicates the type of value domain.
type ValueDomainType string

const (
	// EnumeratedValueDomain has a fixed set of values
	EnumeratedValueDomain ValueDomainType = "enumerated"

	// NonEnumeratedValueDomain has constraints but no fixed set
	NonEnumeratedValueDomain ValueDomainType = "non_enumerated"
)

// PermissibleValue represents a single allowed value in an enumerated domain.
type PermissibleValue struct {
	// Value is the actual value
	Value string

	// Meaning is the semantic meaning of this value
	Meaning string

	// Order is the ordering of this value (if applicable)
	Order int
}

// Constraint represents a constraint on a value domain.
type Constraint struct {
	// Type is the type of constraint (e.g., "range", "pattern", "min", "max")
	Type string

	// Value is the constraint value
	Value string

	// Description is a description of the constraint
	Description string
}

// DataElement is the core ISO 11179 structure - a combination of a DataElementConcept
// (the "what") and a Representation (the "how").
type DataElement struct {
	// Identifier is a unique identifier for the data element (URI)
	Identifier string

	// Name is the human-readable name
	Name string

	// DataElementConceptID references the concept (the "what")
	DataElementConceptID string

	// RepresentationID references the representation (the "how")
	RepresentationID string

	// Definition is a textual definition of the data element
	Definition string

	// Version is the version of this data element
	Version string

	// CreatedAt is when this data element was created
	CreatedAt time.Time

	// UpdatedAt is when this data element was last updated
	UpdatedAt time.Time

	// RegistrationStatus indicates the status
	RegistrationStatus string

	// Steward is the person or organization responsible for this data element
	Steward string

	// Source is where this data element came from (e.g., "Extract Service", "Glean Catalog")
	Source string

	// Metadata is additional metadata as key-value pairs
	Metadata map[string]any
}

// MetadataRegistry is the central registry that manages all ISO 11179 metadata.
type MetadataRegistry struct {
	// Identifier is a unique identifier for the registry
	Identifier string

	// Name is the human-readable name
	Name string

	// DataElements are all registered data elements
	DataElements map[string]*DataElement

	// DataElementConcepts are all registered concepts
	DataElementConcepts map[string]*DataElementConcept

	// Representations are all registered representations
	Representations map[string]*Representation

	// ValueDomains are all registered value domains
	ValueDomains map[string]*ValueDomain

	// Namespace is the URI namespace for this registry
	Namespace string

	// CreatedAt is when this registry was created
	CreatedAt time.Time

	// UpdatedAt is when this registry was last updated
	UpdatedAt time.Time
}

// NewMetadataRegistry creates a new metadata registry.
func NewMetadataRegistry(identifier, name, namespace string) *MetadataRegistry {
	return &MetadataRegistry{
		Identifier:          identifier,
		Name:                name,
		Namespace:           namespace,
		DataElements:         make(map[string]*DataElement),
		DataElementConcepts: make(map[string]*DataElementConcept),
		Representations:     make(map[string]*Representation),
		ValueDomains:        make(map[string]*ValueDomain),
		CreatedAt:           time.Now().UTC(),
		UpdatedAt:           time.Now().UTC(),
	}
}

// RegisterDataElement registers a new data element in the registry.
func (r *MetadataRegistry) RegisterDataElement(element *DataElement) {
	r.DataElements[element.Identifier] = element
	r.UpdatedAt = time.Now().UTC()
}

// RegisterDataElementConcept registers a new concept in the registry.
func (r *MetadataRegistry) RegisterDataElementConcept(concept *DataElementConcept) {
	r.DataElementConcepts[concept.Identifier] = concept
	r.UpdatedAt = time.Now().UTC()
}

// RegisterRepresentation registers a new representation in the registry.
func (r *MetadataRegistry) RegisterRepresentation(representation *Representation) {
	r.Representations[representation.Identifier] = representation
	r.UpdatedAt = time.Now().UTC()
}

// RegisterValueDomain registers a new value domain in the registry.
func (r *MetadataRegistry) RegisterValueDomain(domain *ValueDomain) {
	r.ValueDomains[domain.Identifier] = domain
	r.UpdatedAt = time.Now().UTC()
}

// GetDataElement retrieves a data element by identifier.
func (r *MetadataRegistry) GetDataElement(identifier string) (*DataElement, bool) {
	element, ok := r.DataElements[identifier]
	return element, ok
}

// GetDataElementConcept retrieves a concept by identifier.
func (r *MetadataRegistry) GetDataElementConcept(identifier string) (*DataElementConcept, bool) {
	concept, ok := r.DataElementConcepts[identifier]
	return concept, ok
}

// GetRepresentation retrieves a representation by identifier.
func (r *MetadataRegistry) GetRepresentation(identifier string) (*Representation, bool) {
	representation, ok := r.Representations[identifier]
	return representation, ok
}

// GetValueDomain retrieves a value domain by identifier.
func (r *MetadataRegistry) GetValueDomain(identifier string) (*ValueDomain, bool) {
	domain, ok := r.ValueDomains[identifier]
	return domain, ok
}

