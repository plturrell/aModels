package owl

import (
	"fmt"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

// GenerateOWLFromISO11179 generates an OWL ontology from an ISO 11179 metadata registry.
func GenerateOWLFromISO11179(registry *iso11179.MetadataRegistry, baseURI string) *Ontology {
	ontology := NewOntology(baseURI)

	// Generate ISO 11179 classes
	generateISO11179Classes(ontology)

	// Generate data element concepts as classes
	for _, concept := range registry.DataElementConcepts {
		generateConceptClass(ontology, concept)
	}

	// Generate data elements as individuals
	for _, element := range registry.DataElements {
		generateDataElementIndividual(ontology, element, registry)
	}

	// Generate representations as classes
	for _, representation := range registry.Representations {
		generateRepresentationClass(ontology, representation)
	}

	// Generate value domains as classes
	for _, domain := range registry.ValueDomains {
		generateValueDomainClass(ontology, domain)
	}

	// Generate relationships
	generateRelationships(ontology, registry)

	return ontology
}

// generateISO11179Classes generates the core ISO 11179 OWL classes.
func generateISO11179Classes(ontology *Ontology) {
	baseURI := ontology.BaseURI

	// DataElement class
	dataElementClass := &Class{
		URI:    fmt.Sprintf("%s#DataElement", baseURI),
		Label:  "Data Element",
		Comment: "A data element as defined by ISO 11179, combining a Data Element Concept (the 'what') with a Representation (the 'how')",
		SubClassOf: []string{"rdfs:Resource"},
	}
	ontology.AddClass(dataElementClass)

	// DataElementConcept class
	conceptClass := &Class{
		URI:     fmt.Sprintf("%s#DataElementConcept", baseURI),
		Label:   "Data Element Concept",
		Comment: "The semantic meaning of data, combining an Object Class with a Property",
		SubClassOf: []string{"rdfs:Resource"},
	}
	ontology.AddClass(conceptClass)

	// Representation class
	representationClass := &Class{
		URI:     fmt.Sprintf("%s#Representation", baseURI),
		Label:   "Representation",
		Comment: "The format and constraints of data",
		SubClassOf: []string{"rdfs:Resource"},
	}
	ontology.AddClass(representationClass)

	// ValueDomain class
	valueDomainClass := &Class{
		URI:     fmt.Sprintf("%s#ValueDomain", baseURI),
		Label:   "Value Domain",
		Comment: "The set of permissible values for a data element",
		SubClassOf: []string{"rdfs:Resource"},
	}
	ontology.AddClass(valueDomainClass)

	// Generate properties
	hasConceptProp := &Property{
		URI:     fmt.Sprintf("%s#hasDataElementConcept", baseURI),
		Label:   "has Data Element Concept",
		Comment: "Links a Data Element to its Data Element Concept",
		Type:    ObjectProperty,
		Domain:  []string{fmt.Sprintf("%s#DataElement", baseURI)},
		Range:   []string{fmt.Sprintf("%s#DataElementConcept", baseURI)},
	}
	ontology.AddProperty(hasConceptProp)

	hasRepresentationProp := &Property{
		URI:     fmt.Sprintf("%s#hasRepresentation", baseURI),
		Label:   "has Representation",
		Comment: "Links a Data Element to its Representation",
		Type:    ObjectProperty,
		Domain:  []string{fmt.Sprintf("%s#DataElement", baseURI)},
		Range:   []string{fmt.Sprintf("%s#Representation", baseURI)},
	}
	ontology.AddProperty(hasRepresentationProp)

	hasValueDomainProp := &Property{
		URI:     fmt.Sprintf("%s#hasValueDomain", baseURI),
		Label:   "has Value Domain",
		Comment: "Links a Representation to its Value Domain",
		Type:    ObjectProperty,
		Domain:  []string{fmt.Sprintf("%s#Representation", baseURI)},
		Range:   []string{fmt.Sprintf("%s#ValueDomain", baseURI)},
	}
	ontology.AddProperty(hasValueDomainProp)

	objectClassProp := &Property{
		URI:     fmt.Sprintf("%s#objectClass", baseURI),
		Label:   "object class",
		Comment: "The object class of a Data Element Concept",
		Type:    DataProperty,
		Domain:  []string{fmt.Sprintf("%s#DataElementConcept", baseURI)},
		Range:   []string{"xsd:string"},
	}
	ontology.AddProperty(objectClassProp)

	propertyProp := &Property{
		URI:     fmt.Sprintf("%s#property", baseURI),
		Label:   "property",
		Comment: "The property of a Data Element Concept",
		Type:    DataProperty,
		Domain:  []string{fmt.Sprintf("%s#DataElementConcept", baseURI)},
		Range:   []string{"xsd:string"},
	}
	ontology.AddProperty(propertyProp)

	dataTypeProp := &Property{
		URI:     fmt.Sprintf("%s#dataType", baseURI),
		Label:   "data type",
		Comment: "The data type of a Representation",
		Type:    DataProperty,
		Domain:  []string{fmt.Sprintf("%s#Representation", baseURI)},
		Range:   []string{"xsd:string"},
	}
	ontology.AddProperty(dataTypeProp)
}

// generateConceptClass generates an OWL class for a data element concept.
func generateConceptClass(ontology *Ontology, concept *iso11179.DataElementConcept) {
	classURI := ontology.ExpandURI(concept.Identifier)
	class := &Class{
		URI:     classURI,
		Label:   concept.Name,
		Comment: concept.Definition,
		SubClassOf: []string{fmt.Sprintf("%s#DataElementConcept", ontology.BaseURI)},
	}
	ontology.AddClass(class)

	// Create individual for the concept
	individual := &Individual{
		URI:     classURI,
		Label:   concept.Name,
		Comment: concept.Definition,
		Types:   []string{fmt.Sprintf("%s#DataElementConcept", ontology.BaseURI)},
		Properties: map[string][]PropertyValue{
			fmt.Sprintf("%s#objectClass", ontology.BaseURI): {
				{Value: concept.ObjectClass, Type: "xsd:string"},
			},
			fmt.Sprintf("%s#property", ontology.BaseURI): {
				{Value: concept.Property, Type: "xsd:string"},
			},
			"rdfs:label": {
				{Value: concept.Name, Type: "xsd:string"},
			},
			"rdfs:comment": {
				{Value: concept.Definition, Type: "xsd:string"},
			},
		},
	}
	ontology.AddIndividual(individual)
}

// generateDataElementIndividual generates an OWL individual for a data element.
func generateDataElementIndividual(ontology *Ontology, element *iso11179.DataElement, registry *iso11179.MetadataRegistry) {
	individualURI := ontology.ExpandURI(element.Identifier)
	individual := &Individual{
		URI:     individualURI,
		Label:   element.Name,
		Comment: element.Definition,
		Types:   []string{fmt.Sprintf("%s#DataElement", ontology.BaseURI)},
		Properties: map[string][]PropertyValue{
			"rdfs:label": {
				{Value: element.Name, Type: "xsd:string"},
			},
			"rdfs:comment": {
				{Value: element.Definition, Type: "xsd:string"},
			},
		},
	}

	// Link to concept
	if element.DataElementConceptID != "" {
		conceptURI := ontology.ExpandURI(element.DataElementConceptID)
		individual.Properties[fmt.Sprintf("%s#hasDataElementConcept", ontology.BaseURI)] = []PropertyValue{
			{Value: conceptURI},
		}
	}

	// Link to representation
	if element.RepresentationID != "" {
		representationURI := ontology.ExpandURI(element.RepresentationID)
		individual.Properties[fmt.Sprintf("%s#hasRepresentation", ontology.BaseURI)] = []PropertyValue{
			{Value: representationURI},
		}
	}

	// Add source metadata
	if element.Source != "" {
		individual.Properties["dc:source"] = []PropertyValue{
			{Value: element.Source, Type: "xsd:string"},
		}
	}

	ontology.AddIndividual(individual)
}

// generateRepresentationClass generates an OWL class for a representation.
func generateRepresentationClass(ontology *Ontology, representation *iso11179.Representation) {
	classURI := ontology.ExpandURI(representation.Identifier)
	class := &Class{
		URI:     classURI,
		Label:   representation.Name,
		Comment: fmt.Sprintf("Representation: %s (%s)", representation.Format, representation.DataType),
		SubClassOf: []string{fmt.Sprintf("%s#Representation", ontology.BaseURI)},
	}
	ontology.AddClass(class)

	// Create individual for the representation
	individual := &Individual{
		URI:     classURI,
		Label:   representation.Name,
		Comment: fmt.Sprintf("Representation: %s (%s)", representation.Format, representation.DataType),
		Types:   []string{fmt.Sprintf("%s#Representation", ontology.BaseURI)},
		Properties: map[string][]PropertyValue{
			fmt.Sprintf("%s#dataType", ontology.BaseURI): {
				{Value: representation.DataType, Type: "xsd:string"},
			},
			"rdfs:label": {
				{Value: representation.Name, Type: "xsd:string"},
			},
		},
	}

	// Link to value domain if present
	if representation.ValueDomainID != "" {
		domainURI := ontology.ExpandURI(representation.ValueDomainID)
		individual.Properties[fmt.Sprintf("%s#hasValueDomain", ontology.BaseURI)] = []PropertyValue{
			{Value: domainURI},
		}
	}

	ontology.AddIndividual(individual)
}

// generateValueDomainClass generates an OWL class for a value domain.
func generateValueDomainClass(ontology *Ontology, domain *iso11179.ValueDomain) {
	classURI := ontology.ExpandURI(domain.Identifier)
	class := &Class{
		URI:     classURI,
		Label:   domain.Name,
		Comment: domain.Definition,
		SubClassOf: []string{fmt.Sprintf("%s#ValueDomain", ontology.BaseURI)},
	}
	ontology.AddClass(class)

	// Create individual for the value domain
	individual := &Individual{
		URI:     classURI,
		Label:   domain.Name,
		Comment: domain.Definition,
		Types:   []string{fmt.Sprintf("%s#ValueDomain", ontology.BaseURI)},
		Properties: map[string][]PropertyValue{
			"rdfs:label": {
				{Value: domain.Name, Type: "xsd:string"},
			},
			"rdfs:comment": {
				{Value: domain.Definition, Type: "xsd:string"},
			},
		},
	}

	// Add permissible values as properties (for enumerated domains)
	if domain.Type == iso11179.EnumeratedValueDomain {
		for _, pv := range domain.PermissibleValues {
			individual.Properties[fmt.Sprintf("%s#hasPermissibleValue", ontology.BaseURI)] = append(
				individual.Properties[fmt.Sprintf("%s#hasPermissibleValue", ontology.BaseURI)],
				PropertyValue{Value: pv.Value, Type: "xsd:string"},
			)
		}
	}

	ontology.AddIndividual(individual)
}

// generateRelationships generates OWL relationships between ISO 11179 entities.
func generateRelationships(ontology *Ontology, registry *iso11179.MetadataRegistry) {
	// Relationships are already established through property assertions in individuals
	// Additional axioms can be added here if needed
}

