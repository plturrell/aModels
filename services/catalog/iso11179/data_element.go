package iso11179

import (
	"fmt"
	"time"
)

// NewDataElementConcept creates a new data element concept.
func NewDataElementConcept(identifier, name, objectClass, property, definition string) *DataElementConcept {
	now := time.Now().UTC()
	return &DataElementConcept{
		Identifier:         identifier,
		Name:               name,
		ObjectClass:        objectClass,
		Property:           property,
		Definition:         definition,
		Examples:           []string{},
		Version:            "1.0",
		CreatedAt:          now,
		UpdatedAt:          now,
		RegistrationStatus: "Submitted",
	}
}

// AddExample adds an example value to the concept.
func (dec *DataElementConcept) AddExample(example string) {
	dec.Examples = append(dec.Examples, example)
	dec.UpdatedAt = time.Now().UTC()
}

// FullName returns the full name combining object class and property.
func (dec *DataElementConcept) FullName() string {
	if dec.Property != "" {
		return fmt.Sprintf("%s - %s", dec.ObjectClass, dec.Property)
	}
	return dec.ObjectClass
}

// NewRepresentation creates a new representation.
func NewRepresentation(identifier, name, format, dataType string) *Representation {
	now := time.Now().UTC()
	return &Representation{
		Identifier: identifier,
		Name:       name,
		Format:     format,
		DataType:   dataType,
		Version:    "1.0",
		CreatedAt:  now,
		UpdatedAt:  now,
	}
}

// SetLength sets the length constraint for the representation.
func (r *Representation) SetLength(length int) {
	r.Length = &length
	r.UpdatedAt = time.Now().UTC()
}

// SetPrecisionScale sets the precision and scale for decimal representations.
func (r *Representation) SetPrecisionScale(precision, scale int) {
	r.Precision = &precision
	r.Scale = &scale
	r.UpdatedAt = time.Now().UTC()
}

// NewValueDomain creates a new value domain.
func NewValueDomain(identifier, name string, domainType ValueDomainType) *ValueDomain {
	now := time.Now().UTC()
	return &ValueDomain{
		Identifier:         identifier,
		Name:               name,
		Type:               domainType,
		PermissibleValues:  []PermissibleValue{},
		Constraints:        []Constraint{},
		Version:            "1.0",
		CreatedAt:          now,
		UpdatedAt:          now,
	}
}

// AddPermissibleValue adds a permissible value to an enumerated domain.
func (vd *ValueDomain) AddPermissibleValue(value, meaning string, order int) {
	if vd.Type != EnumeratedValueDomain {
		vd.Type = EnumeratedValueDomain
	}
	vd.PermissibleValues = append(vd.PermissibleValues, PermissibleValue{
		Value:   value,
		Meaning: meaning,
		Order:   order,
	})
	vd.UpdatedAt = time.Now().UTC()
}

// AddConstraint adds a constraint to the value domain.
func (vd *ValueDomain) AddConstraint(constraintType, value, description string) {
	vd.Constraints = append(vd.Constraints, Constraint{
		Type:        constraintType,
		Value:       value,
		Description: description,
	})
	vd.UpdatedAt = time.Now().UTC()
}

// NewDataElement creates a new data element combining a concept and representation.
func NewDataElement(identifier, name, conceptID, representationID, definition string) *DataElement {
	now := time.Now().UTC()
	return &DataElement{
		Identifier:         identifier,
		Name:               name,
		DataElementConceptID: conceptID,
		RepresentationID:   representationID,
		Definition:         definition,
		Version:            "1.0",
		CreatedAt:         now,
		UpdatedAt:          now,
		RegistrationStatus: "Submitted",
		Metadata:           make(map[string]any),
	}
}

// SetSteward sets the steward for the data element.
func (de *DataElement) SetSteward(steward string) {
	de.Steward = steward
	de.UpdatedAt = time.Now().UTC()
}

// SetSource sets the source for the data element.
func (de *DataElement) SetSource(source string) {
	de.Source = source
	de.UpdatedAt = time.Now().UTC()
}

// AddMetadata adds a metadata key-value pair.
func (de *DataElement) AddMetadata(key string, value any) {
	if de.Metadata == nil {
		de.Metadata = make(map[string]any)
	}
	de.Metadata[key] = value
	de.UpdatedAt = time.Now().UTC()
}

// GetFullDefinition returns the full definition including concept and representation details.
func (de *DataElement) GetFullDefinition(registry *MetadataRegistry) string {
	var parts []string
	
	parts = append(parts, de.Definition)
	
	if concept, ok := registry.GetDataElementConcept(de.DataElementConceptID); ok {
		parts = append(parts, fmt.Sprintf("Concept: %s", concept.FullName()))
		if concept.Definition != "" {
			parts = append(parts, fmt.Sprintf("Concept Definition: %s", concept.Definition))
		}
	}
	
	if representation, ok := registry.GetRepresentation(de.RepresentationID); ok {
		parts = append(parts, fmt.Sprintf("Format: %s (%s)", representation.Format, representation.DataType))
		if representation.Length != nil {
			parts = append(parts, fmt.Sprintf("Length: %d", *representation.Length))
		}
		if representation.Precision != nil && representation.Scale != nil {
			parts = append(parts, fmt.Sprintf("Precision: %d, Scale: %d", *representation.Precision, *representation.Scale))
		}
		if representation.ValueDomainID != "" {
			if domain, ok := registry.GetValueDomain(representation.ValueDomainID); ok {
				parts = append(parts, fmt.Sprintf("Value Domain: %s", domain.Name))
				if domain.Definition != "" {
					parts = append(parts, fmt.Sprintf("Domain Definition: %s", domain.Definition))
				}
			}
		}
	}
	
	result := ""
	for i, part := range parts {
		if i > 0 {
			result += "\n"
		}
		result += part
	}
	return result
}

