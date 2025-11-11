package iso11179

import (
	"fmt"
	"strings"
	"time"
)

// Registry provides additional functionality for metadata registry management.

// GenerateURI generates a URI for a given identifier using the registry namespace.
func (r *MetadataRegistry) GenerateURI(localName string) string {
	namespace := strings.TrimSuffix(r.Namespace, "/")
	if !strings.HasSuffix(namespace, "/") && !strings.HasSuffix(namespace, "#") {
		namespace += "/"
	}
	return fmt.Sprintf("%s%s", namespace, localName)
}

// FindDataElementsByConcept finds all data elements that use a specific concept.
func (r *MetadataRegistry) FindDataElementsByConcept(conceptID string) []*DataElement {
	var result []*DataElement
	for _, element := range r.DataElements {
		if element.DataElementConceptID == conceptID {
			result = append(result, element)
		}
	}
	return result
}

// FindDataElementsByRepresentation finds all data elements that use a specific representation.
func (r *MetadataRegistry) FindDataElementsByRepresentation(representationID string) []*DataElement {
	var result []*DataElement
	for _, element := range r.DataElements {
		if element.RepresentationID == representationID {
			result = append(result, element)
		}
	}
	return result
}

// FindDataElementsByObjectClass finds all data elements whose concept has a specific object class.
func (r *MetadataRegistry) FindDataElementsByObjectClass(objectClass string) []*DataElement {
	var result []*DataElement
	for _, element := range r.DataElements {
		concept, ok := r.DataElementConcepts[element.DataElementConceptID]
		if ok && concept.ObjectClass == objectClass {
			result = append(result, element)
		}
	}
	return result
}

// FindDataElementsByProperty finds all data elements whose concept has a specific property.
func (r *MetadataRegistry) FindDataElementsByProperty(property string) []*DataElement {
	var result []*DataElement
	for _, element := range r.DataElements {
		concept, ok := r.DataElementConcepts[element.DataElementConceptID]
		if ok && concept.Property == property {
			result = append(result, element)
		}
	}
	return result
}

// FindDataElementsBySource finds all data elements from a specific source.
func (r *MetadataRegistry) FindDataElementsBySource(source string) []*DataElement {
	var result []*DataElement
	for _, element := range r.DataElements {
		if element.Source == source {
			result = append(result, element)
		}
	}
	return result
}

// GetRegistryStats returns statistics about the registry.
func (r *MetadataRegistry) GetRegistryStats() RegistryStats {
	return RegistryStats{
		DataElementCount:         len(r.DataElements),
		DataElementConceptCount:  len(r.DataElementConcepts),
		RepresentationCount:      len(r.Representations),
		ValueDomainCount:         len(r.ValueDomains),
		LastUpdated:              r.UpdatedAt,
	}
}

// RegistryStats contains statistics about a metadata registry.
type RegistryStats struct {
	DataElementCount        int
	DataElementConceptCount  int
	RepresentationCount     int
	ValueDomainCount        int
	LastUpdated             time.Time
}

// ValidateDataElement validates that a data element references valid concepts and representations.
func (r *MetadataRegistry) ValidateDataElement(element *DataElement) []string {
	var errors []string

	if element.DataElementConceptID == "" {
		errors = append(errors, "data element concept ID is required")
	} else {
		if _, ok := r.DataElementConcepts[element.DataElementConceptID]; !ok {
			errors = append(errors, fmt.Sprintf("data element concept '%s' not found", element.DataElementConceptID))
		}
	}

	if element.RepresentationID == "" {
		errors = append(errors, "representation ID is required")
	} else {
		if _, ok := r.Representations[element.RepresentationID]; !ok {
			errors = append(errors, fmt.Sprintf("representation '%s' not found", element.RepresentationID))
		}
	}

	return errors
}

// ValidateRepresentation validates that a representation references a valid value domain if specified.
func (r *MetadataRegistry) ValidateRepresentation(representation *Representation) []string {
	var errors []string

	if representation.ValueDomainID != "" {
		if _, ok := r.ValueDomains[representation.ValueDomainID]; !ok {
			errors = append(errors, fmt.Sprintf("value domain '%s' not found", representation.ValueDomainID))
		}
	}

	return errors
}

// DeleteDataElement removes a data element from the registry.
func (r *MetadataRegistry) DeleteDataElement(identifier string) bool {
	if _, ok := r.DataElements[identifier]; ok {
		delete(r.DataElements, identifier)
		r.UpdatedAt = time.Now().UTC()
		return true
	}
	return false
}

// DeleteDataElementConcept removes a concept from the registry.
// Note: This will fail if any data elements reference this concept.
func (r *MetadataRegistry) DeleteDataElementConcept(identifier string) (bool, error) {
	// Check if any data elements reference this concept
	elements := r.FindDataElementsByConcept(identifier)
	if len(elements) > 0 {
		return false, fmt.Errorf("cannot delete concept '%s': %d data elements reference it", identifier, len(elements))
	}

	if _, ok := r.DataElementConcepts[identifier]; ok {
		delete(r.DataElementConcepts, identifier)
		r.UpdatedAt = time.Now().UTC()
		return true, nil
	}
	return false, nil
}

// DeleteRepresentation removes a representation from the registry.
// Note: This will fail if any data elements reference this representation.
func (r *MetadataRegistry) DeleteRepresentation(identifier string) (bool, error) {
	// Check if any data elements reference this representation
	elements := r.FindDataElementsByRepresentation(identifier)
	if len(elements) > 0 {
		return false, fmt.Errorf("cannot delete representation '%s': %d data elements reference it", identifier, len(elements))
	}

	if _, ok := r.Representations[identifier]; ok {
		delete(r.Representations, identifier)
		r.UpdatedAt = time.Now().UTC()
		return true, nil
	}
	return false, nil
}

// DeleteValueDomain removes a value domain from the registry.
// Note: This will fail if any representations reference this value domain.
func (r *MetadataRegistry) DeleteValueDomain(identifier string) (bool, error) {
	// Check if any representations reference this value domain
	for _, representation := range r.Representations {
		if representation.ValueDomainID == identifier {
			return false, fmt.Errorf("cannot delete value domain '%s': representation '%s' references it", identifier, representation.Identifier)
		}
	}

	if _, ok := r.ValueDomains[identifier]; ok {
		delete(r.ValueDomains, identifier)
		r.UpdatedAt = time.Now().UTC()
		return true, nil
	}
	return false, nil
}

