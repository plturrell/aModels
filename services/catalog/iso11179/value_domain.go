package iso11179

import (
	"fmt"
	"strconv"
	"strings"
)

// ValueDomain provides additional functionality for value domain management.

// ValidateValue validates a value against the value domain constraints.
func (vd *ValueDomain) ValidateValue(value string) (bool, string) {
	if vd.Type == EnumeratedValueDomain {
		// Check if value is in the permissible values list
		for _, pv := range vd.PermissibleValues {
			if pv.Value == value {
				return true, ""
			}
		}
		return false, fmt.Sprintf("value '%s' is not in the permissible values list", value)
	}

	// For non-enumerated domains, check constraints
	for _, constraint := range vd.Constraints {
		valid, reason := validateConstraint(constraint, value)
		if !valid {
			return false, reason
		}
	}

	return true, ""
}

// validateConstraint validates a value against a single constraint.
func validateConstraint(constraint Constraint, value string) (bool, string) {
	switch constraint.Type {
	case "range":
		// Parse range constraint (e.g., "0-120")
		parts := strings.Split(constraint.Value, "-")
		if len(parts) != 2 {
			return false, fmt.Sprintf("invalid range constraint: %s", constraint.Value)
		}
		min, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
		max, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err1 != nil || err2 != nil {
			return false, fmt.Sprintf("invalid range values: %s", constraint.Value)
		}
		val, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return false, fmt.Sprintf("value '%s' is not a number", value)
		}
		if val < min || val > max {
			return false, fmt.Sprintf("value '%s' is outside range [%g, %g]", value, min, max)
		}
		return true, ""

	case "min":
		min, err := strconv.ParseFloat(constraint.Value, 64)
		if err != nil {
			return false, fmt.Sprintf("invalid min constraint: %s", constraint.Value)
		}
		val, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return false, fmt.Sprintf("value '%s' is not a number", value)
		}
		if val < min {
			return false, fmt.Sprintf("value '%s' is below minimum %g", value, min)
		}
		return true, ""

	case "max":
		max, err := strconv.ParseFloat(constraint.Value, 64)
		if err != nil {
			return false, fmt.Sprintf("invalid max constraint: %s", constraint.Value)
		}
		val, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return false, fmt.Sprintf("value '%s' is not a number", value)
		}
		if val > max {
			return false, fmt.Sprintf("value '%s' is above maximum %g", value, max)
		}
		return true, ""

	case "pattern":
		// Simple pattern matching (for regex patterns, would need regex library)
		if strings.Contains(value, constraint.Value) {
			return true, ""
		}
		return false, fmt.Sprintf("value '%s' does not match pattern '%s'", value, constraint.Value)

	case "length":
		maxLength, err := strconv.Atoi(constraint.Value)
		if err != nil {
			return false, fmt.Sprintf("invalid length constraint: %s", constraint.Value)
		}
		if len(value) > maxLength {
			return false, fmt.Sprintf("value '%s' exceeds maximum length %d", value, maxLength)
		}
		return true, ""

	default:
		return true, "" // Unknown constraint type, assume valid
	}
}

// GetPermissibleValueCount returns the number of permissible values.
func (vd *ValueDomain) GetPermissibleValueCount() int {
	return len(vd.PermissibleValues)
}

// GetPermissibleValuesByOrder returns permissible values sorted by order.
func (vd *ValueDomain) GetPermissibleValuesByOrder() []PermissibleValue {
	values := make([]PermissibleValue, len(vd.PermissibleValues))
	copy(values, vd.PermissibleValues)
	
	// Simple bubble sort by order (could use sort.Slice for better performance)
	for i := 0; i < len(values)-1; i++ {
		for j := i + 1; j < len(values); j++ {
			if values[i].Order > values[j].Order {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
	
	return values
}

// HasConstraints returns whether the value domain has any constraints.
func (vd *ValueDomain) HasConstraints() bool {
	return len(vd.Constraints) > 0
}

// GetConstraintByType returns constraints of a specific type.
func (vd *ValueDomain) GetConstraintByType(constraintType string) []Constraint {
	var result []Constraint
	for _, constraint := range vd.Constraints {
		if constraint.Type == constraintType {
			result = append(result, constraint)
		}
	}
	return result
}

