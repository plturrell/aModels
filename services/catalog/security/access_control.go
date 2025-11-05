package security

import (
	"fmt"
	"time"
)

// AccessControl represents access control for a data product.
type AccessControl struct {
	// Owner is the owner/steward of the data product
	Owner string

	// AllowedUsers are users who have explicit access
	AllowedUsers []string

	// AllowedGroups are groups that have access
	AllowedGroups []string

	// DeniedUsers are users who are explicitly denied access
	DeniedUsers []string

	// DeniedGroups are groups that are denied access
	DeniedGroups []string

	// SensitivityLevel indicates data sensitivity
	SensitivityLevel string // "public", "internal", "confidential", "restricted"

	// DataClassification indicates the type of data
	DataClassification string // "PII", "financial", "health", "general"

	// AccessRules are fine-grained access rules
	AccessRules []AccessRule

	// CreatedAt is when this access control was created
	CreatedAt time.Time

	// UpdatedAt is when this access control was last updated
	UpdatedAt time.Time
}

// AccessRule represents a fine-grained access rule.
type AccessRule struct {
	// Principal is the user or group
	Principal string

	// PrincipalType is "user" or "group"
	PrincipalType string

	// Action is the action allowed (e.g., "read", "write", "delete")
	Action string

	// Conditions are additional conditions (e.g., "time_range", "ip_range")
	Conditions map[string]string

	// Effect is "allow" or "deny"
	Effect string
}

// CheckAccess checks if a user/group has access to perform an action.
func (ac *AccessControl) CheckAccess(principal string, principalType string, action string) (bool, string) {
	// Check explicit denies first
	if ac.isDenied(principal, principalType) {
		return false, "access denied"
	}

	// Check explicit allows
	if ac.isAllowed(principal, principalType) {
		// Check fine-grained rules
		if ac.checkRules(principal, principalType, action) {
			return true, "access granted"
		}
		return false, "action not permitted"
	}

	// Check sensitivity level for default access
	switch ac.SensitivityLevel {
	case "public":
		return true, "public access"
	case "internal":
		// Internal requires authentication but no explicit grant
		return true, "internal access"
	case "confidential", "restricted":
		return false, "requires explicit access grant"
	default:
		return false, "unknown sensitivity level"
	}
}

// isDenied checks if a principal is explicitly denied.
func (ac *AccessControl) isDenied(principal string, principalType string) bool {
	if principalType == "user" {
		for _, denied := range ac.DeniedUsers {
			if denied == principal {
				return true
			}
		}
	}
	if principalType == "group" {
		for _, denied := range ac.DeniedGroups {
			if denied == principal {
				return true
			}
		}
	}
	return false
}

// isAllowed checks if a principal is explicitly allowed.
func (ac *AccessControl) isAllowed(principal string, principalType string) bool {
	if principalType == "user" {
		for _, allowed := range ac.AllowedUsers {
			if allowed == principal {
				return true
			}
		}
	}
	if principalType == "group" {
		for _, allowed := range ac.AllowedGroups {
			if allowed == principal {
				return true
			}
		}
	}
	return false
}

// checkRules checks fine-grained access rules.
func (ac *AccessControl) checkRules(principal string, principalType string, action string) bool {
	for _, rule := range ac.AccessRules {
		if rule.Principal == principal && rule.PrincipalType == principalType {
			if rule.Action == action || rule.Action == "*" {
				if rule.Effect == "allow" {
					return true
				} else if rule.Effect == "deny" {
					return false
				}
			}
		}
	}
	// Default: allow if no specific rule
	return true
}

// GrantAccess grants access to a principal.
func (ac *AccessControl) GrantAccess(principal string, principalType string) {
	if principalType == "user" {
		// Remove from denied if present
		for i, denied := range ac.DeniedUsers {
			if denied == principal {
				ac.DeniedUsers = append(ac.DeniedUsers[:i], ac.DeniedUsers[i+1:]...)
				break
			}
		}
		// Add to allowed if not present
		found := false
		for _, allowed := range ac.AllowedUsers {
			if allowed == principal {
				found = true
				break
			}
		}
		if !found {
			ac.AllowedUsers = append(ac.AllowedUsers, principal)
		}
	} else if principalType == "group" {
		// Remove from denied if present
		for i, denied := range ac.DeniedGroups {
			if denied == principal {
				ac.DeniedGroups = append(ac.DeniedGroups[:i], ac.DeniedGroups[i+1:]...)
				break
			}
		}
		// Add to allowed if not present
		found := false
		for _, allowed := range ac.AllowedGroups {
			if allowed == principal {
				found = true
				break
			}
		}
		if !found {
			ac.AllowedGroups = append(ac.AllowedGroups, principal)
		}
	}
	ac.UpdatedAt = time.Now()
}

// RevokeAccess revokes access from a principal.
func (ac *AccessControl) RevokeAccess(principal string, principalType string) {
	if principalType == "user" {
		for i, allowed := range ac.AllowedUsers {
			if allowed == principal {
				ac.AllowedUsers = append(ac.AllowedUsers[:i], ac.AllowedUsers[i+1:]...)
				break
			}
		}
	} else if principalType == "group" {
		for i, allowed := range ac.AllowedGroups {
			if allowed == principal {
				ac.AllowedGroups = append(ac.AllowedGroups[:i], ac.AllowedGroups[i+1:]...)
				break
			}
		}
	}
	ac.UpdatedAt = time.Now()
}

// NewAccessControl creates a new access control instance.
func NewAccessControl(owner string, sensitivityLevel string) *AccessControl {
	return &AccessControl{
		Owner:           owner,
		SensitivityLevel: sensitivityLevel,
		AllowedUsers:    []string{},
		AllowedGroups:   []string{},
		DeniedUsers:     []string{},
		DeniedGroups:    []string{},
		AccessRules:     []AccessRule{},
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}
}

// SetDataClassification sets the data classification.
func (ac *AccessControl) SetDataClassification(classification string) {
	ac.DataClassification = classification
	ac.UpdatedAt = time.Now()
}

// AddAccessRule adds a fine-grained access rule.
func (ac *AccessControl) AddAccessRule(rule AccessRule) {
	ac.AccessRules = append(ac.AccessRules, rule)
	ac.UpdatedAt = time.Now()
}

// ValidateAccessControl validates the access control configuration.
func (ac *AccessControl) ValidateAccessControl() error {
	validSensitivityLevels := []string{"public", "internal", "confidential", "restricted"}
	valid := false
	for _, level := range validSensitivityLevels {
		if ac.SensitivityLevel == level {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("invalid sensitivity level: %s", ac.SensitivityLevel)
	}

	if ac.Owner == "" {
		return fmt.Errorf("owner is required")
	}

	return nil
}

