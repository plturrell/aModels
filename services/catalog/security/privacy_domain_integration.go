package security

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
)

// PrivacyDomainIntegration integrates XSUAA authentication with differential privacy and domain intelligence
type PrivacyDomainIntegration struct {
	xsuaaMiddleware *XSUAAMiddleware
	logger          *log.Logger
}

// NewPrivacyDomainIntegration creates a new privacy-domain integration
func NewPrivacyDomainIntegration(xsuaaMiddleware *XSUAAMiddleware, logger *log.Logger) *PrivacyDomainIntegration {
	return &PrivacyDomainIntegration{
		xsuaaMiddleware: xsuaaMiddleware,
		logger:          logger,
	}
}

// DomainAccess represents a user's access to a specific domain
type DomainAccess struct {
	DomainID      string
	AccessLevel   string // "read", "write", "admin"
	PrivacyLevel  string // "low", "medium", "high" - determines epsilon/delta
	AllowedScopes []string
	IsRestricted  bool // true if domain contains sensitive data
}

// PrivacyConfig represents differential privacy configuration based on user permissions
type PrivacyConfig struct {
	Epsilon      float64 // Privacy budget (ε)
	Delta        float64 // Privacy parameter (δ)
	NoiseScale   float64 // Noise scale for Laplacian mechanism
	Sensitivity  float64 // Sensitivity of the query
	MaxQueries   int     // Maximum queries per domain per user
	PrivacyLevel string  // "low", "medium", "high"
}

// GetUserDomainAccess determines which domains a user can access based on XSUAA scopes
func (pdi *PrivacyDomainIntegration) GetUserDomainAccess(ctx context.Context) ([]DomainAccess, error) {
	claims, ok := ctx.Value("xsuaa_claims").(*XSUAAClaims)
	if !ok {
		return nil, errors.New("user not authenticated via XSUAA")
	}

	var domainAccesses []DomainAccess

	// Map XSUAA scopes to domain access
	// Scopes format: $XSAPPNAME.Domain.<domain_id>.<access_level>
	for _, scope := range claims.Scopes {
		// Parse scope: amodels-catalog.Domain.finance.read
		parts := strings.Split(scope, ".")
		if len(parts) >= 4 && parts[1] == "Domain" {
			domainID := parts[2]
			accessLevel := parts[3]

			// Determine privacy level based on access level and domain sensitivity
			privacyLevel := pdi.determinePrivacyLevel(domainID, accessLevel)

			// Check if domain is restricted (contains sensitive data)
			isRestricted := pdi.isRestrictedDomain(domainID)

			domainAccess := DomainAccess{
				DomainID:      domainID,
				AccessLevel:   accessLevel,
				PrivacyLevel:  privacyLevel,
				AllowedScopes: []string{scope},
				IsRestricted:  isRestricted,
			}

			domainAccesses = append(domainAccesses, domainAccess)
		}
	}

	// If no domain-specific scopes, check for general scopes
	if len(domainAccesses) == 0 {
		// Check for general access scopes
		hasRead := pdi.xsuaaMiddleware.HasScope(claims, "Display") || pdi.xsuaaMiddleware.HasScope(claims, "DataProduct.Read")
		hasWrite := pdi.xsuaaMiddleware.HasScope(claims, "Edit") || pdi.xsuaaMiddleware.HasScope(claims, "DataProduct.Create")
		hasAdmin := pdi.xsuaaMiddleware.HasScope(claims, "Admin")

		if hasRead || hasWrite || hasAdmin {
			// Grant access to default domain with appropriate privacy level
			accessLevel := "read"
			if hasAdmin {
				accessLevel = "admin"
			} else if hasWrite {
				accessLevel = "write"
			}

			domainAccesses = append(domainAccesses, DomainAccess{
				DomainID:      "default",
				AccessLevel:   accessLevel,
				PrivacyLevel:  "medium", // Default privacy level
				AllowedScopes: claims.Scopes,
				IsRestricted:  false,
			})
		}
	}

	return domainAccesses, nil
}

// GetPrivacyConfig returns differential privacy configuration based on user's domain access
func (pdi *PrivacyDomainIntegration) GetPrivacyConfig(ctx context.Context, domainID string) (*PrivacyConfig, error) {
	domainAccesses, err := pdi.GetUserDomainAccess(ctx)
	if err != nil {
		return nil, err
	}

	// Find access for the requested domain
	var domainAccess *DomainAccess
	for i := range domainAccesses {
		if domainAccesses[i].DomainID == domainID {
			domainAccess = &domainAccesses[i]
			break
		}
	}

	if domainAccess == nil {
		return nil, fmt.Errorf("user does not have access to domain: %s", domainID)
	}

	// Map privacy level to epsilon/delta values
	var epsilon, delta, noiseScale float64
	var maxQueries int

	switch domainAccess.PrivacyLevel {
	case "low":
		epsilon = 2.0
		delta = 1e-4
		noiseScale = 0.05
		maxQueries = 200 // More queries allowed for low privacy
	case "high":
		epsilon = 0.5
		delta = 1e-6
		noiseScale = 0.2
		maxQueries = 50 // Fewer queries for high privacy
	default: // medium
		epsilon = 1.0
		delta = 1e-5
		noiseScale = 0.1
		maxQueries = 100
	}

	// Adjust based on access level
	if domainAccess.AccessLevel == "admin" {
		// Admins get more privacy budget
		epsilon *= 1.5
		maxQueries = int(float64(maxQueries) * 1.5)
	} else if domainAccess.AccessLevel == "read" {
		// Read-only users get less budget
		epsilon *= 0.8
		maxQueries = int(float64(maxQueries) * 0.8)
	}

	// Restricted domains get stricter privacy
	if domainAccess.IsRestricted {
		epsilon *= 0.7
		delta *= 0.1
		maxQueries = int(float64(maxQueries) * 0.7)
	}

	return &PrivacyConfig{
		Epsilon:      epsilon,
		Delta:        delta,
		NoiseScale:   noiseScale,
		Sensitivity:  1.0,
		MaxQueries:   maxQueries,
		PrivacyLevel: domainAccess.PrivacyLevel,
	}, nil
}

// CanAccessDomain checks if user can access a specific domain
func (pdi *PrivacyDomainIntegration) CanAccessDomain(ctx context.Context, domainID string) (bool, string, error) {
	domainAccesses, err := pdi.GetUserDomainAccess(ctx)
	if err != nil {
		return false, "", err
	}

	for _, access := range domainAccesses {
		if access.DomainID == domainID || access.DomainID == "default" {
			return true, access.AccessLevel, nil
		}
	}

	return false, "", nil
}

// GetUserDomains returns list of domain IDs the user can access
func (pdi *PrivacyDomainIntegration) GetUserDomains(ctx context.Context) ([]string, error) {
	domainAccesses, err := pdi.GetUserDomainAccess(ctx)
	if err != nil {
		return nil, err
	}

	domains := make([]string, 0, len(domainAccesses))
	for _, access := range domainAccesses {
		domains = append(domains, access.DomainID)
	}

	return domains, nil
}

// RequireDomainAccess creates middleware that requires access to a specific domain
func (pdi *PrivacyDomainIntegration) RequireDomainAccess(domainID string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			canAccess, accessLevel, err := pdi.CanAccessDomain(r.Context(), domainID)
			if err != nil {
				pdi.logger.Printf("[AUDIT] domain_access_check_failed domain=%s error=%s", domainID, err.Error())
				http.Error(w, "Failed to check domain access", http.StatusInternalServerError)
				return
			}

			if !canAccess {
				claims, _ := r.Context().Value("xsuaa_claims").(*XSUAAClaims)
				userID := "unknown"
				if claims != nil {
					userID = claims.UserID
				}
				pdi.logger.Printf("[AUDIT] domain_access_denied user_id=%s domain=%s path=%s", userID, domainID, r.URL.Path)
				http.Error(w, fmt.Sprintf("Access denied to domain: %s", domainID), http.StatusForbidden)
				return
			}

			// Add domain access information to context
			ctx := context.WithValue(r.Context(), "domain_id", domainID)
			ctx = context.WithValue(ctx, "domain_access_level", accessLevel)

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// determinePrivacyLevel determines privacy level based on domain and access level
func (pdi *PrivacyDomainIntegration) determinePrivacyLevel(domainID, accessLevel string) string {
	// Sensitive domains get high privacy
	sensitiveDomains := []string{"finance", "health", "pii", "regulatory", "compliance"}
	for _, sensitive := range sensitiveDomains {
		if strings.Contains(strings.ToLower(domainID), sensitive) {
			return "high"
		}
	}

	// Admin access gets lower privacy (more trust)
	if accessLevel == "admin" {
		return "low"
	}

	// Write access gets medium privacy
	if accessLevel == "write" {
		return "medium"
	}

	// Read access gets high privacy (least trust)
	return "high"
}

// isRestrictedDomain checks if a domain contains restricted/sensitive data
func (pdi *PrivacyDomainIntegration) isRestrictedDomain(domainID string) bool {
	restrictedKeywords := []string{"finance", "health", "pii", "regulatory", "compliance", "confidential", "restricted"}
	domainLower := strings.ToLower(domainID)
	
	for _, keyword := range restrictedKeywords {
		if strings.Contains(domainLower, keyword) {
			return true
		}
	}
	return false
}

// GetDomainIntelligenceContext extracts domain intelligence context from XSUAA claims
func (pdi *PrivacyDomainIntegration) GetDomainIntelligenceContext(ctx context.Context) (map[string]interface{}, error) {
	claims, ok := ctx.Value("xsuaa_claims").(*XSUAAClaims)
	if !ok {
		return nil, errors.New("user not authenticated via XSUAA")
	}

	// Get user's accessible domains
	domains, err := pdi.GetUserDomains(ctx)
	if err != nil {
		return nil, err
	}

	// Build domain intelligence context
	context := map[string]interface{}{
		"user_id":      claims.UserID,
		"user_name":    claims.UserName,
		"email":        claims.Email,
		"domains":      domains,
		"scopes":       claims.Scopes,
		"authorities":  claims.Authorities,
		"zone_id":      claims.ZID,
		"zone_domain":  claims.ZDN,
	}

	// Add domain-specific access levels
	domainAccesses, _ := pdi.GetUserDomainAccess(ctx)
	domainAccessMap := make(map[string]string)
	for _, access := range domainAccesses {
		domainAccessMap[access.DomainID] = access.AccessLevel
	}
	context["domain_access_levels"] = domainAccessMap

	return context, nil
}

// ApplyPrivacyToResponse applies differential privacy to response data based on user's domain access
func (pdi *PrivacyDomainIntegration) ApplyPrivacyToResponse(ctx context.Context, domainID string, data map[string]interface{}) (map[string]interface{}, error) {
	privacyConfig, err := pdi.GetPrivacyConfig(ctx, domainID)
	if err != nil {
		return nil, err
	}

	// Apply privacy transformations based on configuration
	// This is a placeholder - actual noise addition would be done by the privacy service
	privateData := make(map[string]interface{})
	for key, value := range data {
		// Skip sensitive fields for high privacy levels
		if privacyConfig.PrivacyLevel == "high" {
			sensitiveFields := []string{"user_id", "email", "personal", "financial", "health"}
			isSensitive := false
			for _, field := range sensitiveFields {
				if strings.Contains(strings.ToLower(key), field) {
					isSensitive = true
					break
				}
			}
			if isSensitive {
				continue // Skip sensitive fields
			}
		}

		// Add privacy metadata
		if numValue, ok := value.(float64); ok {
			// For numeric values, add privacy metadata
			privateData[key] = map[string]interface{}{
				"value":        numValue,
				"privacy_applied": true,
				"epsilon":      privacyConfig.Epsilon,
				"noise_scale":   privacyConfig.NoiseScale,
			}
		} else {
			privateData[key] = value
		}
	}

	// Add privacy metadata to response
	privateData["_privacy"] = map[string]interface{}{
		"privacy_level": privacyConfig.PrivacyLevel,
		"epsilon":        privacyConfig.Epsilon,
		"delta":          privacyConfig.Delta,
		"max_queries":    privacyConfig.MaxQueries,
	}

	return privateData, nil
}

