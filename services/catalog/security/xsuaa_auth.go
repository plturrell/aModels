package security

import (
	"context"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// XSUAAConfig holds XSUAA service configuration
type XSUAAConfig struct {
	ClientID     string
	ClientSecret string
	URL          string
	VerificationKey string
	XSAppName    string
}

// XSUAAClaims represents XSUAA JWT token claims
type XSUAAClaims struct {
	UserID      string   `json:"user_id"`
	UserName    string   `json:"user_name"`
	Email       string   `json:"email"`
	GivenName   string   `json:"given_name"`
	FamilyName  string   `json:"family_name"`
	Scopes      []string `json:"scope"`
	Authorities []string `json:"authorities"`
	ZDN         string   `json:"zdn"` // Zone Domain Name
	ZID         string   `json:"zid"` // Zone ID
	jwt.RegisteredClaims
}

// XSUAAMiddleware provides XSUAA-based authentication and authorization
type XSUAAMiddleware struct {
	config          *XSUAAConfig
	logger          *log.Logger
	verificationKey *rsa.PublicKey
	keyCache        sync.Map // Cache for verification keys
	keyCacheExpiry  time.Time
}

// NewXSUAAMiddleware creates a new XSUAA middleware from environment variables
func NewXSUAAMiddleware(logger *log.Logger) (*XSUAAMiddleware, error) {
	// Load XSUAA configuration from environment (set by SAP BTP)
	xsuaaURL := os.Getenv("VCAP_SERVICES")
	if xsuaaURL == "" {
		// Try direct environment variables (for local development)
		xsuaaURL = os.Getenv("XSUAA_URL")
	}

	config := &XSUAAConfig{
		ClientID:     os.Getenv("XSUAA_CLIENT_ID"),
		ClientSecret: os.Getenv("XSUAA_CLIENT_SECRET"),
		URL:          xsuaaURL,
		XSAppName:    os.Getenv("XS_APP_NAME"),
	}

	// Parse VCAP_SERVICES if available (SAP BTP standard)
	if xsuaaURL != "" && strings.Contains(xsuaaURL, "xsuaa") {
		var vcap map[string][]map[string]interface{}
		if err := json.Unmarshal([]byte(xsuaaURL), &vcap); err == nil {
			if xsuaaServices, ok := vcap["xsuaa"]; ok && len(xsuaaServices) > 0 {
				service := xsuaaServices[0]
				if creds, ok := service["credentials"].(map[string]interface{}); ok {
					if url, ok := creds["url"].(string); ok {
						config.URL = url
					}
					if clientID, ok := creds["clientid"].(string); ok {
						config.ClientID = clientID
					}
					if clientSecret, ok := creds["clientsecret"].(string); ok {
						config.ClientSecret = clientSecret
					}
					if xsappname, ok := creds["xsappname"].(string); ok {
						config.XSAppName = xsappname
					}
				}
			}
		}
	}

	// Validate required configuration
	if config.ClientID == "" {
		return nil, fmt.Errorf("XSUAA_CLIENT_ID or VCAP_SERVICES with xsuaa credentials is required")
	}
	if config.URL == "" {
		return nil, fmt.Errorf("XSUAA_URL or VCAP_SERVICES with xsuaa credentials is required")
	}

	middleware := &XSUAAMiddleware{
		config: config,
		logger: logger,
	}

	// Load verification key
	if err := middleware.loadVerificationKey(); err != nil {
		logger.Printf("WARNING: Failed to load XSUAA verification key: %v", err)
		logger.Printf("XSUAA token validation may fail. Ensure XSUAA service is accessible.")
	}

	return middleware, nil
}

// loadVerificationKey loads the public key from XSUAA service for token verification
func (xm *XSUAAMiddleware) loadVerificationKey() error {
	// Try to get key from environment first (for local development)
	keyStr := os.Getenv("XSUAA_VERIFICATION_KEY")
	if keyStr == "" {
		// In SAP BTP, fetch from XSUAA service
		// For now, we'll use the token's public key endpoint
		// This should be implemented to fetch from XSUAA's /token_keys endpoint
		return fmt.Errorf("XSUAA_VERIFICATION_KEY not set and cannot fetch from service")
	}

	// Decode the public key
	keyBytes, err := base64.StdEncoding.DecodeString(keyStr)
	if err != nil {
		return fmt.Errorf("failed to decode verification key: %w", err)
	}

	// Parse RSA public key
	pubKey, err := jwt.ParseRSAPublicKeyFromPEM(keyBytes)
	if err != nil {
		return fmt.Errorf("failed to parse RSA public key: %w", err)
	}

	xm.verificationKey = pubKey
	return nil
}

// ValidateToken validates an XSUAA JWT token
func (xm *XSUAAMiddleware) ValidateToken(tokenString string) (*XSUAAClaims, error) {
	// Parse token
	token, err := jwt.ParseWithClaims(tokenString, &XSUAAClaims{}, func(token *jwt.Token) (interface{}, error) {
		// Verify signing method
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}

		// Use verification key if available
		if xm.verificationKey != nil {
			return xm.verificationKey, nil
		}

		// Fallback: try to get key from token's kid (key ID) header
		// In production, this should fetch from XSUAA's /token_keys endpoint
		return nil, fmt.Errorf("verification key not available")
	})

	if err != nil {
		return nil, fmt.Errorf("failed to parse token: %w", err)
	}

	claims, ok := token.Claims.(*XSUAAClaims)
	if !ok || !token.Valid {
		return nil, errors.New("invalid token")
	}

	// Validate token expiration
	if claims.ExpiresAt != nil && time.Now().After(claims.ExpiresAt.Time) {
		return nil, errors.New("token has expired")
	}

	// Validate issuer (should be XSUAA service)
	if claims.Issuer != "" && !strings.Contains(claims.Issuer, "xsuaa") {
		xm.logger.Printf("WARNING: Token issuer may not be XSUAA: %s", claims.Issuer)
	}

	return claims, nil
}

// HasScope checks if the token has a specific scope
func (xm *XSUAAMiddleware) HasScope(claims *XSUAAClaims, scope string) bool {
	// XSUAA scopes are typically in format: <xsappname>.<scope>
	fullScope := fmt.Sprintf("%s.%s", xm.config.XSAppName, scope)
	
	for _, s := range claims.Scopes {
		if s == fullScope || s == scope {
			return true
		}
	}
	return false
}

// HasAuthority checks if the token has a specific authority (role)
func (xm *XSUAAMiddleware) HasAuthority(claims *XSUAAClaims, authority string) bool {
	for _, a := range claims.Authorities {
		if a == authority {
			return true
		}
	}
	return false
}

// Middleware returns HTTP middleware for XSUAA authentication
func (xm *XSUAAMiddleware) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Extract token from Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Authorization header required", http.StatusUnauthorized)
			return
		}

		// Parse Bearer token
		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			http.Error(w, "Invalid authorization format", http.StatusUnauthorized)
			return
		}

		tokenString := parts[1]
		claims, err := xm.ValidateToken(tokenString)
		if err != nil {
			xm.logger.Printf("[AUDIT] xsuaa_token_validation_failed error=%s ip=%s", err.Error(), r.RemoteAddr)
			http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
			return
		}

		// Add user information to context
		ctx := context.WithValue(r.Context(), "user_id", claims.UserID)
		ctx = context.WithValue(ctx, "user_name", claims.UserName)
		ctx = context.WithValue(ctx, "user_email", claims.Email)
		ctx = context.WithValue(ctx, "user_scopes", claims.Scopes)
		ctx = context.WithValue(ctx, "user_authorities", claims.Authorities)
		ctx = context.WithValue(ctx, "xsuaa_claims", claims)
		ctx = context.WithValue(ctx, "zone_id", claims.ZID)

		// Audit log successful authentication
		if xm.logger != nil {
			xm.logger.Printf("[AUDIT] xsuaa_authenticated user_id=%s user_name=%s email=%s scopes=%v ip=%s path=%s",
				claims.UserID, claims.UserName, claims.Email, claims.Scopes, r.RemoteAddr, r.URL.Path)
		}

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// RequireScope creates middleware that requires a specific scope
func (xm *XSUAAMiddleware) RequireScope(scope string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims, ok := r.Context().Value("xsuaa_claims").(*XSUAAClaims)
			if !ok {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
				return
			}

			if !xm.HasScope(claims, scope) {
				xm.logger.Printf("[AUDIT] scope_denied user_id=%s scope=%s path=%s",
					claims.UserID, scope, r.URL.Path)
				http.Error(w, fmt.Sprintf("Insufficient scope. Required: %s", scope), http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequireAuthority creates middleware that requires a specific authority (role)
func (xm *XSUAAMiddleware) RequireAuthority(authority string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims, ok := r.Context().Value("xsuaa_claims").(*XSUAAClaims)
			if !ok {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
				return
			}

			if !xm.HasAuthority(claims, authority) {
				xm.logger.Printf("[AUDIT] authority_denied user_id=%s authority=%s path=%s",
					claims.UserID, authority, r.URL.Path)
				http.Error(w, fmt.Sprintf("Insufficient authority. Required: %s", authority), http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// GetUserFromXSUAAContext extracts user information from XSUAA context
func GetUserFromXSUAAContext(ctx context.Context) (userID, userName, email string, scopes []string, err error) {
	claims, ok := ctx.Value("xsuaa_claims").(*XSUAAClaims)
	if !ok {
		return "", "", "", nil, errors.New("user not authenticated via XSUAA")
	}

	return claims.UserID, claims.UserName, claims.Email, claims.Scopes, nil
}

