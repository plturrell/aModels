package api

import (
	"crypto/subtle"
	"log"
	"net/http"
	"strings"
)

// AuthConfig holds authentication configuration
type AuthConfig struct {
	Enabled      bool
	APIKeys      map[string]string // key -> service name
	HeaderName   string
	Logger       *log.Logger
}

// AuthMiddleware provides API key authentication
func AuthMiddleware(config *AuthConfig) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip authentication if disabled
			if !config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Skip authentication for health check and metrics
			if r.URL.Path == "/healthz" || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			// Get API key from header
			apiKey := r.Header.Get(config.HeaderName)
			if apiKey == "" {
				// Also check Authorization header with Bearer token
				auth := r.Header.Get("Authorization")
				if strings.HasPrefix(auth, "Bearer ") {
					apiKey = strings.TrimPrefix(auth, "Bearer ")
				}
			}

			if apiKey == "" {
				config.Logger.Printf("Authentication failed: missing API key from %s", r.RemoteAddr)
				http.Error(w, "Missing API key", http.StatusUnauthorized)
				return
			}

			// Validate API key
			serviceName, valid := validateAPIKey(apiKey, config.APIKeys)
			if !valid {
				config.Logger.Printf("Authentication failed: invalid API key from %s", r.RemoteAddr)
				http.Error(w, "Invalid API key", http.StatusUnauthorized)
				return
			}

			// Add service name to request context for logging
			config.Logger.Printf("Authenticated request from service: %s, path: %s", serviceName, r.URL.Path)

			next.ServeHTTP(w, r)
		})
	}
}

// validateAPIKey validates an API key using constant-time comparison
func validateAPIKey(providedKey string, validKeys map[string]string) (string, bool) {
	for validKey, serviceName := range validKeys {
		// Use constant-time comparison to prevent timing attacks
		if subtle.ConstantTimeCompare([]byte(providedKey), []byte(validKey)) == 1 {
			return serviceName, true
		}
	}
	return "", false
}
