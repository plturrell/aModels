package middleware

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// AuthConfig holds authentication configuration
type AuthConfig struct {
	Enabled          bool
	AuthType         string // "jwt", "apikey", or "none"
	JWTSecret        string
	APIKeys          []string
	AllowPublicPaths []string // Paths that don't require auth
}

// LoadAuthConfig loads authentication configuration from environment
func LoadAuthConfig() *AuthConfig {
	enabled := os.Getenv("EXTRACT_AUTH_ENABLED") == "true"
	authType := os.Getenv("EXTRACT_AUTH_TYPE")
	if authType == "" {
		authType = "apikey" // Default to API key
	}

	// Load API keys from environment (comma-separated)
	apiKeysStr := os.Getenv("EXTRACT_API_KEYS")
	var apiKeys []string
	if apiKeysStr != "" {
		apiKeys = strings.Split(apiKeysStr, ",")
		for i := range apiKeys {
			apiKeys[i] = strings.TrimSpace(apiKeys[i])
		}
	}

	// Public paths that don't require authentication
	publicPathsStr := os.Getenv("EXTRACT_PUBLIC_PATHS")
	var publicPaths []string
	if publicPathsStr != "" {
		publicPaths = strings.Split(publicPathsStr, ",")
		for i := range publicPaths {
			publicPaths[i] = strings.TrimSpace(publicPaths[i])
		}
	} else {
		// Default public paths
		publicPaths = []string{"/health", "/healthz", "/ready"}
	}

	return &AuthConfig{
		Enabled:          enabled,
		AuthType:         authType,
		JWTSecret:        os.Getenv("EXTRACT_JWT_SECRET"),
		APIKeys:          apiKeys,
		AllowPublicPaths: publicPaths,
	}
}

// AuthMiddleware provides authentication middleware
type AuthMiddleware struct {
	config *AuthConfig
	logger *log.Logger
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(config *AuthConfig, logger *log.Logger) *AuthMiddleware {
	return &AuthMiddleware{
		config: config,
		logger: logger,
	}
}

// Middleware returns HTTP middleware for authentication
func (am *AuthMiddleware) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth if disabled
		if !am.config.Enabled {
			next.ServeHTTP(w, r)
			return
		}

		// Check if path is public
		for _, publicPath := range am.config.AllowPublicPaths {
			if r.URL.Path == publicPath || strings.HasPrefix(r.URL.Path, publicPath+"/") {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Authenticate based on type
		var authenticated bool
		var userID string

		switch am.config.AuthType {
		case "jwt":
			authenticated, userID = am.authenticateJWT(r)
		case "apikey":
			authenticated, userID = am.authenticateAPIKey(r)
		default:
			// No auth required
			authenticated = true
		}

		if !authenticated {
			am.logger.Printf("[AUDIT] authentication_failed method=%s path=%s ip=%s",
				r.Method, r.URL.Path, r.RemoteAddr)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// Add user info to context
		if userID != "" {
			ctx := context.WithValue(r.Context(), "user_id", userID)
			r = r.WithContext(ctx)
		}

		am.logger.Printf("[AUDIT] authenticated method=%s path=%s user=%s ip=%s",
			r.Method, r.URL.Path, userID, r.RemoteAddr)

		next.ServeHTTP(w, r)
	})
}

// authenticateJWT validates JWT token
func (am *AuthMiddleware) authenticateJWT(r *http.Request) (bool, string) {
	if am.config.JWTSecret == "" {
		return false, ""
	}

	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return false, ""
	}

	parts := strings.Split(authHeader, " ")
	if len(parts) != 2 || parts[0] != "Bearer" {
		return false, ""
	}

	tokenString := parts[1]
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(am.config.JWTSecret), nil
	})

	if err != nil || !token.Valid {
		return false, ""
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return false, ""
	}

	userID, _ := claims["sub"].(string)
	if userID == "" {
		userID, _ = claims["user_id"].(string)
	}

	return true, userID
}

// authenticateAPIKey validates API key
func (am *AuthMiddleware) authenticateAPIKey(r *http.Request) (bool, string) {
	if len(am.config.APIKeys) == 0 {
		return false, ""
	}

	// Check X-API-Key header
	apiKey := r.Header.Get("X-API-Key")
	if apiKey == "" {
		// Fallback to Authorization header with "ApiKey" prefix
		authHeader := r.Header.Get("Authorization")
		if strings.HasPrefix(authHeader, "ApiKey ") {
			apiKey = strings.TrimPrefix(authHeader, "ApiKey ")
		}
	}

	if apiKey == "" {
		return false, ""
	}

	// Validate against configured keys
	for _, validKey := range am.config.APIKeys {
		if hmac.Equal([]byte(apiKey), []byte(validKey)) {
			// Generate a simple user ID from key hash for tracking
			hash := sha256.Sum256([]byte(apiKey))
			userID := base64.URLEncoding.EncodeToString(hash[:])[:16]
			return true, userID
		}
	}

	return false, ""
}

// GetUserFromContext extracts user ID from request context
func GetUserFromContext(ctx context.Context) (string, bool) {
	userID, ok := ctx.Value("user_id").(string)
	return userID, ok
}

