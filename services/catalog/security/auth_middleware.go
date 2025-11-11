package security

import (
	"context"
	"log"
	"net/http"
	"strings"
)

// AuthMiddleware provides authentication and authorization middleware.
type AuthMiddleware struct {
	// In production, would integrate with actual auth provider
	// For now, supports basic token validation
	allowedTokens map[string]string // token -> user
	logger        *log.Logger
}

// NewAuthMiddleware creates a new auth middleware.
func NewAuthMiddleware(logger *log.Logger) *AuthMiddleware {
	// In production, would load from config or external auth service
	return &AuthMiddleware{
		allowedTokens: make(map[string]string),
		logger:        logger,
	}
}

// RegisterToken registers a valid token for a user.
func (am *AuthMiddleware) RegisterToken(token, userID string) {
	am.allowedTokens[token] = userID
}

// Middleware returns HTTP middleware for authentication.
func (am *AuthMiddleware) Middleware(next http.Handler) http.Handler {
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
		
		token := parts[1]
		userID, ok := am.allowedTokens[token]
		if !ok {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}
		
		// Add user to context
		ctx := context.WithValue(r.Context(), "user_id", userID)
		ctx = context.WithValue(ctx, "token", token)
		
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// CheckAccess checks if a user has access to a resource.
func (am *AuthMiddleware) CheckAccess(ctx context.Context, userID string, resource string, action string, accessControl *AccessControl) (bool, string) {
	if accessControl == nil {
		// Default: public access if no access control defined
		return true, "public access"
	}
	
	return accessControl.CheckAccess(userID, "user", action)
}

// GetUserFromContext extracts user ID from request context.
func GetUserFromContext(ctx context.Context) string {
	if userID, ok := ctx.Value("user_id").(string); ok {
		return userID
	}
	return ""
}

// AuditLog logs access for audit purposes.
func (am *AuthMiddleware) AuditLog(userID, resource, action, result string) {
	if am.logger != nil {
		am.logger.Printf("[AUDIT] user=%s resource=%s action=%s result=%s",
			userID, resource, action, result)
	}
}

