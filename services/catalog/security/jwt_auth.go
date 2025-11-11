package security

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// JWTClaims represents the JWT token claims.
type JWTClaims struct {
	UserID    string `json:"user_id"`
	Email     string `json:"email,omitempty"`
	Roles     []string `json:"roles,omitempty"`
	jwt.RegisteredClaims
}

// JWTAuthMiddleware provides JWT-based authentication and authorization middleware.
type JWTAuthMiddleware struct {
	secretKey     []byte
	logger        *log.Logger
	tokenExpiry   time.Duration
	refreshExpiry time.Duration
}

// NewJWTAuthMiddleware creates a new JWT auth middleware.
// Requires JWT_SECRET_KEY environment variable to be set.
func NewJWTAuthMiddleware(logger *log.Logger) (*JWTAuthMiddleware, error) {
	secretKey := os.Getenv("JWT_SECRET_KEY")
	if secretKey == "" {
		// Generate a random secret key if not set (for development only)
		// In production, this should always be set via environment variable
		if os.Getenv("ENVIRONMENT") != "production" {
			key := make([]byte, 32)
			if _, err := rand.Read(key); err != nil {
				return nil, fmt.Errorf("failed to generate secret key: %w", err)
			}
			secretKey = base64.URLEncoding.EncodeToString(key)
			logger.Printf("WARNING: JWT_SECRET_KEY not set, generated temporary key. This should be set in production!")
		} else {
			return nil, fmt.Errorf("JWT_SECRET_KEY environment variable is required in production")
		}
	}

	tokenExpiry := 15 * time.Minute // Access token expires in 15 minutes
	if expiryStr := os.Getenv("JWT_TOKEN_EXPIRY"); expiryStr != "" {
		if parsed, err := time.ParseDuration(expiryStr); err == nil {
			tokenExpiry = parsed
		}
	}

	refreshExpiry := 7 * 24 * time.Hour // Refresh token expires in 7 days
	if expiryStr := os.Getenv("JWT_REFRESH_EXPIRY"); expiryStr != "" {
		if parsed, err := time.ParseDuration(expiryStr); err == nil {
			refreshExpiry = parsed
		}
	}

	return &JWTAuthMiddleware{
		secretKey:     []byte(secretKey),
		logger:        logger,
		tokenExpiry:   tokenExpiry,
		refreshExpiry: refreshExpiry,
	}, nil
}

// GenerateToken generates a new JWT access token for a user.
func (jam *JWTAuthMiddleware) GenerateToken(userID, email string, roles []string) (string, error) {
	now := time.Now()
	claims := JWTClaims{
		UserID: userID,
		Email:  email,
		Roles:  roles,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(now.Add(jam.tokenExpiry)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			Issuer:    "aModels-catalog",
			Subject:   userID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(jam.secretKey)
	if err != nil {
		return "", fmt.Errorf("failed to sign token: %w", err)
	}

	// Audit log token generation
	if jam.logger != nil {
		jam.logger.Printf("[AUDIT] token_generated user_id=%s email=%s roles=%v",
			userID, email, roles)
	}

	return tokenString, nil
}

// GenerateRefreshToken generates a refresh token for token rotation.
func (jam *JWTAuthMiddleware) GenerateRefreshToken(userID string) (string, error) {
	now := time.Now()
	claims := jwt.RegisteredClaims{
		ExpiresAt: jwt.NewNumericDate(now.Add(jam.refreshExpiry)),
		IssuedAt:  jwt.NewNumericDate(now),
		NotBefore: jwt.NewNumericDate(now),
		Issuer:    "aModels-catalog",
		Subject:   userID,
		ID:        fmt.Sprintf("refresh-%s-%d", userID, now.Unix()),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(jam.secretKey)
	if err != nil {
		return "", fmt.Errorf("failed to sign refresh token: %w", err)
	}

	return tokenString, nil
}

// ValidateToken validates a JWT token and returns the claims.
func (jam *JWTAuthMiddleware) ValidateToken(tokenString string) (*JWTClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (interface{}, error) {
		// Validate signing method
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jam.secretKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to parse token: %w", err)
	}

	claims, ok := token.Claims.(*JWTClaims)
	if !ok || !token.Valid {
		return nil, errors.New("invalid token")
	}

	// Check expiration
	if claims.ExpiresAt != nil && time.Now().After(claims.ExpiresAt.Time) {
		return nil, errors.New("token has expired")
	}

	return claims, nil
}

// Middleware returns HTTP middleware for JWT authentication.
func (jam *JWTAuthMiddleware) Middleware(next http.Handler) http.Handler {
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
		claims, err := jam.ValidateToken(tokenString)
		if err != nil {
			jam.logger.Printf("[AUDIT] token_validation_failed error=%s ip=%s", err.Error(), r.RemoteAddr)
			http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
			return
		}

		// Add user information to context
		ctx := context.WithValue(r.Context(), "user_id", claims.UserID)
		ctx = context.WithValue(ctx, "user_email", claims.Email)
		ctx = context.WithValue(ctx, "user_roles", claims.Roles)
		ctx = context.WithValue(ctx, "token_claims", claims)

		// Audit log successful authentication
		if jam.logger != nil {
			jam.logger.Printf("[AUDIT] authenticated user_id=%s email=%s ip=%s path=%s",
				claims.UserID, claims.Email, r.RemoteAddr, r.URL.Path)
		}

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// GetUserFromContext extracts user ID from request context.
func GetUserFromJWTContext(ctx context.Context) (string, []string, error) {
	userID, ok := ctx.Value("user_id").(string)
	if !ok || userID == "" {
		return "", nil, errors.New("user not authenticated")
	}

	roles, _ := ctx.Value("user_roles").([]string)
	return userID, roles, nil
}

// HasRole checks if the user in the context has a specific role.
func HasRole(ctx context.Context, role string) bool {
	roles, ok := ctx.Value("user_roles").([]string)
	if !ok {
		return false
	}

	for _, r := range roles {
		if r == role {
			return true
		}
	}
	return false
}

