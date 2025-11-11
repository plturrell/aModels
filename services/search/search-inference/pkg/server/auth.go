package server

import (
	"crypto/subtle"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// AuthConfig holds authentication configuration
type AuthConfig struct {
	Enabled    bool
	APIKeys    map[string]string // key -> service name
	HeaderName string
	Logger     *log.Logger
}

// RateLimiter provides rate limiting functionality
type RateLimiter struct {
	requestsPerMinute int
	requests          map[string][]time.Time
	mu                sync.RWMutex
	cleanupInterval   time.Duration
	lastCleanup       time.Time
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(requestsPerMinute int) *RateLimiter {
	rl := &RateLimiter{
		requestsPerMinute: requestsPerMinute,
		requests:          make(map[string][]time.Time),
		cleanupInterval:   5 * time.Minute,
		lastCleanup:       time.Now(),
	}
	return rl
}

// IsAllowed checks if a request is allowed and returns (allowed, retryAfter)
func (rl *RateLimiter) IsAllowed(key string) (bool, int) {
	now := time.Now()
	cutoff := now.Add(-1 * time.Minute)

	rl.mu.Lock()
	defer rl.mu.Unlock()

	// Cleanup old entries periodically
	if now.Sub(rl.lastCleanup) > rl.cleanupInterval {
		rl.cleanup(cutoff)
		rl.lastCleanup = now
	}

	// Get requests in the last minute
	recentRequests := make([]time.Time, 0)
	for _, reqTime := range rl.requests[key] {
		if reqTime.After(cutoff) {
			recentRequests = append(recentRequests, reqTime)
		}
	}

	if len(recentRequests) >= rl.requestsPerMinute {
		// Calculate retry after
		oldestRequest := recentRequests[0]
		for _, reqTime := range recentRequests {
			if reqTime.Before(oldestRequest) {
				oldestRequest = reqTime
			}
		}
		retryAfter := int(time.Until(oldestRequest.Add(1*time.Minute)).Seconds()) + 1
		return false, retryAfter
	}

	// Add current request
	recentRequests = append(recentRequests, now)
	rl.requests[key] = recentRequests
	return true, 0
}

func (rl *RateLimiter) cleanup(cutoff time.Time) {
	for key, requests := range rl.requests {
		recent := make([]time.Time, 0)
		for _, reqTime := range requests {
			if reqTime.After(cutoff) {
				recent = append(recent, reqTime)
			}
		}
		if len(recent) > 0 {
			rl.requests[key] = recent
		} else {
			delete(rl.requests, key)
		}
	}
}

// GetClientIdentifier extracts client identifier from request
func GetClientIdentifier(r *http.Request) string {
	// Try X-Forwarded-For header (if behind proxy)
	forwardedFor := r.Header.Get("X-Forwarded-For")
	if forwardedFor != "" {
		// Take the first IP (original client)
		ips := strings.Split(forwardedFor, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// Fall back to direct client IP
	if r.RemoteAddr != "" {
		// Remove port if present
		ip := r.RemoteAddr
		if idx := strings.LastIndex(ip, ":"); idx != -1 {
			ip = ip[:idx]
		}
		return ip
	}

	return "unknown"
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

			// Skip authentication for health check
			if r.URL.Path == "/health" {
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
				if config.Logger != nil {
					config.Logger.Printf("Authentication failed: missing API key from %s", r.RemoteAddr)
				}
				http.Error(w, "Missing API key", http.StatusUnauthorized)
				return
			}

			// Validate API key
			serviceName, valid := validateAPIKey(apiKey, config.APIKeys)
			if !valid {
				if config.Logger != nil {
					config.Logger.Printf("Authentication failed: invalid API key from %s", r.RemoteAddr)
				}
				http.Error(w, "Invalid API key", http.StatusUnauthorized)
				return
			}

			// Log successful authentication
			if config.Logger != nil && serviceName != "" {
				config.Logger.Printf("Authenticated request from service: %s, path: %s", serviceName, r.URL.Path)
			}

			next.ServeHTTP(w, r)
		})
	}
}

// validateAPIKey validates an API key using constant-time comparison
func validateAPIKey(providedKey string, validKeys map[string]string) (string, bool) {
	for key, serviceName := range validKeys {
		if subtle.ConstantTimeCompare([]byte(providedKey), []byte(key)) == 1 {
			return serviceName, true
		}
	}
	return "", false
}

// RateLimitMiddleware provides rate limiting middleware
func RateLimitMiddleware(limiter *RateLimiter, enabled bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip rate limiting if disabled
			if !enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Skip rate limiting for health endpoint
			if r.URL.Path == "/health" {
				next.ServeHTTP(w, r)
				return
			}

			clientID := GetClientIdentifier(r)
			allowed, retryAfter := limiter.IsAllowed(clientID)

			if !allowed {
				w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
				http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// LoadAuthConfig loads authentication configuration from environment
func LoadAuthConfig(logger *log.Logger) *AuthConfig {
	enabled := os.Getenv("AUTH_ENABLED") == "true"
	headerName := os.Getenv("AUTH_HEADER_NAME")
	if headerName == "" {
		headerName = "X-API-Key"
	}

	apiKeys := make(map[string]string)
	keysStr := os.Getenv("API_KEYS")
	if keysStr != "" {
		keys := strings.Split(keysStr, ",")
		for i, key := range keys {
			key = strings.TrimSpace(key)
			if key != "" {
				apiKeys[key] = "service-" + string(rune(i+1))
			}
		}
	}

	return &AuthConfig{
		Enabled:    enabled,
		APIKeys:    apiKeys,
		HeaderName: headerName,
		Logger:     logger,
	}
}

// LoadRateLimiterConfig loads rate limiter configuration from environment
func LoadRateLimiterConfig() (*RateLimiter, bool) {
	enabled := os.Getenv("RATE_LIMIT_ENABLED")
	if enabled == "" {
		enabled = "true" // Default to enabled
	}

	if enabled != "true" {
		return nil, false
	}

	requestsPerMinute := 60 // Default
	if rpmStr := os.Getenv("RATE_LIMIT_PER_MINUTE"); rpmStr != "" {
		if rpm, err := strconv.Atoi(rpmStr); err == nil && rpm > 0 {
			requestsPerMinute = rpm
		}
	}

	return NewRateLimiter(requestsPerMinute), true
}

