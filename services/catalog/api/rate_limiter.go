package api

import (
	"net/http"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// RateLimiter provides rate limiting for API endpoints.
type RateLimiter struct {
	limiters map[string]*rate.Limiter
	mu       sync.RWMutex
	global   *rate.Limiter
}

// NewRateLimiter creates a new rate limiter.
func NewRateLimiter(globalRPS float64, burst int) *RateLimiter {
	rl := &RateLimiter{
		limiters: make(map[string]*rate.Limiter),
		global:   rate.NewLimiter(rate.Limit(globalRPS), burst),
	}

	// Cleanup old limiters periodically
	go rl.cleanup()

	return rl
}

// GetLimiter gets or creates a rate limiter for a key (e.g., IP address).
func (rl *RateLimiter) GetLimiter(key string, rps float64, burst int) *rate.Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if limiter, exists := rl.limiters[key]; exists {
		return limiter
	}

	limiter := rate.NewLimiter(rate.Limit(rps), burst)
	rl.limiters[key] = limiter
	return limiter
}

// Allow checks if a request is allowed (global limiter).
func (rl *RateLimiter) Allow() bool {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	return rl.global.Allow()
}

// AllowKey checks if a request is allowed for a specific key.
func (rl *RateLimiter) AllowKey(key string) bool {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	
	if limiter, exists := rl.limiters[key]; exists {
		return limiter.Allow()
	}
	
	return rl.global.Allow()
}

// cleanup removes old limiters periodically.
func (rl *RateLimiter) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		rl.mu.Lock()
		// Keep only active limiters (simplified - in production would track last access)
		if len(rl.limiters) > 1000 {
			// Clear half of limiters (FIFO approximation)
			cleared := 0
			for k := range rl.limiters {
				if cleared >= len(rl.limiters)/2 {
					break
				}
				delete(rl.limiters, k)
				cleared++
			}
		}
		rl.mu.Unlock()
	}
}

// RateLimitMiddleware provides rate limiting middleware.
func RateLimitMiddleware(limiter *RateLimiter, rps float64, burst int) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Get client IP
			clientIP := r.RemoteAddr
			if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
				clientIP = forwarded
			}

			// Get or create limiter for this IP
			keyLimiter := limiter.GetLimiter(clientIP, rps, burst)

			// Check rate limit
			if !keyLimiter.Allow() {
				w.Header().Set("Retry-After", "60")
				http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// DefaultRateLimiter creates a default rate limiter (100 req/min per IP, 1000/min global).
func DefaultRateLimiter() *RateLimiter {
	return NewRateLimiter(1000.0/60.0, 100) // 1000 requests per minute globally, burst of 100
}

