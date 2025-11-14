package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRateLimiter_Basic(t *testing.T) {
	// Create limiter: 2 requests per second, burst of 1
	limiter := NewRateLimiter(2.0, 1)

	// Create test handler
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Wrap with rate limiter
	rateLimitedHandler := limiter.Middleware(handler)

	// First request should succeed
	req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1.RemoteAddr = "192.168.1.1:12345"
	rr1 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr1, req1)

	if rr1.Code != http.StatusOK {
		t.Errorf("First request should succeed, got status %d", rr1.Code)
	}

	// Second request immediately after should also succeed (burst)
	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.RemoteAddr = "192.168.1.1:12345"
	rr2 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusOK {
		t.Errorf("Second request should succeed (burst), got status %d", rr2.Code)
	}

	// Third request immediately should be rate limited
	req3 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req3.RemoteAddr = "192.168.1.1:12345"
	rr3 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr3, req3)

	if rr3.Code != http.StatusTooManyRequests {
		t.Errorf("Third request should be rate limited, got status %d", rr3.Code)
	}
}

func TestRateLimiter_DifferentIPs(t *testing.T) {
	// Create limiter: 1 request per second, burst of 1
	limiter := NewRateLimiter(1.0, 1)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	rateLimitedHandler := limiter.Middleware(handler)

	// Request from IP 1
	req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1.RemoteAddr = "192.168.1.1:12345"
	rr1 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr1, req1)

	if rr1.Code != http.StatusOK {
		t.Errorf("Request from IP1 should succeed, got status %d", rr1.Code)
	}

	// Request from IP 2 (should succeed independently)
	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.RemoteAddr = "192.168.1.2:12345"
	rr2 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusOK {
		t.Errorf("Request from IP2 should succeed, got status %d", rr2.Code)
	}

	// Another request from IP 1 (should be rate limited)
	req3 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req3.RemoteAddr = "192.168.1.1:12345"
	rr3 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr3, req3)

	if rr3.Code != http.StatusTooManyRequests {
		t.Errorf("Second request from IP1 should be rate limited, got status %d", rr3.Code)
	}
}

func TestRateLimiter_XForwardedFor(t *testing.T) {
	limiter := NewRateLimiter(1.0, 1)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	rateLimitedHandler := limiter.Middleware(handler)

	// First request with X-Forwarded-For header
	req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1.RemoteAddr = "192.168.1.1:12345"
	req1.Header.Set("X-Forwarded-For", "10.0.0.1")
	rr1 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr1, req1)

	if rr1.Code != http.StatusOK {
		t.Errorf("First request should succeed, got status %d", rr1.Code)
	}

	// Second request with same X-Forwarded-For
	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.RemoteAddr = "192.168.1.2:12345" // Different RemoteAddr
	req2.Header.Set("X-Forwarded-For", "10.0.0.1") // Same forwarded IP
	rr2 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusTooManyRequests {
		t.Errorf("Second request from same forwarded IP should be rate limited, got status %d", rr2.Code)
	}
}

func TestRateLimiter_Recovery(t *testing.T) {
	// Create limiter: 10 requests per second
	limiter := NewRateLimiter(10.0, 1)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	rateLimitedHandler := limiter.Middleware(handler)

	// Make requests until rate limited
	req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req1.RemoteAddr = "192.168.1.1:12345"
	rr1 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr1, req1)

	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.RemoteAddr = "192.168.1.1:12345"
	rr2 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusTooManyRequests {
		t.Errorf("Second request should be rate limited, got status %d", rr2.Code)
	}

	// Wait for token bucket to refill (100ms should be enough for 10 req/sec)
	time.Sleep(150 * time.Millisecond)

	// Request should succeed now
	req3 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req3.RemoteAddr = "192.168.1.1:12345"
	rr3 := httptest.NewRecorder()
	rateLimitedHandler.ServeHTTP(rr3, req3)

	if rr3.Code != http.StatusOK {
		t.Errorf("Request after wait should succeed, got status %d", rr3.Code)
	}
}

func TestRateLimiter_Cleanup(t *testing.T) {
	limiter := NewRateLimiter(1.0, 1)

	// Simulate many IPs to trigger cleanup threshold
	for i := 0; i < 1100; i++ {
		ip := "192.168.1." + string(rune(i%256))
		limiter.getLimiter(ip)
	}

	initialCount := len(limiter.limiters)

	// Run cleanup
	limiter.cleanup()

	// Should have been cleaned up
	if len(limiter.limiters) >= initialCount {
		t.Errorf("Expected cleanup to reduce limiter count from %d, got %d", initialCount, len(limiter.limiters))
	}
}

func BenchmarkRateLimiter(b *testing.B) {
	limiter := NewRateLimiter(1000.0, 10)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	rateLimitedHandler := limiter.Middleware(handler)

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.RemoteAddr = "192.168.1.1:12345"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rr := httptest.NewRecorder()
		rateLimitedHandler.ServeHTTP(rr, req)
	}
}
