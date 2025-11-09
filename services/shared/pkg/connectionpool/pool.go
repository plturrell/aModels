package connectionpool

import (
	"context"
	"net/http"
	"sync"
	"time"
)

// HTTPPoolConfig configures HTTP connection pooling
type HTTPPoolConfig struct {
	MaxIdleConns        int           // Default: 100
	MaxIdleConnsPerHost int           // Default: 10
	IdleConnTimeout     time.Duration // Default: 90s
	MaxConnsPerHost     int           // Default: 50
	Timeout             time.Duration // Default: 30s
}

// DefaultHTTPPoolConfig returns default HTTP pool configuration
func DefaultHTTPPoolConfig() *HTTPPoolConfig {
	return &HTTPPoolConfig{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		MaxConnsPerHost:     50,
		Timeout:             30 * time.Second,
	}
}

// HTTPPoolManager manages HTTP connection pools
type HTTPPoolManager struct {
	config *HTTPPoolConfig
	client *http.Client
	mu     sync.RWMutex
}

// NewHTTPPoolManager creates a new HTTP pool manager
func NewHTTPPoolManager(config *HTTPPoolConfig) *HTTPPoolManager {
	if config == nil {
		config = DefaultHTTPPoolConfig()
	}

	transport := &http.Transport{
		MaxIdleConns:        config.MaxIdleConns,
		MaxIdleConnsPerHost: config.MaxIdleConnsPerHost,
		IdleConnTimeout:     config.IdleConnTimeout,
		MaxConnsPerHost:     config.MaxConnsPerHost,
	}

	return &HTTPPoolManager{
		config: config,
		client: &http.Client{
			Transport: transport,
			Timeout:   config.Timeout,
		},
	}
}

// GetClient returns the shared HTTP client
func (h *HTTPPoolManager) GetClient() *http.Client {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.client
}

// Do performs an HTTP request using the pooled client
func (h *HTTPPoolManager) Do(req *http.Request) (*http.Response, error) {
	return h.GetClient().Do(req)
}

// DoWithContext performs an HTTP request with context using the pooled client
func (h *HTTPPoolManager) DoWithContext(ctx context.Context, req *http.Request) (*http.Response, error) {
	req = req.WithContext(ctx)
	return h.Do(req)
}

// UpdateConfig updates the pool configuration (requires restart to take effect)
func (h *HTTPPoolManager) UpdateConfig(config *HTTPPoolConfig) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.config = config
	// Note: Existing client will continue using old config
	// New client will be created on next GetClient() call
}

// Global HTTP pool manager instance
var (
	globalHTTPPool *HTTPPoolManager
	httpPoolOnce   sync.Once
)

// GetGlobalHTTPPool returns the global HTTP pool manager
func GetGlobalHTTPPool() *HTTPPoolManager {
	httpPoolOnce.Do(func() {
		globalHTTPPool = NewHTTPPoolManager(DefaultHTTPPoolConfig())
	})
	return globalHTTPPool
}

