package pool

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// HTTPPool manages HTTP client connection pooling
type HTTPPool struct {
	client     *http.Client
	stats      *HTTPPoolStats
	mu         sync.RWMutex
	maxRetries int
	timeout    time.Duration
}

// HTTPPoolConfig holds HTTP pool configuration
type HTTPPoolConfig struct {
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	MaxConnsPerHost     int
	IdleConnTimeout     time.Duration
	Timeout             time.Duration
	KeepAlive           time.Duration
	TLSHandshakeTimeout time.Duration
	DisableKeepAlives   bool
	DisableCompression  bool
	MaxRetries          int
	InsecureSkipVerify  bool
}

// HTTPPoolStats tracks HTTP pool performance
type HTTPPoolStats struct {
	RequestCount  int64
	SuccessCount  int64
	ErrorCount    int64
	RetryCount    int64
	AvgLatency    time.Duration
	TotalBytes    int64
	mu            sync.RWMutex
}

// NewHTTPPool creates a new HTTP connection pool
func NewHTTPPool(cfg *HTTPPoolConfig) *HTTPPool {
	if cfg == nil {
		cfg = DefaultHTTPPoolConfig()
	}

	// Set defaults
	if cfg.MaxIdleConns == 0 {
		cfg.MaxIdleConns = 100
	}
	if cfg.MaxIdleConnsPerHost == 0 {
		cfg.MaxIdleConnsPerHost = 10
	}
	if cfg.MaxConnsPerHost == 0 {
		cfg.MaxConnsPerHost = 20
	}
	if cfg.IdleConnTimeout == 0 {
		cfg.IdleConnTimeout = 90 * time.Second
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.KeepAlive == 0 {
		cfg.KeepAlive = 30 * time.Second
	}
	if cfg.TLSHandshakeTimeout == 0 {
		cfg.TLSHandshakeTimeout = 10 * time.Second
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 3
	}

	// Configure transport
	transport := &http.Transport{
		MaxIdleConns:        cfg.MaxIdleConns,
		MaxIdleConnsPerHost: cfg.MaxIdleConnsPerHost,
		MaxConnsPerHost:     cfg.MaxConnsPerHost,
		IdleConnTimeout:     cfg.IdleConnTimeout,
		DisableKeepAlives:   cfg.DisableKeepAlives,
		DisableCompression:  cfg.DisableCompression,
		TLSHandshakeTimeout: cfg.TLSHandshakeTimeout,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: cfg.KeepAlive,
		}).DialContext,
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: cfg.InsecureSkipVerify,
		},
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   cfg.Timeout,
	}

	return &HTTPPool{
		client:     client,
		stats:      &HTTPPoolStats{},
		maxRetries: cfg.MaxRetries,
		timeout:    cfg.Timeout,
	}
}

// DefaultHTTPPoolConfig returns default configuration
func DefaultHTTPPoolConfig() *HTTPPoolConfig {
	return &HTTPPoolConfig{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		MaxConnsPerHost:     20,
		IdleConnTimeout:     90 * time.Second,
		Timeout:             30 * time.Second,
		KeepAlive:           30 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
		MaxRetries:          3,
		InsecureSkipVerify:  false,
	}
}

// Do executes an HTTP request with retry logic
func (p *HTTPPool) Do(ctx context.Context, req *http.Request) (*http.Response, error) {
	start := time.Now()
	
	p.stats.mu.Lock()
	p.stats.RequestCount++
	p.stats.mu.Unlock()

	var resp *http.Response
	var err error

	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			p.stats.mu.Lock()
			p.stats.RetryCount++
			p.stats.mu.Unlock()

			// Exponential backoff
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		resp, err = p.client.Do(req.WithContext(ctx))
		if err == nil && resp.StatusCode < 500 {
			// Success or client error (don't retry)
			break
		}

		if resp != nil {
			resp.Body.Close()
		}
	}

	duration := time.Since(start)
	
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	
	if err == nil {
		p.stats.SuccessCount++
		p.stats.AvgLatency = (p.stats.AvgLatency + duration) / 2
	} else {
		p.stats.ErrorCount++
	}

	return resp, err
}

// Get performs a GET request
func (p *HTTPPool) Get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	return p.Do(ctx, req)
}

// Post performs a POST request
func (p *HTTPPool) Post(ctx context.Context, url, contentType string, body interface{}) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodPost, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	return p.Do(ctx, req)
}

// GetStats returns pool statistics
func (p *HTTPPool) GetStats() map[string]interface{} {
	p.stats.mu.RLock()
	defer p.stats.mu.RUnlock()

	successRate := 0.0
	if p.stats.RequestCount > 0 {
		successRate = float64(p.stats.SuccessCount) / float64(p.stats.RequestCount)
	}

	return map[string]interface{}{
		"request_count":  p.stats.RequestCount,
		"success_count":  p.stats.SuccessCount,
		"error_count":    p.stats.ErrorCount,
		"retry_count":    p.stats.RetryCount,
		"success_rate":   successRate,
		"avg_latency_ms": p.stats.AvgLatency.Milliseconds(),
		"total_bytes":    p.stats.TotalBytes,
	}
}

// Close closes the HTTP pool
func (p *HTTPPool) Close() {
	p.client.CloseIdleConnections()
}

// Health checks if the pool is healthy
func (p *HTTPPool) Health(ctx context.Context, healthURL string) error {
	if healthURL == "" {
		return nil
	}

	req, err := http.NewRequest(http.MethodGet, healthURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	resp, err := p.client.Do(req.WithContext(ctx))
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	return nil
}
