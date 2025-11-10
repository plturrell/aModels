package rest

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/plturrell/aModels/services/runtime/observability"
)

// TrainingServiceClient fetches analytics from the training service
type TrainingServiceClient struct {
	baseURL    string
	httpClient *http.Client
	cache      *serviceCache
	mu         sync.RWMutex
	clock      func() time.Time
	sleeper    func(time.Duration)
	logger     observability.Logger
}

// serviceCache caches service responses
type serviceCache struct {
	data      interface{}
	expiresAt time.Time
	ttl       time.Duration
}

// NewTrainingServiceClient creates a new training service client
func NewTrainingServiceClient(baseURL string) *TrainingServiceClient {
	if baseURL == "" {
		baseURL = "http://localhost:8001" // Default training service port
	}
	return &TrainingServiceClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		cache: &serviceCache{
			ttl: 30 * time.Second, // Cache for 30 seconds
		},
		clock:   time.Now,
		sleeper: time.Sleep,
		logger:  observability.NewServiceLogger("training"),
	}
}

// TrainingMetrics represents training service metrics
type TrainingMetrics struct {
	ActiveExperiments   int                    `json:"active_experiments"`
	CompletedRuns       int                    `json:"completed_runs"`
	AverageAccuracy     float64                `json:"average_accuracy"`
	TotalTrainingRuns   int                    `json:"total_training_runs"`
	AverageTrainingTime float64                `json:"average_training_time_seconds"`
	Domains             map[string]interface{} `json:"domains,omitempty"`
}

// FetchAnalytics fetches analytics from the training service with caching and retry
func (c *TrainingServiceClient) FetchAnalytics(ctx context.Context) (*TrainingMetrics, error) {
	// Check cache first
	c.mu.RLock()
	if c.cache.data != nil && c.clock().Before(c.cache.expiresAt) {
		if metrics, ok := c.cache.data.(*TrainingMetrics); ok {
			c.mu.RUnlock()
			return metrics, nil
		}
	}
	c.mu.RUnlock()

	// Try to fetch from dashboard/progress endpoint with retry
	var metrics *TrainingMetrics
	var err error

	maxRetries := 3
	for i := 0; i < maxRetries; i++ {
		metrics, err = c.fetchAnalyticsOnce(ctx)
		if err == nil {
			// Cache successful response
			c.mu.Lock()
			c.cache.data = metrics
			c.cache.expiresAt = c.clock().Add(c.cache.ttl)
			c.mu.Unlock()
			return metrics, nil
		}

		// Exponential backoff
		if i < maxRetries-1 {
			backoff := time.Duration(1<<uint(i)) * 100 * time.Millisecond
			c.sleeper(backoff)
		}
	}

	// Return cached data if available, even if expired
	c.mu.RLock()
	if c.cache.data != nil {
		if cachedMetrics, ok := c.cache.data.(*TrainingMetrics); ok {
			c.mu.RUnlock()
			return cachedMetrics, nil // Return stale cache on error
		}
	}
	c.mu.RUnlock()

	// Return default metrics on error
	return &TrainingMetrics{
		ActiveExperiments: 0,
		CompletedRuns:     0,
		AverageAccuracy:   0.0,
	}, err
}

// fetchAnalyticsOnce performs a single fetch attempt
func (c *TrainingServiceClient) fetchAnalyticsOnce(ctx context.Context) (*TrainingMetrics, error) {
	url := fmt.Sprintf("%s/dashboard/progress", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if c.logger != nil {
			c.logger.Debug("training analytics request failed", map[string]any{
				"error": err.Error(),
				"url":   url,
			})
		}
		return nil, fmt.Errorf("failed to fetch training analytics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if c.logger != nil {
			c.logger.Debug("training analytics returned non-success", map[string]any{
				"status": resp.StatusCode,
				"url":    url,
			})
		}
		return nil, fmt.Errorf("training service returned status %d", resp.StatusCode)
	}

	var progressData struct {
		Status                 string  `json:"status"`
		CurrentEpoch           int     `json:"current_epoch"`
		TotalEpochs            int     `json:"total_epochs"`
		ElapsedTime            float64 `json:"elapsed_time"`
		EstimatedTimeRemaining float64 `json:"estimated_time_remaining"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&progressData); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Aggregate metrics from progress data
	metrics := &TrainingMetrics{
		ActiveExperiments: 0,
		CompletedRuns:     0,
		AverageAccuracy:   0.0,
	}

	if progressData.Status == "running" {
		metrics.ActiveExperiments = 1
	}

	return metrics, nil
}

// SearchServiceClient fetches analytics from the search service
type SearchServiceClient struct {
	baseURL    string
	httpClient *http.Client
	cache      *serviceCache
	mu         sync.RWMutex
	clock      func() time.Time
	sleeper    func(time.Duration)
	logger     observability.Logger
}

// NewSearchServiceClient creates a new search service client
func NewSearchServiceClient(baseURL string) *SearchServiceClient {
	if baseURL == "" {
		baseURL = "http://localhost:8000" // Default search service port
	}
	return &SearchServiceClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		cache: &serviceCache{
			ttl: 30 * time.Second, // Cache for 30 seconds
		},
		clock:   time.Now,
		sleeper: time.Sleep,
		logger:  observability.NewServiceLogger("search"),
	}
}

// SearchMetrics represents search service metrics
type SearchMetrics struct {
	TotalQueries     int                      `json:"total_queries"`
	AverageLatencyMs float64                  `json:"average_latency_ms"`
	CacheHitRate     float64                  `json:"cache_hit_rate"`
	TotalCacheHits   int                      `json:"total_cache_hits"`
	TotalCacheMisses int                      `json:"total_cache_misses"`
	PopularQueries   []map[string]interface{} `json:"popular_queries,omitempty"`
	ErrorRate        float64                  `json:"error_rate"`
}

// FetchAnalytics fetches analytics from the search service with caching and retry
func (c *SearchServiceClient) FetchAnalytics(ctx context.Context) (*SearchMetrics, error) {
	// Check cache first
	c.mu.RLock()
	if c.cache.data != nil && c.clock().Before(c.cache.expiresAt) {
		if metrics, ok := c.cache.data.(*SearchMetrics); ok {
			c.mu.RUnlock()
			return metrics, nil
		}
	}
	c.mu.RUnlock()

	// Try to fetch with retry
	var metrics *SearchMetrics
	var err error

	maxRetries := 3
	for i := 0; i < maxRetries; i++ {
		metrics, err = c.fetchAnalyticsOnce(ctx)
		if err == nil {
			// Cache successful response
			c.mu.Lock()
			c.cache.data = metrics
			c.cache.expiresAt = c.clock().Add(c.cache.ttl)
			c.mu.Unlock()
			return metrics, nil
		}

		// Exponential backoff
		if i < maxRetries-1 {
			backoff := time.Duration(1<<uint(i)) * 100 * time.Millisecond
			c.sleeper(backoff)
		}
	}

	// Return cached data if available, even if expired
	c.mu.RLock()
	if c.cache.data != nil {
		if cachedMetrics, ok := c.cache.data.(*SearchMetrics); ok {
			c.mu.RUnlock()
			return cachedMetrics, nil // Return stale cache on error
		}
	}
	c.mu.RUnlock()

	// Return default metrics on error
	return &SearchMetrics{
		TotalQueries:     0,
		AverageLatencyMs: 0.0,
		CacheHitRate:     0.0,
	}, err
}

// fetchAnalyticsOnce performs a single fetch attempt
func (c *SearchServiceClient) fetchAnalyticsOnce(ctx context.Context) (*SearchMetrics, error) {
	url := fmt.Sprintf("%s/v1/search/analytics", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if c.logger != nil {
			c.logger.Debug("search analytics request failed", map[string]any{
				"error": err.Error(),
				"url":   url,
			})
		}
		return nil, fmt.Errorf("failed to fetch search analytics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if c.logger != nil {
			c.logger.Debug("search analytics returned non-success", map[string]any{
				"status": resp.StatusCode,
				"url":    url,
			})
		}
		return nil, fmt.Errorf("search service returned status %d", resp.StatusCode)
	}

	var analyticsData struct {
		TotalQueries     int                      `json:"total_queries"`
		AverageLatencyMs float64                  `json:"average_latency_ms"`
		CacheHitRate     float64                  `json:"cache_hit_rate"`
		PopularQueries   []map[string]interface{} `json:"popular_queries,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&analyticsData); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &SearchMetrics{
		TotalQueries:     analyticsData.TotalQueries,
		AverageLatencyMs: analyticsData.AverageLatencyMs,
		CacheHitRate:     analyticsData.CacheHitRate,
		PopularQueries:   analyticsData.PopularQueries,
	}, nil
}
