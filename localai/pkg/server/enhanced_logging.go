package server

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
)

// HANALoggerInterface defines the interface for HANA logging
type HANALoggerInterface interface {
	LogInference(ctx context.Context, logEntry *storage.InferenceLog) error
}

// EnhancedLogging provides advanced logging capabilities for the VaultGemma server
type EnhancedLogging struct {
	hanaLogger   HANALoggerInterface
	requestCount int64
	errorCount   int64
	startTime    time.Time
}

// NewEnhancedLogging creates a new enhanced logging instance
func NewEnhancedLogging(hanaLogger HANALoggerInterface) *EnhancedLogging {
	return &EnhancedLogging{
		hanaLogger: hanaLogger,
		startTime:  time.Now(),
	}
}

// LogRequestStart logs the start of a request
func (el *EnhancedLogging) LogRequestStart(ctx context.Context, requestID, model, domain, prompt string, userID, sessionID string) {
	if el.hanaLogger == nil {
		return
	}

	// Log request start with minimal data
	logEntry := &storage.InferenceLog{
		RequestID:   requestID,
		Model:       model,
		Domain:      domain,
		Prompt:      prompt,
		UserID:      userID,
		SessionID:   sessionID,
		RequestTime: time.Now(),
		Metadata: map[string]interface{}{
			"event_type": "request_start",
			"status":     "started",
		},
	}

	// Log asynchronously to avoid blocking
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log request start: %v", err)
		}
	}()
}

// LogRequestEnd logs the completion of a request
func (el *EnhancedLogging) LogRequestEnd(ctx context.Context, requestID, model, domain, prompt, response string, tokensUsed int, latencyMs int64, temperature float64, maxTokens int, cacheHit, semanticHit bool, userID, sessionID string, metadata map[string]interface{}) {
	if el.hanaLogger == nil {
		return
	}

	// Enhance metadata with additional context
	enhancedMetadata := make(map[string]interface{})
	for k, v := range metadata {
		enhancedMetadata[k] = v
	}

	enhancedMetadata["event_type"] = "request_complete"
	enhancedMetadata["cache_hit"] = cacheHit
	enhancedMetadata["semantic_hit"] = semanticHit
	enhancedMetadata["total_requests"] = el.requestCount
	enhancedMetadata["uptime_seconds"] = time.Since(el.startTime).Seconds()

	// Create comprehensive log entry
	logEntry := &storage.InferenceLog{
		RequestID:    requestID,
		Model:        model,
		Domain:       domain,
		Prompt:       prompt,
		Response:     response,
		TokensUsed:   tokensUsed,
		LatencyMs:    latencyMs,
		Temperature:  temperature,
		MaxTokens:    maxTokens,
		CacheHit:     cacheHit || semanticHit,
		UserID:       userID,
		SessionID:    sessionID,
		RequestTime:  time.Now().Add(-time.Duration(latencyMs) * time.Millisecond),
		ResponseTime: time.Now(),
		Metadata:     enhancedMetadata,
	}

	// Log asynchronously to avoid blocking response
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log request completion: %v", err)
		}
	}()

	// Update counters
	el.requestCount++
}

// LogError logs an error that occurred during request processing
func (el *EnhancedLogging) LogError(ctx context.Context, requestID, model, domain, prompt, errorMsg string, userID, sessionID string, metadata map[string]interface{}) {
	if el.hanaLogger == nil {
		return
	}

	// Enhance metadata with error context
	enhancedMetadata := make(map[string]interface{})
	for k, v := range metadata {
		enhancedMetadata[k] = v
	}

	enhancedMetadata["event_type"] = "request_error"
	enhancedMetadata["error_count"] = el.errorCount
	enhancedMetadata["uptime_seconds"] = time.Since(el.startTime).Seconds()

	// Create error log entry
	logEntry := &storage.InferenceLog{
		RequestID:    requestID,
		Model:        model,
		Domain:       domain,
		Prompt:       prompt,
		Response:     "",
		TokensUsed:   0,
		LatencyMs:    0,
		Temperature:  0,
		MaxTokens:    0,
		CacheHit:     false,
		UserID:       userID,
		SessionID:    sessionID,
		RequestTime:  time.Now(),
		ResponseTime: time.Now(),
		Error:        errorMsg,
		Metadata:     enhancedMetadata,
	}

	// Log asynchronously
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log error: %v", err)
		}
	}()

	// Update error counter
	el.errorCount++
}

// LogCacheHit logs a cache hit event
func (el *EnhancedLogging) LogCacheHit(ctx context.Context, requestID, model, domain, prompt, response string, tokensUsed int, cacheType string, similarityScore float64, userID, sessionID string) {
	if el.hanaLogger == nil {
		return
	}

	metadata := map[string]interface{}{
		"event_type":       "cache_hit",
		"cache_type":       cacheType,
		"similarity_score": similarityScore,
		"uptime_seconds":   time.Since(el.startTime).Seconds(),
	}

	logEntry := &storage.InferenceLog{
		RequestID:    requestID,
		Model:        model,
		Domain:       domain,
		Prompt:       prompt,
		Response:     response,
		TokensUsed:   tokensUsed,
		LatencyMs:    0, // Cache hits have minimal latency
		Temperature:  0,
		MaxTokens:    0,
		CacheHit:     true,
		UserID:       userID,
		SessionID:    sessionID,
		RequestTime:  time.Now(),
		ResponseTime: time.Now(),
		Metadata:     metadata,
	}

	// Log asynchronously
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log cache hit: %v", err)
		}
	}()
}

// LogModelSwitch logs when a model fallback occurs
func (el *EnhancedLogging) LogModelSwitch(ctx context.Context, requestID, originalModel, fallbackModel, domain, prompt string, reason string, userID, sessionID string) {
	if el.hanaLogger == nil {
		return
	}

	metadata := map[string]interface{}{
		"event_type":     "model_switch",
		"original_model": originalModel,
		"fallback_model": fallbackModel,
		"switch_reason":  reason,
		"uptime_seconds": time.Since(el.startTime).Seconds(),
	}

	logEntry := &storage.InferenceLog{
		RequestID:    requestID,
		Model:        fallbackModel,
		Domain:       domain,
		Prompt:       prompt,
		Response:     "",
		TokensUsed:   0,
		LatencyMs:    0,
		Temperature:  0,
		MaxTokens:    0,
		CacheHit:     false,
		UserID:       userID,
		SessionID:    sessionID,
		RequestTime:  time.Now(),
		ResponseTime: time.Now(),
		Metadata:     metadata,
	}

	// Log asynchronously
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log model switch: %v", err)
		}
	}()
}

// LogPerformanceMetrics logs performance-related metrics
func (el *EnhancedLogging) LogPerformanceMetrics(ctx context.Context, requestID, model, domain string, metrics map[string]interface{}) {
	if el.hanaLogger == nil {
		return
	}

	// Enhance metrics with server-level data
	enhancedMetrics := make(map[string]interface{})
	for k, v := range metrics {
		enhancedMetrics[k] = v
	}

	enhancedMetrics["event_type"] = "performance_metrics"
	enhancedMetrics["total_requests"] = el.requestCount
	enhancedMetrics["error_count"] = el.errorCount
	enhancedMetrics["uptime_seconds"] = time.Since(el.startTime).Seconds()

	logEntry := &storage.InferenceLog{
		RequestID:    requestID,
		Model:        model,
		Domain:       domain,
		Prompt:       "",
		Response:     "",
		TokensUsed:   0,
		LatencyMs:    0,
		Temperature:  0,
		MaxTokens:    0,
		CacheHit:     false,
		UserID:       "",
		SessionID:    "",
		RequestTime:  time.Now(),
		ResponseTime: time.Now(),
		Metadata:     enhancedMetrics,
	}

	// Log asynchronously
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log performance metrics: %v", err)
		}
	}()
}

// GetStats returns current logging statistics
func (el *EnhancedLogging) GetStats() map[string]interface{} {
	uptimeSeconds := time.Since(el.startTime).Seconds()
	if uptimeSeconds <= 0 {
		uptimeSeconds = 1
	}

	requestsPerSecond := float64(el.requestCount) / uptimeSeconds
	errorRate := 0.0
	if el.requestCount > 0 {
		errorRate = float64(el.errorCount) / float64(el.requestCount)
	}

	return map[string]interface{}{
		"total_requests":      el.requestCount,
		"error_count":         el.errorCount,
		"uptime_seconds":      uptimeSeconds,
		"requests_per_second": requestsPerSecond,
		"error_rate":          errorRate,
	}
}

// LogHealthCheck logs a health check event
func (el *EnhancedLogging) LogHealthCheck(ctx context.Context, status string, details map[string]interface{}) {
	if el.hanaLogger == nil {
		return
	}

	metadata := map[string]interface{}{
		"event_type":     "health_check",
		"status":         status,
		"uptime_seconds": time.Since(el.startTime).Seconds(),
	}

	for k, v := range details {
		metadata[k] = v
	}

	logEntry := &storage.InferenceLog{
		RequestID:    fmt.Sprintf("health_%d", time.Now().Unix()),
		Model:        "system",
		Domain:       "health",
		Prompt:       "",
		Response:     status,
		TokensUsed:   0,
		LatencyMs:    0,
		Temperature:  0,
		MaxTokens:    0,
		CacheHit:     false,
		UserID:       "system",
		SessionID:    "health",
		RequestTime:  time.Now(),
		ResponseTime: time.Now(),
		Metadata:     metadata,
	}

	// Log asynchronously
	go func() {
		if err := el.hanaLogger.LogInference(context.Background(), logEntry); err != nil {
			log.Printf("⚠️ Failed to log health check: %v", err)
		}
	}()
}
