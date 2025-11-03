package server

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
)

// MockHANALogger for testing
type MockHANALogger struct {
	mu   sync.Mutex
	logs []*storage.InferenceLog
}

func (m *MockHANALogger) LogInference(ctx context.Context, logEntry *storage.InferenceLog) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.logs = append(m.logs, logEntry)
	return nil
}

func (m *MockHANALogger) GetLogs() []*storage.InferenceLog {
	m.mu.Lock()
	defer m.mu.Unlock()
	copied := make([]*storage.InferenceLog, len(m.logs))
	copy(copied, m.logs)
	return copied
}

func (m *MockHANALogger) ClearLogs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.logs = []*storage.InferenceLog{}
}

func waitForLogs(t *testing.T, logger *MockHANALogger, expected int) {
	t.Helper()
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) {
		if len(logger.GetLogs()) >= expected {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestEnhancedLogging(t *testing.T) {
	// Create mock HANA logger
	mockLogger := &MockHANALogger{}

	// Create enhanced logging
	enhancedLogging := NewEnhancedLogging(mockLogger)
	ctx := context.Background()

	t.Run("LogRequestStart", func(t *testing.T) {
		enhancedLogging.LogRequestStart(ctx, "req-123", "vaultgemma-1b", "general", "Test prompt", "user-456", "session-789")

		waitForLogs(t, mockLogger, 1)
		logs := mockLogger.GetLogs()
		if len(logs) == 0 {
			t.Fatal("Expected at least one log entry")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.RequestID != "req-123" {
			t.Errorf("Expected RequestID 'req-123', got '%s'", lastLog.RequestID)
		}
		if lastLog.Model != "vaultgemma-1b" {
			t.Errorf("Expected Model 'vaultgemma-1b', got '%s'", lastLog.Model)
		}
		if lastLog.Prompt != "Test prompt" {
			t.Errorf("Expected Prompt 'Test prompt', got '%s'", lastLog.Prompt)
		}
		if lastLog.UserID != "user-456" {
			t.Errorf("Expected UserID 'user-456', got '%s'", lastLog.UserID)
		}
		if lastLog.SessionID != "session-789" {
			t.Errorf("Expected SessionID 'session-789', got '%s'", lastLog.SessionID)
		}
	})

	t.Run("LogRequestEnd", func(t *testing.T) {
		metadata := map[string]interface{}{
			"test_key": "test_value",
		}

		enhancedLogging.LogRequestEnd(ctx, "req-456", "granite-4.0", "blockchain", "Test prompt", "Test response", 150, 1000, 0.7, 1000, true, false, "user-789", "session-123", metadata)

		waitForLogs(t, mockLogger, 2)
		logs := mockLogger.GetLogs()
		if len(logs) < 2 {
			t.Fatal("Expected at least 2 log entries")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.RequestID != "req-456" {
			t.Errorf("Expected RequestID 'req-456', got '%s'", lastLog.RequestID)
		}
		if lastLog.Model != "granite-4.0" {
			t.Errorf("Expected Model 'granite-4.0', got '%s'", lastLog.Model)
		}
		if lastLog.Response != "Test response" {
			t.Errorf("Expected Response 'Test response', got '%s'", lastLog.Response)
		}
		if lastLog.TokensUsed != 150 {
			t.Errorf("Expected TokensUsed 150, got %d", lastLog.TokensUsed)
		}
		if lastLog.LatencyMs != 1000 {
			t.Errorf("Expected LatencyMs 1000, got %d", lastLog.LatencyMs)
		}
		if !lastLog.CacheHit {
			t.Error("Expected CacheHit to be true")
		}
	})

	t.Run("LogError", func(t *testing.T) {
		metadata := map[string]interface{}{
			"error_context": "test_error",
		}

		enhancedLogging.LogError(ctx, "req-789", "vaultgemma-1b", "general", "Test prompt", "Test error message", "user-999", "session-456", metadata)

		waitForLogs(t, mockLogger, 3)
		logs := mockLogger.GetLogs()
		if len(logs) < 3 {
			t.Fatal("Expected at least 3 log entries")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.RequestID != "req-789" {
			t.Errorf("Expected RequestID 'req-789', got '%s'", lastLog.RequestID)
		}
		if lastLog.Error != "Test error message" {
			t.Errorf("Expected Error 'Test error message', got '%s'", lastLog.Error)
		}
		if lastLog.CacheHit {
			t.Error("Expected CacheHit to be false for error")
		}
	})

	t.Run("LogCacheHit", func(t *testing.T) {
		enhancedLogging.LogCacheHit(ctx, "req-cache", "vaultgemma-1b", "general", "Test prompt", "Cached response", 100, "exact", 1.0, "user-cache", "session-cache")

		waitForLogs(t, mockLogger, 4)
		logs := mockLogger.GetLogs()
		if len(logs) < 4 {
			t.Fatal("Expected at least 4 log entries")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.RequestID != "req-cache" {
			t.Errorf("Expected RequestID 'req-cache', got '%s'", lastLog.RequestID)
		}
		if lastLog.Response != "Cached response" {
			t.Errorf("Expected Response 'Cached response', got '%s'", lastLog.Response)
		}
		if !lastLog.CacheHit {
			t.Error("Expected CacheHit to be true")
		}
	})

	t.Run("LogModelSwitch", func(t *testing.T) {
		enhancedLogging.LogModelSwitch(ctx, "req-switch", "vaultgemma-1b", "granite-4.0", "blockchain", "Test prompt", "model_unavailable", "user-switch", "session-switch")

		waitForLogs(t, mockLogger, 5)
		logs := mockLogger.GetLogs()
		if len(logs) < 5 {
			t.Fatal("Expected at least 5 log entries")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.RequestID != "req-switch" {
			t.Errorf("Expected RequestID 'req-switch', got '%s'", lastLog.RequestID)
		}
		if lastLog.Model != "granite-4.0" {
			t.Errorf("Expected Model 'granite-4.0', got '%s'", lastLog.Model)
		}
		if lastLog.Domain != "blockchain" {
			t.Errorf("Expected Domain 'blockchain', got '%s'", lastLog.Domain)
		}
	})

	t.Run("LogPerformanceMetrics", func(t *testing.T) {
		metrics := map[string]interface{}{
			"cpu_usage":    75.5,
			"memory_usage": 80.2,
		}

		enhancedLogging.LogPerformanceMetrics(ctx, "req-metrics", "vaultgemma-1b", "general", metrics)

		waitForLogs(t, mockLogger, 6)
		logs := mockLogger.GetLogs()
		if len(logs) < 6 {
			t.Fatal("Expected at least 6 log entries")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.RequestID != "req-metrics" {
			t.Errorf("Expected RequestID 'req-metrics', got '%s'", lastLog.RequestID)
		}
		if lastLog.Prompt != "" {
			t.Errorf("Expected empty Prompt for metrics, got '%s'", lastLog.Prompt)
		}
	})

	t.Run("LogHealthCheck", func(t *testing.T) {
		details := map[string]interface{}{
			"database_status": "connected",
			"cache_status":    "healthy",
		}

		enhancedLogging.LogHealthCheck(ctx, "healthy", details)

		waitForLogs(t, mockLogger, 7)
		logs := mockLogger.GetLogs()
		if len(logs) < 7 {
			t.Fatal("Expected at least 7 log entries")
		}

		lastLog := logs[len(logs)-1]
		if lastLog.Model != "system" {
			t.Errorf("Expected Model 'system', got '%s'", lastLog.Model)
		}
		if lastLog.Domain != "health" {
			t.Errorf("Expected Domain 'health', got '%s'", lastLog.Domain)
		}
		if lastLog.Response != "healthy" {
			t.Errorf("Expected Response 'healthy', got '%s'", lastLog.Response)
		}
	})

	t.Run("GetStats", func(t *testing.T) {
		stats := enhancedLogging.GetStats()

		if stats["total_requests"] == nil {
			t.Error("Expected total_requests in stats")
		}
		if stats["error_count"] == nil {
			t.Error("Expected error_count in stats")
		}
		if stats["uptime_seconds"] == nil {
			t.Error("Expected uptime_seconds in stats")
		}
		if stats["requests_per_second"] == nil {
			t.Error("Expected requests_per_second in stats")
		}
		if stats["error_rate"] == nil {
			t.Error("Expected error_rate in stats")
		}
	})
}

func TestEnhancedLoggingNilLogger(t *testing.T) {
	// Test with nil logger (should not panic)
	enhancedLogging := NewEnhancedLogging(nil)
	ctx := context.Background()

	// These should not panic
	enhancedLogging.LogRequestStart(ctx, "req-123", "model", "domain", "prompt", "user", "session")
	enhancedLogging.LogRequestEnd(ctx, "req-123", "model", "domain", "prompt", "response", 100, 1000, 0.7, 1000, false, false, "user", "session", nil)
	enhancedLogging.LogError(ctx, "req-123", "model", "domain", "prompt", "error", "user", "session", nil)
	enhancedLogging.LogCacheHit(ctx, "req-123", "model", "domain", "prompt", "response", 100, "exact", 1.0, "user", "session")
	enhancedLogging.LogModelSwitch(ctx, "req-123", "model1", "model2", "domain", "prompt", "reason", "user", "session")
	enhancedLogging.LogPerformanceMetrics(ctx, "req-123", "model", "domain", nil)
	enhancedLogging.LogHealthCheck(ctx, "healthy", nil)

	// Get stats should still work
	stats := enhancedLogging.GetStats()
	if stats == nil {
		t.Error("Expected stats to be returned even with nil logger")
	}
}

func TestEnhancedLoggingConcurrency(t *testing.T) {
	mockLogger := &MockHANALogger{}
	enhancedLogging := NewEnhancedLogging(mockLogger)
	ctx := context.Background()

	// Test concurrent logging
	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func(id int) {
			enhancedLogging.LogRequestStart(ctx, fmt.Sprintf("req-%d", id), "model", "domain", "prompt", "user", "session")
			enhancedLogging.LogRequestEnd(ctx, fmt.Sprintf("req-%d", id), "model", "domain", "prompt", "response", 100, 1000, 0.7, 1000, false, false, "user", "session", nil)
			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}

	expectedLogs := 20 // 10 start + 10 end
	// Check that we have the expected number of logs
	waitForLogs(t, mockLogger, expectedLogs)
	logs := mockLogger.GetLogs()
	if len(logs) != expectedLogs {
		t.Errorf("Expected %d log entries, got %d", expectedLogs, len(logs))
	}
}

func BenchmarkEnhancedLogging(b *testing.B) {
	mockLogger := &MockHANALogger{}
	enhancedLogging := NewEnhancedLogging(mockLogger)
	ctx := context.Background()

	b.Run("LogRequestStart", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			enhancedLogging.LogRequestStart(ctx, fmt.Sprintf("req-%d", i), "model", "domain", "prompt", "user", "session")
		}
	})

	b.Run("LogRequestEnd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			enhancedLogging.LogRequestEnd(ctx, fmt.Sprintf("req-%d", i), "model", "domain", "prompt", "response", 100, 1000, 0.7, 1000, false, false, "user", "session", nil)
		}
	})

	b.Run("LogError", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			enhancedLogging.LogError(ctx, fmt.Sprintf("req-%d", i), "model", "domain", "prompt", "error", "user", "session", nil)
		}
	})

	b.Run("LogCacheHit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			enhancedLogging.LogCacheHit(ctx, fmt.Sprintf("req-%d", i), "model", "domain", "prompt", "response", 100, "exact", 1.0, "user", "session")
		}
	})

	b.Run("GetStats", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = enhancedLogging.GetStats()
		}
	})
}

func ExampleEnhancedLogging() {
	// Create mock HANA logger (in real usage, this would be a real HANALogger)
	mockLogger := &MockHANALogger{}

	// Create enhanced logging
	enhancedLogging := NewEnhancedLogging(mockLogger)
	ctx := context.Background()

	// Log request start
	enhancedLogging.LogRequestStart(ctx, "req-123", "vaultgemma-1b", "general", "What is AI?", "user-456", "session-789")

	// Log cache hit
	enhancedLogging.LogCacheHit(ctx, "req-123", "vaultgemma-1b", "general", "What is AI?", "AI is artificial intelligence...", 50, "exact", 1.0, "user-456", "session-789")

	// Log request completion
	metadata := map[string]interface{}{
		"processing_time": 150,
		"cache_hit":       true,
	}
	enhancedLogging.LogRequestEnd(ctx, "req-123", "vaultgemma-1b", "general", "What is AI?", "AI is artificial intelligence...", 50, 150, 0.7, 1000, true, false, "user-456", "session-789", metadata)

	// Log error if something goes wrong
	enhancedLogging.LogError(ctx, "req-456", "vaultgemma-1b", "general", "Complex query", "Model overloaded", "user-789", "session-123", nil)

	// Log model switch
	enhancedLogging.LogModelSwitch(ctx, "req-789", "vaultgemma-1b", "granite-4.0", "blockchain", "Blockchain question", "model_unavailable", "user-999", "session-456")

	// Log performance metrics
	metrics := map[string]interface{}{
		"cpu_usage":     75.5,
		"memory_usage":  80.2,
		"response_time": 150,
	}
	enhancedLogging.LogPerformanceMetrics(ctx, "req-metrics", "vaultgemma-1b", "general", metrics)

	// Log health check
	healthDetails := map[string]interface{}{
		"database_status": "connected",
		"cache_status":    "healthy",
		"model_status":    "available",
	}
	enhancedLogging.LogHealthCheck(ctx, "healthy", healthDetails)

	// Get statistics
	stats := enhancedLogging.GetStats()
	fmt.Printf("Total requests: %v\n", stats["total_requests"])
	fmt.Printf("Error count: %v\n", stats["error_count"])
	fmt.Printf("Uptime: %v seconds\n", stats["uptime_seconds"])
	fmt.Printf("Requests per second: %v\n", stats["requests_per_second"])
	fmt.Printf("Error rate: %v\n", stats["error_rate"])
}
