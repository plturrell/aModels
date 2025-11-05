package observability

import (
	"encoding/json"
	"log"
	"os"
	"time"
)

// LogLevel represents the log level.
type LogLevel string

const (
	LogLevelDebug LogLevel = "DEBUG"
	LogLevelInfo  LogLevel = "INFO"
	LogLevelWarn  LogLevel = "WARN"
	LogLevelError LogLevel = "ERROR"
)

// StructuredLogger provides structured JSON logging.
type StructuredLogger struct {
	logger *log.Logger
	level  LogLevel
}

// LogEntry represents a structured log entry.
type LogEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Service   string                 `json:"service"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// NewStructuredLogger creates a new structured logger.
func NewStructuredLogger(level LogLevel) *StructuredLogger {
	return &StructuredLogger{
		logger: log.New(os.Stdout, "", 0), // No prefix, we'll format everything
		level:  level,
	}
}

// shouldLog checks if a log level should be logged.
func (sl *StructuredLogger) shouldLog(level LogLevel) bool {
	levels := map[LogLevel]int{
		LogLevelDebug: 0,
		LogLevelInfo:  1,
		LogLevelWarn:  2,
		LogLevelError: 3,
	}
	return levels[level] >= levels[sl.level]
}

// log writes a structured log entry.
func (sl *StructuredLogger) log(level LogLevel, message string, fields map[string]interface{}, err error) {
	if !sl.shouldLog(level) {
		return
	}

	entry := LogEntry{
		Timestamp: time.Now(),
		Level:     string(level),
		Message:   message,
		Service:   "catalog",
		Fields:    fields,
	}

	if err != nil {
		entry.Error = err.Error()
	}

	jsonData, jsonErr := json.Marshal(entry)
	if jsonErr != nil {
		// Fallback to simple logging if JSON marshaling fails
		sl.logger.Printf("[%s] %s: %v", level, message, fields)
		if err != nil {
			sl.logger.Printf("Error: %v", err)
		}
		return
	}

	sl.logger.Println(string(jsonData))
}

// Debug logs a debug message.
func (sl *StructuredLogger) Debug(message string, fields map[string]interface{}) {
	sl.log(LogLevelDebug, message, fields, nil)
}

// Info logs an info message.
func (sl *StructuredLogger) Info(message string, fields map[string]interface{}) {
	sl.log(LogLevelInfo, message, fields, nil)
}

// Warn logs a warning message.
func (sl *StructuredLogger) Warn(message string, fields map[string]interface{}) {
	sl.log(LogLevelWarn, message, fields, nil)
}

// Error logs an error message.
func (sl *StructuredLogger) Error(message string, err error, fields map[string]interface{}) {
	if fields == nil {
		fields = make(map[string]interface{})
	}
	sl.log(LogLevelError, message, fields, err)
}

// WithRequest adds request context to log fields.
func WithRequest(method, path, requestID string) map[string]interface{} {
	return map[string]interface{}{
		"request_method": method,
		"request_path":   path,
		"request_id":     requestID,
	}
}

// WithUser adds user context to log fields.
func WithUser(userID string) map[string]interface{} {
	return map[string]interface{}{
		"user_id": userID,
	}
}

// WithDuration adds duration to log fields.
func WithDuration(duration time.Duration) map[string]interface{} {
	return map[string]interface{}{
		"duration_ms": duration.Milliseconds(),
	}
}

// ParseLogLevel parses a log level from string.
func ParseLogLevel(level string) LogLevel {
	switch level {
	case "DEBUG", "debug":
		return LogLevelDebug
	case "INFO", "info":
		return LogLevelInfo
	case "WARN", "warn":
		return LogLevelWarn
	case "ERROR", "error":
		return LogLevelError
	default:
		return LogLevelInfo
	}
}

// DefaultLogger returns a default structured logger.
func DefaultLogger() *StructuredLogger {
	level := ParseLogLevel(os.Getenv("LOG_LEVEL"))
	return NewStructuredLogger(level)
}

