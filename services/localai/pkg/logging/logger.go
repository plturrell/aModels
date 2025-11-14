package logging

import (
	"context"
	"io"
	"os"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// Logger provides structured logging with contextual information
type Logger struct {
	logger zerolog.Logger
}

// Config holds logger configuration
type Config struct {
	Level      string // debug, info, warn, error
	Format     string // json, console
	TimeFormat string // rfc3339, unix, etc.
	Output     io.Writer
}

// DefaultConfig returns sensible defaults for production
func DefaultConfig() *Config {
	return &Config{
		Level:      "info",
		Format:     "json",
		TimeFormat: "rfc3339",
		Output:     os.Stdout,
	}
}

// NewLogger creates a new structured logger
func NewLogger(cfg *Config) *Logger {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	// Set global time format
	zerolog.TimeFieldFormat = time.RFC3339
	if cfg.TimeFormat == "unix" {
		zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	}

	// Configure output format
	var output io.Writer = cfg.Output
	if cfg.Format == "console" {
		output = zerolog.ConsoleWriter{
			Out:        cfg.Output,
			TimeFormat: time.RFC3339,
			NoColor:    os.Getenv("NO_COLOR") == "1",
		}
	}

	// Set log level
	level := zerolog.InfoLevel
	switch cfg.Level {
	case "debug":
		level = zerolog.DebugLevel
	case "info":
		level = zerolog.InfoLevel
	case "warn":
		level = zerolog.WarnLevel
	case "error":
		level = zerolog.ErrorLevel
	}

	logger := zerolog.New(output).
		Level(level).
		With().
		Timestamp().
		Caller().
		Logger()

	return &Logger{logger: logger}
}

// InitGlobalLogger initializes the global logger from environment variables
func InitGlobalLogger() *Logger {
	cfg := &Config{
		Level:      getEnv("LOG_LEVEL", "info"),
		Format:     getEnv("LOG_FORMAT", "json"),
		TimeFormat: getEnv("LOG_TIME_FORMAT", "rfc3339"),
		Output:     os.Stdout,
	}

	logger := NewLogger(cfg)
	log.Logger = logger.logger
	return logger
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// WithContext returns a logger with trace and span IDs from context
func (l *Logger) WithContext(ctx context.Context) *Logger {
	logger := l.logger
	
	// Extract trace ID if available
	if traceID := ctx.Value("trace_id"); traceID != nil {
		logger = logger.With().Str("trace_id", traceID.(string)).Logger()
	}
	
	// Extract span ID if available
	if spanID := ctx.Value("span_id"); spanID != nil {
		logger = logger.With().Str("span_id", spanID.(string)).Logger()
	}
	
	return &Logger{logger: logger}
}

// WithFields returns a logger with additional fields
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	logger := l.logger
	for key, value := range fields {
		logger = logger.With().Interface(key, value).Logger()
	}
	return &Logger{logger: logger}
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, fields ...map[string]interface{}) {
	event := l.logger.Debug()
	for _, f := range fields {
		for k, v := range f {
			event = event.Interface(k, v)
		}
	}
	event.Msg(msg)
}

// Info logs an info message
func (l *Logger) Info(msg string, fields ...map[string]interface{}) {
	event := l.logger.Info()
	for _, f := range fields {
		for k, v := range f {
			event = event.Interface(k, v)
		}
	}
	event.Msg(msg)
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, fields ...map[string]interface{}) {
	event := l.logger.Warn()
	for _, f := range fields {
		for k, v := range f {
			event = event.Interface(k, v)
		}
	}
	event.Msg(msg)
}

// Error logs an error message
func (l *Logger) Error(msg string, err error, fields ...map[string]interface{}) {
	event := l.logger.Error()
	if err != nil {
		event = event.Err(err)
	}
	for _, f := range fields {
		for k, v := range f {
			event = event.Interface(k, v)
		}
	}
	event.Msg(msg)
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(msg string, err error, fields ...map[string]interface{}) {
	event := l.logger.Fatal()
	if err != nil {
		event = event.Err(err)
	}
	for _, f := range fields {
		for k, v := range f {
			event = event.Interface(k, v)
		}
	}
	event.Msg(msg)
}

// HTTP request logging
func (l *Logger) LogHTTPRequest(method, path string, statusCode, latencyMs int, fields map[string]interface{}) {
	event := l.logger.Info().
		Str("method", method).
		Str("path", path).
		Int("status_code", statusCode).
		Int("latency_ms", latencyMs)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("http_request")
}

// Model inference logging
func (l *Logger) LogInference(model, domain string, tokensUsed, latencyMs int, cacheHit bool, fields map[string]interface{}) {
	event := l.logger.Info().
		Str("model", model).
		Str("domain", domain).
		Int("tokens_used", tokensUsed).
		Int("latency_ms", latencyMs).
		Bool("cache_hit", cacheHit)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("inference")
}

// Cache operation logging
func (l *Logger) LogCacheOp(operation, key string, hit bool, latencyMs int, fields map[string]interface{}) {
	event := l.logger.Debug().
		Str("operation", operation).
		Str("key", key).
		Bool("hit", hit).
		Int("latency_ms", latencyMs)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("cache_operation")
}

// Database operation logging
func (l *Logger) LogDBOp(operation, table string, rowsAffected int, latencyMs int, fields map[string]interface{}) {
	event := l.logger.Debug().
		Str("operation", operation).
		Str("table", table).
		Int("rows_affected", rowsAffected).
		Int("latency_ms", latencyMs)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("database_operation")
}

// Model loading logging
func (l *Logger) LogModelLoad(model, domain string, memoryMB int64, latencyMs int, success bool, fields map[string]interface{}) {
	event := l.logger.Info().
		Str("model", model).
		Str("domain", domain).
		Int64("memory_mb", memoryMB).
		Int("latency_ms", latencyMs).
		Bool("success", success)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("model_load")
}
