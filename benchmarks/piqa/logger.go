package piqa

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"
)

// Logger provides structured logging for PIQA benchmark
type Logger struct {
	level      LogLevel
	format     LogFormat
	output     io.Writer
	mu         sync.Mutex
	fields     map[string]interface{}
	enableTime bool
}

// LogLevel represents logging severity
type LogLevel int

const (
	DebugLevel LogLevel = iota
	InfoLevel
	WarnLevel
	ErrorLevel
)

// LogFormat represents log output format
type LogFormat int

const (
	TextFormat LogFormat = iota
	JSONFormat
)

// LogEntry represents a single log entry
type LogEntry struct {
	Time    time.Time              `json:"time,omitempty"`
	Level   string                 `json:"level"`
	Message string                 `json:"message"`
	Fields  map[string]interface{} `json:"fields,omitempty"`
}

// NewLogger creates a new logger instance
func NewLogger(cfg LoggingConfig) *Logger {
	var output io.Writer = os.Stdout
	if cfg.Output == "file" && cfg.FilePath != "" {
		f, err := os.OpenFile(cfg.FilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			log.Printf("Failed to open log file: %v, using stdout", err)
		} else {
			output = f
		}
	}

	level := InfoLevel
	switch cfg.Level {
	case "debug":
		level = DebugLevel
	case "warn":
		level = WarnLevel
	case "error":
		level = ErrorLevel
	}

	format := TextFormat
	if cfg.Format == "json" {
		format = JSONFormat
	}

	return &Logger{
		level:      level,
		format:     format,
		output:     output,
		fields:     make(map[string]interface{}),
		enableTime: true,
	}
}

// WithFields returns a new logger with additional fields
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	newLogger := &Logger{
		level:      l.level,
		format:     l.format,
		output:     l.output,
		fields:     make(map[string]interface{}),
		enableTime: l.enableTime,
	}

	// Copy existing fields
	for k, v := range l.fields {
		newLogger.fields[k] = v
	}

	// Add new fields
	for k, v := range fields {
		newLogger.fields[k] = v
	}

	return newLogger
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, fields ...map[string]interface{}) {
	if l.level <= DebugLevel {
		l.log(DebugLevel, msg, fields...)
	}
}

// Info logs an info message
func (l *Logger) Info(msg string, fields ...map[string]interface{}) {
	if l.level <= InfoLevel {
		l.log(InfoLevel, msg, fields...)
	}
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, fields ...map[string]interface{}) {
	if l.level <= WarnLevel {
		l.log(WarnLevel, msg, fields...)
	}
}

// Error logs an error message
func (l *Logger) Error(msg string, fields ...map[string]interface{}) {
	if l.level <= ErrorLevel {
		l.log(ErrorLevel, msg, fields...)
	}
}

// log writes a log entry
func (l *Logger) log(level LogLevel, msg string, fields ...map[string]interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry := LogEntry{
		Level:   levelString(level),
		Message: msg,
		Fields:  make(map[string]interface{}),
	}

	if l.enableTime {
		entry.Time = time.Now()
	}

	// Add logger fields
	for k, v := range l.fields {
		entry.Fields[k] = v
	}

	// Add call-specific fields
	for _, f := range fields {
		for k, v := range f {
			entry.Fields[k] = v
		}
	}

	// Format and write
	var output string
	if l.format == JSONFormat {
		data, _ := json.Marshal(entry)
		output = string(data) + "\n"
	} else {
		output = l.formatText(entry)
	}

	l.output.Write([]byte(output))
}

// formatText formats log entry as text
func (l *Logger) formatText(entry LogEntry) string {
	var timeStr string
	if l.enableTime {
		timeStr = entry.Time.Format("2006-01-02 15:04:05") + " "
	}

	fieldsStr := ""
	if len(entry.Fields) > 0 {
		fieldsStr = " "
		for k, v := range entry.Fields {
			fieldsStr += fmt.Sprintf("%s=%v ", k, v)
		}
	}

	return fmt.Sprintf("%s[%s] %s%s\n", timeStr, entry.Level, entry.Message, fieldsStr)
}

// levelString converts LogLevel to string
func levelString(level LogLevel) string {
	switch level {
	case DebugLevel:
		return "DEBUG"
	case InfoLevel:
		return "INFO"
	case WarnLevel:
		return "WARN"
	case ErrorLevel:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Close closes the logger (if file-based)
func (l *Logger) Close() error {
	if closer, ok := l.output.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
