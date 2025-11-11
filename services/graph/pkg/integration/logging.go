// Package integration provides standardized error handling and logging utilities
// for cross-service integration in the lang infrastructure.
package integration

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
)

// CorrelationIDKey is the context key for correlation IDs.
// Correlation IDs are used to track requests across service boundaries.
type CorrelationIDKey struct{}

// GetCorrelationID retrieves the correlation ID from context, or generates a new one.
func GetCorrelationID(ctx context.Context) string {
	if id, ok := ctx.Value(CorrelationIDKey{}).(string); ok && id != "" {
		return id
	}
	return uuid.New().String()
}

// WithCorrelationID adds a correlation ID to the context.
func WithCorrelationID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, CorrelationIDKey{}, id)
}

// WithNewCorrelationID generates and adds a new correlation ID to the context.
func WithNewCorrelationID(ctx context.Context) context.Context {
	return WithCorrelationID(ctx, uuid.New().String())
}

// LoggedOperation represents a logged operation with timing and correlation ID.
type LoggedOperation struct {
	CorrelationID string
	OperationName string
	StartTime     time.Time
	Logger        *log.Logger
}

// StartOperation starts a logged operation and returns a LoggedOperation.
func StartOperation(ctx context.Context, logger *log.Logger, operationName string) *LoggedOperation {
	correlationID := GetCorrelationID(ctx)
	if logger != nil {
		logger.Printf("[START] %s [correlation_id=%s]", operationName, correlationID)
	}
	return &LoggedOperation{
		CorrelationID: correlationID,
		OperationName: operationName,
		StartTime:     time.Now(),
		Logger:        logger,
	}
}

// End logs the end of an operation with duration.
func (op *LoggedOperation) End(err error) {
	duration := time.Since(op.StartTime)
	if err != nil {
		if op.Logger != nil {
			op.Logger.Printf("[ERROR] %s [correlation_id=%s, duration=%v, error=%v]", 
				op.OperationName, op.CorrelationID, duration, err)
		}
	} else {
		if op.Logger != nil {
			op.Logger.Printf("[END] %s [correlation_id=%s, duration=%v]", 
				op.OperationName, op.CorrelationID, duration)
		}
	}
}

// Log logs a message with correlation ID.
func (op *LoggedOperation) Log(format string, args ...interface{}) {
	if op.Logger != nil {
		message := fmt.Sprintf(format, args...)
		op.Logger.Printf("[LOG] %s [correlation_id=%s] %s", 
			op.OperationName, op.CorrelationID, message)
	}
}

// LogHTTPRequest logs an HTTP request with correlation ID.
func LogHTTPRequest(
	ctx context.Context,
	logger *log.Logger,
	method, url string,
	statusCode int,
	duration time.Duration,
	err error,
) {
	correlationID := GetCorrelationID(ctx)
	if err != nil {
		logger.Printf("[HTTP_ERROR] %s %s [correlation_id=%s, status=%d, duration=%v, error=%v]",
			method, url, correlationID, statusCode, duration, err)
	} else {
		logger.Printf("[HTTP] %s %s [correlation_id=%s, status=%d, duration=%v]",
			method, url, correlationID, statusCode, duration)
	}
}

// LogIntegrationCall logs an integration call between services.
func LogIntegrationCall(
	ctx context.Context,
	logger *log.Logger,
	fromService, toService, operation string,
	duration time.Duration,
	err error,
) {
	correlationID := GetCorrelationID(ctx)
	if err != nil {
		if logger != nil {
			logger.Printf("[INTEGRATION_ERROR] %s -> %s:%s [correlation_id=%s, duration=%v, error=%v]",
				fromService, toService, operation, correlationID, duration, err)
		}
	} else {
		if logger != nil {
			logger.Printf("[INTEGRATION] %s -> %s:%s [correlation_id=%s, duration=%v]",
				fromService, toService, operation, correlationID, duration)
		}
	}
}

