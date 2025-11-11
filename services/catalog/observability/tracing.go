package observability

import (
	"context"
	"fmt"
	"time"
)

// TraceID represents a trace identifier.
type TraceID string

// SpanID represents a span identifier.
type SpanID string

// Span represents a tracing span.
type Span struct {
	TraceID   TraceID
	SpanID    SpanID
	ParentID  SpanID
	Operation string
	StartTime time.Time
	EndTime   time.Time
	Tags      map[string]string
	Logs      []SpanLog
}

// SpanLog represents a log entry in a span.
type SpanLog struct {
	Timestamp time.Time
	Message   string
	Fields    map[string]interface{}
}

// Tracer provides distributed tracing capabilities.
type Tracer struct {
	// In production, would integrate with OpenTelemetry
	// For now, simple in-memory tracing
	spans map[SpanID]*Span
}

// NewTracer creates a new tracer.
func NewTracer() *Tracer {
	return &Tracer{
		spans: make(map[SpanID]*Span),
	}
}

// StartSpan starts a new span.
func (t *Tracer) StartSpan(ctx context.Context, operation string) (context.Context, *Span) {
	traceID := TraceID(extractTraceID(ctx))
	if traceID == "" {
		traceID = TraceID(generateID())
	}

	parentID := extractSpanID(ctx)
	spanID := SpanID(generateID())

	span := &Span{
		TraceID:   traceID,
		SpanID:    spanID,
		ParentID:  parentID,
		Operation: operation,
		StartTime: time.Now(),
		Tags:      make(map[string]string),
		Logs:      []SpanLog{},
	}

	t.spans[spanID] = span

	// Add to context
	ctx = context.WithValue(ctx, "trace_id", traceID)
	ctx = context.WithValue(ctx, "span_id", spanID)

	return ctx, span
}

// EndSpan ends a span.
func (t *Tracer) EndSpan(spanID SpanID) {
	if span, ok := t.spans[spanID]; ok {
		span.EndTime = time.Now()
		// In production, would export to OpenTelemetry collector
		// For now, just log the span
		duration := span.EndTime.Sub(span.StartTime)
		fmt.Printf("[TRACE] %s %s duration=%v\n", span.TraceID, span.Operation, duration)
	}
}

// AddTag adds a tag to a span.
func (s *Span) AddTag(key, value string) {
	if s.Tags == nil {
		s.Tags = make(map[string]string)
	}
	s.Tags[key] = value
}

// AddLog adds a log entry to a span.
func (s *Span) AddLog(message string, fields map[string]interface{}) {
	s.Logs = append(s.Logs, SpanLog{
		Timestamp: time.Now(),
		Message:   message,
		Fields:    fields,
	})
}

// extractTraceID extracts trace ID from context.
func extractTraceID(ctx context.Context) string {
	if traceID, ok := ctx.Value("trace_id").(TraceID); ok {
		return string(traceID)
	}
	return ""
}

// extractSpanID extracts span ID from context.
func extractSpanID(ctx context.Context) SpanID {
	if spanID, ok := ctx.Value("span_id").(SpanID); ok {
		return spanID
	}
	return ""
}

// generateID generates a unique ID.
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// GetTraceID extracts trace ID from context.
func GetTraceID(ctx context.Context) string {
	return extractTraceID(ctx)
}

// GetSpanID extracts span ID from context.
func GetSpanID(ctx context.Context) string {
	if spanID, ok := ctx.Value("span_id").(SpanID); ok {
		return string(spanID)
	}
	return ""
}

