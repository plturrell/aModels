package observability

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.opentelemetry.io/otel/trace"
)

// TracerProvider wraps OpenTelemetry tracer provider
type TracerProvider struct {
	tp *sdktrace.TracerProvider
}

// InitTracing initializes OpenTelemetry tracing with support for multiple exporters and full attribute capture.
func InitTracing(serviceName string, logger *log.Logger) (*TracerProvider, error) {
	// Check if tracing is enabled
	if os.Getenv("OTEL_TRACES_ENABLED") != "true" {
		logger.Println("OpenTelemetry tracing disabled")
		return nil, nil
	}

	// Collect all exporters
	exporters := make([]sdktrace.SpanExporter, 0)

	// Add Jaeger exporter if enabled
	if shouldAddExporter("jaeger") {
		jaegerEndpoint := os.Getenv("JAEGER_ENDPOINT")
		if jaegerEndpoint == "" {
			jaegerEndpoint = "http://localhost:14268/api/traces"
		}
		exporter, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(jaegerEndpoint)))
		if err != nil {
			logger.Printf("failed to create Jaeger exporter: %v", err)
		} else {
			exporters = append(exporters, exporter)
			logger.Printf("Jaeger exporter enabled (endpoint: %s)", jaegerEndpoint)
		}
	}

	// Add OTLP HTTP exporter if enabled
	if shouldAddExporter("otlp") {
		otlpEndpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
		if otlpEndpoint == "" {
			otlpEndpoint = "http://localhost:4318"
		}
		exporter, err := otlptracehttp.New(context.Background(),
			otlptracehttp.WithEndpoint(otlpEndpoint),
		)
		if err != nil {
			logger.Printf("failed to create OTLP exporter: %v", err)
		} else {
			exporters = append(exporters, exporter)
			logger.Printf("OTLP exporter enabled (endpoint: %s)", otlpEndpoint)
		}
	}

	// Add file exporter if enabled
	if os.Getenv("OTEL_EXPORT_FILE_ENABLED") == "true" {
		filePath := os.Getenv("OTEL_EXPORT_FILE_PATH")
		if filePath == "" {
			filePath = "/app/data/traces"
		}
		// File exporter will be added via export manager in telemetry-exporter service
		// For now, log that it's configured
		logger.Printf("File export enabled (path: %s) - will be handled by telemetry-exporter service", filePath)
	}

	// Add Signavio exporter if enabled
	if os.Getenv("OTEL_EXPORT_SIGNAVIO_ENABLED") == "true" {
		// Signavio exporter will be added via export manager in telemetry-exporter service
		// For now, log that it's configured
		logger.Printf("Signavio export enabled - will be handled by telemetry-exporter service")
	}

	if len(exporters) == 0 {
		logger.Println("No exporters enabled, tracing disabled")
		return nil, nil
	}

	// Create multi-exporter if we have multiple exporters
	var exporter sdktrace.SpanExporter
	if len(exporters) == 1 {
		exporter = exporters[0]
	} else {
		exporter = &multiExporter{exporters: exporters, logger: logger}
	}

	// Create resource with enhanced attributes
	resAttrs := []attribute.KeyValue{
		semconv.ServiceNameKey.String(serviceName),
		semconv.ServiceVersionKey.String(getEnvOrDefault("SERVICE_VERSION", "1.0.0")),
		attribute.String("service.instance.id", getEnvOrDefault("SERVICE_INSTANCE_ID", generateInstanceID())),
		attribute.String("deployment.environment", getEnvOrDefault("DEPLOYMENT_ENVIRONMENT", "production")),
	}

	// Add agent framework type if specified
	if agentType := os.Getenv("AGENT_FRAMEWORK_TYPE"); agentType != "" {
		resAttrs = append(resAttrs, attribute.String("agent.framework.type", agentType))
	}

	res, err := resource.New(context.Background(), resource.WithAttributes(resAttrs...))
	if err != nil {
		return nil, err
	}

	// Create tracer provider with full attribute capture
	// Use AlwaysSample to capture all traces, or configure sampling as needed
	sampleRatio := 1.0
	if ratioStr := os.Getenv("OTEL_TRACES_SAMPLER_RATIO"); ratioStr != "" {
		if ratio, err := strconv.ParseFloat(ratioStr, 64); err == nil {
			sampleRatio = ratio
		}
	}

	var sampler sdktrace.Sampler
	if sampleRatio >= 1.0 {
		sampler = sdktrace.AlwaysSample() // Capture all traces
	} else {
		sampler = sdktrace.TraceIDRatioBased(sampleRatio)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
		// Enable full attribute capture - no limits
		sdktrace.WithRawSpanLimits(sdktrace.SpanLimits{
			AttributeValueLengthLimit:    -1, // No limit
			AttributeCountLimit:          -1, // No limit
			EventCountLimit:               -1, // No limit
			LinkCountLimit:                -1, // No limit
			EventAttributeCountLimit:      -1, // No limit
			LinkAttributeCountLimit:       -1, // No limit
		}),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	logger.Printf("OpenTelemetry tracing initialized (exporters=%d, service=%s, sample_ratio=%.2f)", len(exporters), serviceName, sampleRatio)

	return &TracerProvider{tp: tp}, nil
}

// Helper functions

func shouldAddExporter(exporterType string) bool {
	exporterEnv := os.Getenv("OTEL_EXPORTER_TYPE")
	if exporterEnv == "" {
		// Default behavior: enable jaeger if no explicit config
		return exporterType == "jaeger"
	}
	
	// Support comma-separated list of exporters
	exporters := strings.Split(exporterEnv, ",")
	for _, exp := range exporters {
		if strings.TrimSpace(exp) == exporterType {
			return true
		}
	}
	return false
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func generateInstanceID() string {
	hostname, _ := os.Hostname()
	if hostname != "" {
		return fmt.Sprintf("%s-%d", hostname, os.Getpid())
	}
	return fmt.Sprintf("instance-%d", os.Getpid())
}

// multiExporter exports to multiple exporters.
type multiExporter struct {
	exporters []sdktrace.SpanExporter
	logger    *log.Logger
}

// ExportSpans exports spans to all exporters.
func (me *multiExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadWriteSpan) error {
	var errs []error
	for _, exporter := range me.exporters {
		if err := exporter.ExportSpans(ctx, spans); err != nil {
			errs = append(errs, err)
			me.logger.Printf("Export failed: %v", err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("export errors: %v", errs)
	}
	return nil
}

// Shutdown shuts down all exporters.
func (me *multiExporter) Shutdown(ctx context.Context) error {
	var errs []error
	for _, exporter := range me.exporters {
		if err := exporter.Shutdown(ctx); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}
	return nil
}

// Shutdown shuts down the tracer provider
func (tp *TracerProvider) Shutdown(ctx context.Context) error {
	if tp == nil || tp.tp == nil {
		return nil
	}
	return tp.tp.Shutdown(ctx)
}

// StartSpan starts a new span
func StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	tracer := otel.Tracer("extract-service")
	return tracer.Start(ctx, name, opts...)
}

// AddSpanAttributes adds attributes to the current span
func AddSpanAttributes(ctx context.Context, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.SetAttributes(attrs...)
	}
}

// RecordError records an error in the current span
func RecordError(ctx context.Context, err error) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.RecordError(err)
	}
}

// RecordEvent records an event in the current span with full attributes.
func RecordEvent(ctx context.Context, name string, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.AddEvent(name, trace.WithAttributes(attrs...))
	}
}

// AddSpanLink adds a link to the current span.
func AddSpanLink(ctx context.Context, spanContext trace.SpanContext, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		// Note: Links can only be added when starting a span, not to existing spans
		// This is a helper for documentation - actual links should be added via SpanStartOption
	}
}

