package observability

import (
	"context"
	"log"
	"os"

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

// InitTracing initializes OpenTelemetry tracing
func InitTracing(serviceName string, logger *log.Logger) (*TracerProvider, error) {
	// Check if tracing is enabled
	if os.Getenv("OTEL_TRACES_ENABLED") != "true" {
		logger.Println("OpenTelemetry tracing disabled")
		return nil, nil
	}

	exporterType := os.Getenv("OTEL_EXPORTER_TYPE")
	if exporterType == "" {
		exporterType = "jaeger" // Default to Jaeger
	}

	var exporter sdktrace.SpanExporter
	var err error

	switch exporterType {
	case "jaeger":
		jaegerEndpoint := os.Getenv("JAEGER_ENDPOINT")
		if jaegerEndpoint == "" {
			jaegerEndpoint = "http://localhost:14268/api/traces"
		}
		// Jaeger exporter is deprecated, but still functional
		// In production, consider migrating to OTLP
		exporter, err = jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(jaegerEndpoint)))
		if err != nil {
			logger.Printf("failed to create Jaeger exporter: %v, falling back to no-op", err)
			return nil, nil
		}
	case "otlp":
		otlpEndpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
		if otlpEndpoint == "" {
			otlpEndpoint = "http://localhost:4318"
		}
		exporter, err = otlptracehttp.New(context.Background(),
			otlptracehttp.WithEndpoint(otlpEndpoint),
		)
		if err != nil {
			return nil, err
		}
	default:
		logger.Printf("Unknown exporter type: %s, tracing disabled", exporterType)
		return nil, nil
	}

	// Create resource
	res, err := resource.New(context.Background(),
		resource.WithAttributes(
			semconv.ServiceNameKey.String(serviceName),
			semconv.ServiceVersionKey.String("1.0.0"),
		),
	)
	if err != nil {
		return nil, err
	}

	// Create tracer provider
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	logger.Printf("OpenTelemetry tracing initialized (exporter=%s, service=%s)", exporterType, serviceName)

	return &TracerProvider{tp: tp}, nil
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

