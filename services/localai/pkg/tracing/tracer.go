package tracing

import (
	"context"
	"log"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	oteltrace "go.opentelemetry.io/otel/trace"
)

// InitTracer initializes OpenTelemetry tracing with Jaeger exporter
func InitTracer(serviceName, serviceVersion string) (oteltrace.Tracer, func(context.Context) error, error) {
	// Check if tracing is enabled
	tracingEnabled := os.Getenv("OTEL_TRACING_ENABLED")
	if tracingEnabled != "1" && tracingEnabled != "true" {
		log.Printf("📊 OpenTelemetry tracing disabled (set OTEL_TRACING_ENABLED=1 to enable)")
		return otel.Tracer(serviceName), func(ctx context.Context) error { return nil }, nil
	}

	// Get Jaeger endpoint from environment
	jaegerEndpoint := os.Getenv("OTEL_EXPORTER_JAEGER_ENDPOINT")
	if jaegerEndpoint == "" {
		jaegerEndpoint = "http://localhost:14268/api/traces"
	}

	// Create Jaeger exporter
	exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(jaegerEndpoint)))
	if err != nil {
		return nil, nil, err
	}

	// Create resource
	res, err := resource.New(context.Background(),
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion(serviceVersion),
		),
	)
	if err != nil {
		return nil, nil, err
	}

	// Create trace provider
	tp := trace.NewTracerProvider(
		trace.WithBatcher(exp),
		trace.WithResource(res),
	)

	otel.SetTracerProvider(tp)

	log.Printf("✅ OpenTelemetry tracing enabled (endpoint: %s)", jaegerEndpoint)

	// Return tracer and shutdown function
	tracer := tp.Tracer(serviceName)
	shutdown := func(ctx context.Context) error {
		return tp.Shutdown(ctx)
	}

	return tracer, shutdown, nil
}
