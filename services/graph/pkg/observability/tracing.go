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
	"github.com/plturrell/aModels/pkg/observability/llm"
)

// TracerProvider wraps OpenTelemetry tracer provider
type TracerProvider struct {
	tp *sdktrace.TracerProvider
}

// InitTracing initializes OpenTelemetry tracing for Graph service (LangGraph workflows).
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

	// Create resource with enhanced attributes for LangGraph
	resAttrs := []attribute.KeyValue{
		semconv.ServiceNameKey.String(serviceName),
		semconv.ServiceVersionKey.String(getEnvOrDefault("SERVICE_VERSION", "1.0.0")),
		attribute.String("service.instance.id", getEnvOrDefault("SERVICE_INSTANCE_ID", generateInstanceID())),
		attribute.String("deployment.environment", getEnvOrDefault("DEPLOYMENT_ENVIRONMENT", "production")),
		attribute.String("agent.framework.type", "langgraph"), // LangGraph framework
	}

	res, err := resource.New(context.Background(), resource.WithAttributes(resAttrs...))
	if err != nil {
		return nil, err
	}

	// Create tracer provider with full attribute capture
	sampleRatio := 1.0
	if ratioStr := os.Getenv("OTEL_TRACES_SAMPLER_RATIO"); ratioStr != "" {
		if ratio, err := strconv.ParseFloat(ratioStr, 64); err == nil {
			sampleRatio = ratio
		}
	}

	var sampler sdktrace.Sampler
	if sampleRatio >= 1.0 {
		sampler = sdktrace.AlwaysSample()
	} else {
		sampler = sdktrace.TraceIDRatioBased(sampleRatio)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
		sdktrace.WithRawSpanLimits(sdktrace.SpanLimits{
			AttributeValueLengthLimit:    -1,
			AttributeCountLimit:          -1,
			EventCountLimit:               -1,
			LinkCountLimit:                -1,
			EventAttributeCountLimit:      -1,
			LinkAttributeCountLimit:       -1,
		}),
	)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	logger.Printf("OpenTelemetry tracing initialized for Graph service (exporters=%d, sample_ratio=%.2f)", len(exporters), sampleRatio)

	return &TracerProvider{tp: tp}, nil
}

// Shutdown shuts down the tracer provider
func (tp *TracerProvider) Shutdown(ctx context.Context) error {
	if tp == nil || tp.tp == nil {
		return nil
	}
	return tp.tp.Shutdown(ctx)
}

// StartSpan starts a new span for LangGraph workflow execution
func StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	tracer := otel.Tracer("graph-service")
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

// RecordEvent records an event in the current span
func RecordEvent(ctx context.Context, name string, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.AddEvent(name, trace.WithAttributes(attrs...))
	}
}

// Helper functions

func shouldAddExporter(exporterType string) bool {
	exporterEnv := os.Getenv("OTEL_EXPORTER_TYPE")
	if exporterEnv == "" {
		return exporterType == "jaeger"
	}
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

// multiExporter exports to multiple exporters
type multiExporter struct {
	exporters []sdktrace.SpanExporter
	logger    *log.Logger
}

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

// AddLLMRequestAttributes adds OpenLLMetry request attributes to the current span
func AddLLMRequestAttributes(ctx context.Context, config llm.LLMRequestConfig) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		llm.AddLLMRequestAttributes(span, config)
	}
}

// AddLLMResponseAttributes adds OpenLLMetry response attributes to the current span
func AddLLMResponseAttributes(ctx context.Context, info llm.LLMResponseInfo) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		llm.AddLLMResponseAttributes(span, info)
	}
}

// AddLLMAttributes adds both OpenLLMetry request and response attributes to the current span
func AddLLMAttributes(ctx context.Context, config llm.LLMRequestConfig, info llm.LLMResponseInfo) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		llm.AddLLMAttributes(span, config, info)
	}
}

