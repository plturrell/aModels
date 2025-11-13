package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace"
	oteltrace "go.opentelemetry.io/otel/trace"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	resourcepb "go.opentelemetry.io/proto/otlp/resource/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"
)

// FileExporter exports traces to files in both JSON and Protobuf formats.
type FileExporter struct {
	jsonPath     string
	protobufPath string
	mu           sync.Mutex
	jsonFile     *os.File
	protobufFile *os.File
	encoder      *json.Encoder
	maxFileSize  int64
	maxFiles     int
	currentSize  int64
	fileCounter  int
	logger       func(string, ...interface{})
}

// FileExporterConfig configures the file exporter.
type FileExporterConfig struct {
	BasePath    string // Base directory for trace files
	MaxFileSize int64  // Maximum file size in bytes before rotation (default: 100MB)
	MaxFiles    int    // Maximum number of files to keep (default: 10)
	Logger      func(string, ...interface{})
}

// NewFileExporter creates a new file exporter.
func NewFileExporter(cfg FileExporterConfig) (*FileExporter, error) {
	if cfg.BasePath == "" {
		return nil, fmt.Errorf("base path is required")
	}
	if cfg.MaxFileSize == 0 {
		cfg.MaxFileSize = 100 * 1024 * 1024 // 100MB default
	}
	if cfg.MaxFiles == 0 {
		cfg.MaxFiles = 10
	}
	if cfg.Logger == nil {
		cfg.Logger = func(string, ...interface{}) {} // no-op logger
	}

	// Ensure directory exists
	if err := os.MkdirAll(cfg.BasePath, 0755); err != nil {
		return nil, fmt.Errorf("create directory: %w", err)
	}

	fe := &FileExporter{
		jsonPath:     filepath.Join(cfg.BasePath, "traces.jsonl"),
		protobufPath: filepath.Join(cfg.BasePath, "traces.pb"),
		maxFileSize:  cfg.MaxFileSize,
		maxFiles:     cfg.MaxFiles,
		logger:       cfg.Logger,
	}

	// Open initial files
	if err := fe.openFiles(); err != nil {
		return nil, err
	}

	return fe, nil
}

// openFiles opens or rotates the export files.
func (fe *FileExporter) openFiles() error {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	// Close existing files
	if fe.jsonFile != nil {
		fe.jsonFile.Close()
	}
	if fe.protobufFile != nil {
		fe.protobufFile.Close()
	}

	// Check if rotation is needed
	if fe.currentSize >= fe.maxFileSize {
		fe.fileCounter++
		fe.currentSize = 0
		timestamp := time.Now().Format("20060102-150405")
		baseDir := filepath.Dir(fe.jsonPath)
		fe.jsonPath = filepath.Join(baseDir, fmt.Sprintf("traces-%s-%d.jsonl", timestamp, fe.fileCounter))
		fe.protobufPath = filepath.Join(baseDir, fmt.Sprintf("traces-%s-%d.pb", timestamp, fe.fileCounter))
		
		// Clean up old files
		fe.cleanupOldFiles()
	}

	// Open JSON file (append mode)
	jsonFile, err := os.OpenFile(fe.jsonPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open JSON file: %w", err)
	}
	fe.jsonFile = jsonFile
	fe.encoder = json.NewEncoder(jsonFile)

	// Open Protobuf file (append mode)
	pbFile, err := os.OpenFile(fe.protobufPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open Protobuf file: %w", err)
	}
	fe.protobufFile = pbFile

	// Get current file size
	if info, err := jsonFile.Stat(); err == nil {
		fe.currentSize = info.Size()
	}

	return nil
}

// cleanupOldFiles removes old trace files beyond the max limit.
func (fe *FileExporter) cleanupOldFiles() {
	baseDir := filepath.Dir(fe.jsonPath)
	pattern := filepath.Join(baseDir, "traces-*.jsonl")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fe.logger("failed to glob trace files: %v", err)
		return
	}

	if len(matches) > fe.maxFiles {
		// Sort by modification time and remove oldest
		// For simplicity, just remove files beyond max
		for i := 0; i < len(matches)-fe.maxFiles; i++ {
			os.Remove(matches[i])
			// Also remove corresponding .pb file
			pbFile := matches[i][:len(matches[i])-6] + ".pb"
			os.Remove(pbFile)
		}
	}
}

// ExportTraces exports traces to files.
func (fe *FileExporter) ExportTraces(ctx context.Context, traces *coltracepb.ExportTraceServiceRequest) error {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	// Check if rotation is needed
	if fe.currentSize >= fe.maxFileSize {
		fe.mu.Unlock()
		if err := fe.openFiles(); err != nil {
			return err
		}
		fe.mu.Lock()
	}

	// Export as JSON Lines
	jsonData, err := json.Marshal(traces)
	if err != nil {
		return fmt.Errorf("marshal JSON: %w", err)
	}
	if _, err := fe.jsonFile.Write(jsonData); err != nil {
		return fmt.Errorf("write JSON: %w", err)
	}
	if _, err := fe.jsonFile.WriteString("\n"); err != nil {
		return fmt.Errorf("write newline: %w", err)
	}
	fe.currentSize += int64(len(jsonData) + 1)

	// Export as Protobuf
	pbData, err := proto.Marshal(traces)
	if err != nil {
		return fmt.Errorf("marshal Protobuf: %w", err)
	}
	// Write length prefix for protobuf streaming format
	lengthBytes := make([]byte, 4)
	lengthBytes[0] = byte(len(pbData) >> 24)
	lengthBytes[1] = byte(len(pbData) >> 16)
	lengthBytes[2] = byte(len(pbData) >> 8)
	lengthBytes[3] = byte(len(pbData))
	if _, err := fe.protobufFile.Write(lengthBytes); err != nil {
		return fmt.Errorf("write length: %w", err)
	}
	if _, err := fe.protobufFile.Write(pbData); err != nil {
		return fmt.Errorf("write Protobuf: %w", err)
	}

	return nil
}

// Shutdown closes the file exporter.
func (fe *FileExporter) Shutdown(ctx context.Context) error {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	var errs []error
	if fe.jsonFile != nil {
		if err := fe.jsonFile.Close(); err != nil {
			errs = append(errs, err)
		}
		fe.jsonFile = nil
	}
	if fe.protobufFile != nil {
		if err := fe.protobufFile.Close(); err != nil {
			errs = append(errs, err)
		}
		fe.protobufFile = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}
	return nil
}

// ToSpanExporter converts FileExporter to trace.SpanExporter.
func (fe *FileExporter) ToSpanExporter() trace.SpanExporter {
	return &fileSpanExporter{fe: fe}
}

// fileSpanExporter adapts FileExporter to trace.SpanExporter interface.
type fileSpanExporter struct {
	fe *FileExporter
}

// ExportSpans exports spans to files.
func (fse *fileSpanExporter) ExportSpans(ctx context.Context, spans []trace.ReadOnlySpan) error {
	// Convert spans to OTLP format
	request := convertSpansToOTLP(spans)
	return fse.fe.ExportTraces(ctx, request)
}

// Shutdown shuts down the exporter.
func (fse *fileSpanExporter) Shutdown(ctx context.Context) error {
	return fse.fe.Shutdown(ctx)
}

// convertSpansToOTLP converts OpenTelemetry SDK spans to OTLP format.
func convertSpansToOTLP(spans []trace.ReadOnlySpan) *coltracepb.ExportTraceServiceRequest {
	// Group spans by resource
	resourceSpansMap := make(map[string]*tracepb.ResourceSpans)
	
	for _, span := range spans {
		// Get resource
		resource := span.Resource()
		resourceKey := resourceKey(resource.Attributes())
		
		rs, exists := resourceSpansMap[resourceKey]
		if !exists {
			rs = &tracepb.ResourceSpans{
				Resource: &resourcepb.Resource{
					Attributes: attributesToKeyValue(resource.Attributes()),
				},
				ScopeSpans: []*tracepb.ScopeSpans{},
			}
			resourceSpansMap[resourceKey] = rs
		}

		// Convert span to OTLP
		otlpSpan := spanToOTLP(span)
		
		// Find or create scope spans
		scope := span.InstrumentationScope()
		scopeName := scope.Name
		scopeVersion := scope.Version
		var scopeSpans *tracepb.ScopeSpans
		for _, ss := range rs.ScopeSpans {
			if ss.Scope != nil && ss.Scope.Name == scopeName && ss.Scope.Version == scopeVersion {
				scopeSpans = ss
				break
			}
		}
		if scopeSpans == nil {
			scopeSpans = &tracepb.ScopeSpans{
				Scope: &commonpb.InstrumentationScope{
					Name:    scopeName,
					Version: scopeVersion,
				},
				Spans: []*tracepb.Span{},
			}
			rs.ScopeSpans = append(rs.ScopeSpans, scopeSpans)
		}

		scopeSpans.Spans = append(scopeSpans.Spans, otlpSpan)
	}

	// Convert map to slice
	resourceSpans := make([]*tracepb.ResourceSpans, 0, len(resourceSpansMap))
	for _, rs := range resourceSpansMap {
		resourceSpans = append(resourceSpans, rs)
	}

	return &coltracepb.ExportTraceServiceRequest{
		ResourceSpans: resourceSpans,
	}
}

// Helper functions to convert OpenTelemetry types to OTLP protobuf types

func resourceKey(attrs []attribute.KeyValue) string {
	// Create a simple key from resource attributes
	// In production, use a proper hash or serialization
	key := ""
	for _, attr := range attrs {
		key += string(attr.Key) + "=" + attr.Value.AsString() + ";"
	}
	return key
}

func attributesToKeyValue(attrs []attribute.KeyValue) []*commonpb.KeyValue {
	result := make([]*commonpb.KeyValue, 0, len(attrs))
	for _, attr := range attrs {
		kv := &commonpb.KeyValue{
			Key: string(attr.Key),
		}
		kv.Value = attributeValueToAnyValue(attr.Value)
		result = append(result, kv)
	}
	return result
}

func attributeValueToAnyValue(v attribute.Value) *commonpb.AnyValue {
	switch v.Type() {
	case attribute.STRING:
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: v.AsString()}}
	case attribute.INT64:
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_IntValue{IntValue: v.AsInt64()}}
	case attribute.FLOAT64:
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_DoubleValue{DoubleValue: v.AsFloat64()}}
	case attribute.BOOL:
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_BoolValue{BoolValue: v.AsBool()}}
	case attribute.STRINGSLICE:
		values := v.AsStringSlice()
		arrayValues := make([]*commonpb.AnyValue, len(values))
		for i, s := range values {
			arrayValues[i] = &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: s}}
		}
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_ArrayValue{ArrayValue: &commonpb.ArrayValue{Values: arrayValues}}}
	case attribute.INT64SLICE:
		values := v.AsInt64Slice()
		arrayValues := make([]*commonpb.AnyValue, len(values))
		for i, n := range values {
			arrayValues[i] = &commonpb.AnyValue{Value: &commonpb.AnyValue_IntValue{IntValue: n}}
		}
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_ArrayValue{ArrayValue: &commonpb.ArrayValue{Values: arrayValues}}}
	case attribute.FLOAT64SLICE:
		values := v.AsFloat64Slice()
		arrayValues := make([]*commonpb.AnyValue, len(values))
		for i, f := range values {
			arrayValues[i] = &commonpb.AnyValue{Value: &commonpb.AnyValue_DoubleValue{DoubleValue: f}}
		}
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_ArrayValue{ArrayValue: &commonpb.ArrayValue{Values: arrayValues}}}
	case attribute.BOOLSLICE:
		values := v.AsBoolSlice()
		arrayValues := make([]*commonpb.AnyValue, len(values))
		for i, b := range values {
			arrayValues[i] = &commonpb.AnyValue{Value: &commonpb.AnyValue_BoolValue{BoolValue: b}}
		}
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_ArrayValue{ArrayValue: &commonpb.ArrayValue{Values: arrayValues}}}
	default:
		// Fallback to string representation
		return &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: v.AsString()}}
	}
}

func spanToOTLP(span trace.ReadOnlySpan) *tracepb.Span {
	spanContext := span.SpanContext()
	
	// Convert TraceID and SpanID to bytes
	traceID := spanContext.TraceID()
	spanID := spanContext.SpanID()
	
	otlpSpan := &tracepb.Span{
		TraceId:           traceID[:],
		SpanId:            spanID[:],
		Name:              span.Name(),
		Kind:              spanKindToOTLP(span.SpanKind()),
		StartTimeUnixNano: uint64(span.StartTime().UnixNano()),
		EndTimeUnixNano:   uint64(span.EndTime().UnixNano()),
		Attributes:        attributesToKeyValue(span.Attributes()),
		Events:            eventsToOTLP(span.Events()),
		Links:             linksToOTLP(span.Links()),
		Status:            statusToOTLP(span.Status()),
	}

	// Note: Parent span ID is typically extracted from context during span creation
	// and is not directly available from ReadOnlySpan. Parent relationships
	// are preserved through trace context propagation.

	return otlpSpan
}

func spanKindToOTLP(kind oteltrace.SpanKind) tracepb.Span_SpanKind {
	switch kind {
	case oteltrace.SpanKindInternal:
		return tracepb.Span_SPAN_KIND_INTERNAL
	case oteltrace.SpanKindServer:
		return tracepb.Span_SPAN_KIND_SERVER
	case oteltrace.SpanKindClient:
		return tracepb.Span_SPAN_KIND_CLIENT
	case oteltrace.SpanKindProducer:
		return tracepb.Span_SPAN_KIND_PRODUCER
	case oteltrace.SpanKindConsumer:
		return tracepb.Span_SPAN_KIND_CONSUMER
	default:
		return tracepb.Span_SPAN_KIND_UNSPECIFIED
	}
}

func eventsToOTLP(events []trace.Event) []*tracepb.Span_Event {
	result := make([]*tracepb.Span_Event, len(events))
	for i, event := range events {
		result[i] = &tracepb.Span_Event{
			TimeUnixNano: uint64(event.Time.UnixNano()),
			Name:         event.Name,
			Attributes:   attributesToKeyValue(event.Attributes),
		}
	}
	return result
}

func linksToOTLP(links []trace.Link) []*tracepb.Span_Link {
	result := make([]*tracepb.Span_Link, len(links))
	for i, link := range links {
		traceID := link.SpanContext.TraceID()
		spanID := link.SpanContext.SpanID()
		result[i] = &tracepb.Span_Link{
			TraceId:    traceID[:],
			SpanId:     spanID[:],
			Attributes: attributesToKeyValue(link.Attributes),
		}
	}
	return result
}

func statusToOTLP(status trace.Status) *tracepb.Status {
	otlpStatus := &tracepb.Status{
		Code: statusCodeToOTLP(status.Code),
	}
	if status.Description != "" {
		otlpStatus.Message = status.Description
	}
	return otlpStatus
}

func statusCodeToOTLP(code codes.Code) tracepb.Status_StatusCode {
	switch code {
	case codes.Ok:
		return tracepb.Status_STATUS_CODE_OK
	case codes.Error:
		return tracepb.Status_STATUS_CODE_ERROR
	default:
		return tracepb.Status_STATUS_CODE_UNSET
	}
}

// ConvertSpansToOTLP is exported for use by other exporters.
func ConvertSpansToOTLP(spans []trace.ReadOnlySpan) *coltracepb.ExportTraceServiceRequest {
	return convertSpansToOTLP(spans)
}
