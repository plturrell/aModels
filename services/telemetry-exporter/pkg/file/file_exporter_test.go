package file

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	resourcepb "go.opentelemetry.io/proto/otlp/resource/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
)

func TestFileExporter(t *testing.T) {
	// Create temporary directory
	tmpDir := t.TempDir()

	cfg := FileExporterConfig{
		BasePath:    tmpDir,
		MaxFileSize: 1024 * 1024, // 1MB
		MaxFiles:    5,
		Logger:      func(string, ...interface{}) {},
	}

	exporter, err := NewFileExporter(cfg)
	if err != nil {
		t.Fatalf("Failed to create file exporter: %v", err)
	}
	defer exporter.Shutdown(context.Background())

	// Create a test trace request
	request := &coltracepb.ExportTraceServiceRequest{
		ResourceSpans: []*tracepb.ResourceSpans{
			{
				Resource: &resourcepb.Resource{
					Attributes: []*commonpb.KeyValue{},
				},
				ScopeSpans: []*tracepb.ScopeSpans{},
			},
		},
	}

	// Export traces
	err = exporter.ExportTraces(context.Background(), request)
	if err != nil {
		t.Fatalf("Failed to export traces: %v", err)
	}

	// Check that files were created
	jsonFile := filepath.Join(tmpDir, "traces.jsonl")
	pbFile := filepath.Join(tmpDir, "traces.pb")

	if _, err := os.Stat(jsonFile); os.IsNotExist(err) {
		t.Errorf("JSON file was not created: %v", err)
	}

	if _, err := os.Stat(pbFile); os.IsNotExist(err) {
		t.Errorf("Protobuf file was not created: %v", err)
	}
}

func TestFileExporterRotation(t *testing.T) {
	// Create temporary directory
	tmpDir := t.TempDir()

	cfg := FileExporterConfig{
		BasePath:    tmpDir,
		MaxFileSize: 100, // Very small to trigger rotation
		MaxFiles:    3,
		Logger:      func(string, ...interface{}) {},
	}

	exporter, err := NewFileExporter(cfg)
	if err != nil {
		t.Fatalf("Failed to create file exporter: %v", err)
	}
	defer exporter.Shutdown(context.Background())

	// Export multiple times to trigger rotation
	for i := 0; i < 5; i++ {
		request := &coltracepb.ExportTraceServiceRequest{
			ResourceSpans: []*tracepb.ResourceSpans{},
		}
		err = exporter.ExportTraces(context.Background(), request)
		if err != nil {
			t.Fatalf("Failed to export traces: %v", err)
		}
	}

	// Check that rotation occurred
	matches, err := filepath.Glob(filepath.Join(tmpDir, "traces-*.jsonl"))
	if err != nil {
		t.Fatalf("Failed to glob files: %v", err)
	}

	if len(matches) == 0 {
		t.Error("Expected rotated files, but none found")
	}
}

