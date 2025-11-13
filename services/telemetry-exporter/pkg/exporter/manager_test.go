package exporter

import (
	"context"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/file"
	"go.opentelemetry.io/otel/sdk/trace"
)

func TestExportManager(t *testing.T) {
	cfg := ExportManagerConfig{
		Mode:          ExportModeBoth,
		FlushInterval: 1 * time.Second,
		FileEnabled:   true,
		FileConfig: file.FileExporterConfig{
			BasePath:    t.TempDir(),
			MaxFileSize: 1024 * 1024,
			MaxFiles:    10,
		},
		SignavioEnabled: false,
		Logger:         nil,
	}

	manager, err := NewExportManager(cfg)
	if err != nil {
		t.Fatalf("Failed to create export manager: %v", err)
	}
	defer manager.Shutdown(context.Background())

	// Get span exporter
	exporter := manager.GetSpanExporter()
	if exporter == nil {
		t.Fatal("Expected span exporter, got nil")
	}

	// Test export
	spans := []trace.ReadOnlySpan{}
	err = exporter.ExportSpans(context.Background(), spans)
	if err != nil {
		t.Errorf("Failed to export spans: %v", err)
	}
}

func TestExportManagerShutdown(t *testing.T) {
	cfg := ExportManagerConfig{
		Mode:          ExportModeContinuous,
		FlushInterval: 1 * time.Second,
		FileEnabled:   true,
		FileConfig: file.FileExporterConfig{
			BasePath:    t.TempDir(),
			MaxFileSize: 1024 * 1024,
			MaxFiles:    10,
		},
		SignavioEnabled: false,
		Logger:         nil,
	}

	manager, err := NewExportManager(cfg)
	if err != nil {
		t.Fatalf("Failed to create export manager: %v", err)
	}

	// Shutdown should not error
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = manager.Shutdown(ctx)
	if err != nil {
		t.Errorf("Shutdown failed: %v", err)
	}
}

