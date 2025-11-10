package exporter

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/testing"
)

// SignavioExporter handles exporting telemetry to Signavio.
type SignavioExporter struct {
	client   *testing.SignavioClient
	dataset  string
	logger   *log.Logger
	exported map[string]bool // Track exported sessions for idempotency
}

// NewSignavioExporter creates a new Signavio exporter.
func NewSignavioExporter(
	baseURL, apiKey, tenantID, dataset string,
	enabled bool,
	timeout time.Duration,
	maxRetries int,
	logger *log.Logger,
) *SignavioExporter {
	client := testing.NewSignavioClient(
		baseURL,
		apiKey,
		tenantID,
		enabled,
		timeout,
		maxRetries,
		logger,
	)

	return &SignavioExporter{
		client:   client,
		dataset:  dataset,
		logger:   logger,
		exported: make(map[string]bool),
	}
}

// ExportSession exports a single session's telemetry to Signavio.
func (e *SignavioExporter) ExportSession(ctx context.Context, record *testing.SignavioTelemetryRecord) error {
	return e.ExportSessionToDataset(ctx, record, e.dataset)
}

// ExportSessionToDataset exports a single session's telemetry to Signavio with custom dataset.
func (e *SignavioExporter) ExportSessionToDataset(ctx context.Context, record *testing.SignavioTelemetryRecord, dataset string) error {
	if !e.client.IsEnabled() {
		return fmt.Errorf("Signavio client is not enabled")
	}

	if dataset == "" {
		dataset = e.dataset
	}

	// Check if already exported (idempotency)
	exportKey := fmt.Sprintf("%s:%s", record.AgentRunID, dataset)
	if e.exported[exportKey] {
		e.logger.Printf("Session %s already exported to dataset %s, skipping", record.AgentRunID, dataset)
		return nil
	}

	// Upload to Signavio
	if err := e.client.UploadTelemetry(ctx, dataset, []testing.SignavioTelemetryRecord{*record}); err != nil {
		return fmt.Errorf("upload telemetry: %w", err)
	}

	// Mark as exported
	e.exported[exportKey] = true
	e.logger.Printf("Successfully exported session %s to Signavio dataset %s", record.AgentRunID, dataset)

	return nil
}

// ExportBatch exports multiple sessions' telemetry to Signavio.
func (e *SignavioExporter) ExportBatch(ctx context.Context, records []*testing.SignavioTelemetryRecord) error {
	return e.ExportBatchToDataset(ctx, records, e.dataset)
}

// ExportBatchToDataset exports multiple sessions' telemetry to Signavio with custom dataset.
func (e *SignavioExporter) ExportBatchToDataset(ctx context.Context, records []*testing.SignavioTelemetryRecord, dataset string) error {
	if !e.client.IsEnabled() {
		return fmt.Errorf("Signavio client is not enabled")
	}

	if len(records) == 0 {
		return fmt.Errorf("no records to export")
	}

	if dataset == "" {
		dataset = e.dataset
	}

	// Filter out already exported sessions
	toExport := make([]testing.SignavioTelemetryRecord, 0, len(records))
	for _, record := range records {
		exportKey := fmt.Sprintf("%s:%s", record.AgentRunID, dataset)
		if !e.exported[exportKey] {
			toExport = append(toExport, *record)
		}
	}

	if len(toExport) == 0 {
		e.logger.Printf("All sessions already exported to dataset %s, skipping batch", dataset)
		return nil
	}

	// Upload batch to Signavio
	if err := e.client.UploadTelemetry(ctx, dataset, toExport); err != nil {
		return fmt.Errorf("upload batch telemetry: %w", err)
	}

	// Mark all as exported
	for _, record := range toExport {
		exportKey := fmt.Sprintf("%s:%s", record.AgentRunID, dataset)
		e.exported[exportKey] = true
	}

	e.logger.Printf("Successfully exported %d sessions to Signavio dataset %s", len(toExport), dataset)

	return nil
}

// ValidateConnection checks if Signavio connection is valid.
func (e *SignavioExporter) ValidateConnection(ctx context.Context) error {
	return e.client.HealthCheck(ctx)
}

// IsEnabled checks if Signavio exporter is enabled.
func (e *SignavioExporter) IsEnabled() bool {
	return e.client.IsEnabled()
}

// GetExportStatus checks if a session has been exported.
func (e *SignavioExporter) GetExportStatus(sessionID string) bool {
	return e.exported[sessionID]
}

