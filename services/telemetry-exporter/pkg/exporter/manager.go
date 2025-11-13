package exporter

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"go.opentelemetry.io/otel/sdk/trace"

	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/file"
	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/signavio"
)

// ExportMode defines how traces are exported.
type ExportMode string

const (
	ExportModeContinuous ExportMode = "continuous"
	ExportModeOnDemand   ExportMode = "on-demand"
	ExportModeBoth       ExportMode = "both"
)

// ExportManager manages multiple trace exporters with support for continuous and on-demand export.
type ExportManager struct {
	exporters      []trace.SpanExporter
	fileExporter   *file.FileExporter
	signavioExporter *signavio.SignavioExporter
	mode           ExportMode
	flushInterval  time.Duration
	stopChan       chan struct{}
	wg             sync.WaitGroup
	mu             sync.Mutex
	logger         *log.Logger
}

// ExportManagerConfig configures the export manager.
type ExportManagerConfig struct {
	Mode           ExportMode
	FlushInterval  time.Duration
	FileEnabled    bool
	FileConfig     file.FileExporterConfig
	SignavioEnabled bool
	SignavioConfig signavio.SignavioExporterConfig
	Logger         *log.Logger
}

// NewExportManager creates a new export manager.
func NewExportManager(cfg ExportManagerConfig) (*ExportManager, error) {
	if cfg.Logger == nil {
		cfg.Logger = log.New(os.Stdout, "[export-manager] ", log.LstdFlags)
	}
	if cfg.FlushInterval == 0 {
		cfg.FlushInterval = 30 * time.Second // Default flush interval
	}
	if cfg.Mode == "" {
		cfg.Mode = ExportModeBoth
	}

	em := &ExportManager{
		mode:          cfg.Mode,
		flushInterval: cfg.FlushInterval,
		stopChan:      make(chan struct{}),
		logger:        cfg.Logger,
		exporters:     make([]trace.SpanExporter, 0),
	}

	// Initialize file exporter if enabled
	if cfg.FileEnabled {
		fileExp, err := file.NewFileExporter(cfg.FileConfig)
		if err != nil {
			return nil, fmt.Errorf("create file exporter: %w", err)
		}
		em.fileExporter = fileExp
		em.exporters = append(em.exporters, fileExp.ToSpanExporter())
		cfg.Logger.Printf("File exporter enabled (path: %s)", cfg.FileConfig.BasePath)
	}

	// Initialize Signavio exporter if enabled
	if cfg.SignavioEnabled {
		signavioExp, err := signavio.NewSignavioExporter(cfg.SignavioConfig)
		if err != nil {
			return nil, fmt.Errorf("create Signavio exporter: %w", err)
		}
		em.signavioExporter = signavioExp
		em.exporters = append(em.exporters, signavioExp.ToSpanExporter())
		cfg.Logger.Printf("Signavio exporter enabled (dataset: %s)", cfg.SignavioConfig.Dataset)
	}

	if len(em.exporters) == 0 {
		return nil, fmt.Errorf("no exporters enabled")
	}

	// Start continuous export if enabled
	if cfg.Mode == ExportModeContinuous || cfg.Mode == ExportModeBoth {
		em.startContinuousExport()
	}

	return em, nil
}

// startContinuousExport starts the background worker for continuous export.
func (em *ExportManager) startContinuousExport() {
	em.wg.Add(1)
	go func() {
		defer em.wg.Done()
		ticker := time.NewTicker(em.flushInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				em.flushAll()
			case <-em.stopChan:
				return
			}
		}
	}()
	em.logger.Printf("Continuous export started (interval: %v)", em.flushInterval)
}

// flushAll flushes all exporters.
func (em *ExportManager) flushAll() {
	em.mu.Lock()
	defer em.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if em.signavioExporter != nil {
		if err := em.signavioExporter.Flush(ctx); err != nil {
			em.logger.Printf("Failed to flush Signavio exporter: %v", err)
		}
	}
}

// GetSpanExporter returns a multi-exporter that exports to all configured exporters.
func (em *ExportManager) GetSpanExporter() trace.SpanExporter {
	return &multiSpanExporter{exporters: em.exporters, logger: em.logger}
}

// ExportOnDemand exports traces on demand.
func (em *ExportManager) ExportOnDemand(ctx context.Context, spans []trace.ReadOnlySpan) error {
	if em.mode == ExportModeContinuous {
		return fmt.Errorf("on-demand export not enabled (mode: continuous)")
	}

	exporter := em.GetSpanExporter()
	return exporter.ExportSpans(ctx, spans)
}

// Shutdown shuts down all exporters.
func (em *ExportManager) Shutdown(ctx context.Context) error {
	close(em.stopChan)
	em.wg.Wait()

	em.mu.Lock()
	defer em.mu.Unlock()

	var errs []error
	for _, exporter := range em.exporters {
		if err := exporter.Shutdown(ctx); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}
	return nil
}

// multiSpanExporter exports to multiple exporters.
type multiSpanExporter struct {
	exporters []trace.SpanExporter
	logger    *log.Logger
}

// ExportSpans exports spans to all exporters.
func (mse *multiSpanExporter) ExportSpans(ctx context.Context, spans []trace.ReadOnlySpan) error {
	var errs []error
	for _, exporter := range mse.exporters {
		if err := exporter.ExportSpans(ctx, spans); err != nil {
			errs = append(errs, err)
			mse.logger.Printf("Export failed: %v", err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("export errors: %v", errs)
	}
	return nil
}

// Shutdown shuts down all exporters.
func (mse *multiSpanExporter) Shutdown(ctx context.Context) error {
	var errs []error
	for _, exporter := range mse.exporters {
		if err := exporter.Shutdown(ctx); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}
	return nil
}

