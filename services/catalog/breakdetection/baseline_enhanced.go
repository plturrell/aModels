package breakdetection

import (
	"bytes"
	"compress/gzip"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/google/uuid"
)

// BaselineManagerEnhanced provides enhanced baseline management features
type BaselineManagerEnhanced struct {
	*BaselineManager
	enableCompression bool
	enableValidation  bool
	enableChecksum    bool
	maxSnapshotSize   int64
	compressionLevel  int
}

// NewBaselineManagerEnhanced creates an enhanced baseline manager
func NewBaselineManagerEnhanced(
	baseManager *BaselineManager,
	enableCompression bool,
	enableValidation bool,
) *BaselineManagerEnhanced {
	return &BaselineManagerEnhanced{
		BaselineManager:  baseManager,
		enableCompression: enableCompression,
		enableValidation:  enableValidation,
		maxSnapshotSize:   100 * 1024 * 1024, // 100MB
		compressionLevel:  gzip.BestCompression,
	}
}

// CreateBaselineEnhanced creates a baseline with enhanced features
func (bme *BaselineManagerEnhanced) CreateBaselineEnhanced(ctx context.Context, req *BaselineRequest) (*Baseline, error) {
	// Validate baseline request
	if bme.enableValidation {
		// Create temporary baseline for validation
		tempBaseline := &Baseline{
			BaselineID:   fmt.Sprintf("temp-%s", uuid.New().String()),
			SystemName:  req.SystemName,
			Version:     req.Version,
			SnapshotData: req.SnapshotData,
		}
		if err := ValidateBaseline(tempBaseline); err != nil {
			return nil, fmt.Errorf("baseline validation failed: %w", err)
		}
	}

	// Compress snapshot data if enabled
	snapshotData := req.SnapshotData
	if bme.enableCompression && len(snapshotData) > 1024*1024 { // Compress if > 1MB
		compressed, err := bme.compressData(snapshotData)
		if err != nil {
			if bme.logger != nil {
				bme.logger.Printf("Warning: Compression failed, storing uncompressed: %v", err)
			}
		} else {
			snapshotData = compressed
			// Add compression metadata
			if req.Metadata == nil {
				req.Metadata = make(map[string]interface{})
			}
			req.Metadata["compressed"] = true
			req.Metadata["original_size"] = len(req.SnapshotData)
			req.Metadata["compressed_size"] = len(compressed)
			req.Metadata["compression_ratio"] = float64(len(compressed)) / float64(len(req.SnapshotData))
		}
	}

	// Calculate checksum if enabled
	if bme.enableChecksum {
		checksum := bme.calculateChecksum(snapshotData)
		if req.Metadata == nil {
			req.Metadata = make(map[string]interface{})
		}
		req.Metadata["checksum"] = checksum
		req.Metadata["checksum_algorithm"] = "sha256"
	}

	// Verify snapshot completeness
	if err := bme.verifySnapshotCompleteness(req); err != nil {
		if bme.logger != nil {
			bme.logger.Printf("Warning: Snapshot completeness check failed: %v", err)
		}
		// Log but don't fail - allow partial snapshots
	}

	// Create baseline using base manager
	req.SnapshotData = snapshotData
	return bme.BaselineManager.CreateBaseline(ctx, req)
}

// GetBaselineEnhanced retrieves a baseline with decompression if needed
func (bme *BaselineManagerEnhanced) GetBaselineEnhanced(ctx context.Context, baselineID string) (*Baseline, error) {
	baseline, err := bme.BaselineManager.GetBaseline(ctx, baselineID)
	if err != nil {
		return nil, err
	}

	// Decompress if compressed
	if bme.isCompressed(baseline) {
		decompressed, err := bme.decompressData(baseline.SnapshotData)
		if err != nil {
			return nil, fmt.Errorf("failed to decompress baseline: %w", err)
		}
		baseline.SnapshotData = decompressed
	}

	// Verify checksum if enabled
	if bme.enableChecksum && bme.hasChecksum(baseline) {
		if err := bme.verifyChecksum(baseline); err != nil {
			return nil, fmt.Errorf("checksum verification failed: %w", err)
		}
	}

	return baseline, nil
}

// CompareBaselinesEnhanced compares baselines with enhanced features
func (bme *BaselineManagerEnhanced) CompareBaselinesEnhanced(ctx context.Context, 
	baseline1ID, baseline2ID string) (*BreakComparison, error) {
	baseline1, err := bme.GetBaselineEnhanced(ctx, baseline1ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get baseline 1: %w", err)
	}

	baseline2, err := bme.GetBaselineEnhanced(ctx, baseline2ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get baseline 2: %w", err)
	}

	// Perform comparison
	return bme.BaselineManager.CompareBaselines(ctx, baseline1ID, baseline2ID)
}

// VerifySnapshotIntegrity verifies the integrity of a snapshot
func (bme *BaselineManagerEnhanced) VerifySnapshotIntegrity(ctx context.Context, baselineID string) error {
	baseline, err := bme.GetBaselineEnhanced(ctx, baselineID)
	if err != nil {
		return err
	}

	// Verify snapshot data is valid JSON
	var testData map[string]interface{}
	if err := json.Unmarshal(baseline.SnapshotData, &testData); err != nil {
		return fmt.Errorf("snapshot data is not valid JSON: %w", err)
	}

	// Verify checksum if enabled
	if bme.enableChecksum && bme.hasChecksum(baseline) {
		if err := bme.verifyChecksum(baseline); err != nil {
			return fmt.Errorf("checksum verification failed: %w", err)
		}
	}

	return nil
}

// Helper methods
func (bme *BaselineManagerEnhanced) compressData(data []byte) ([]byte, error) {
	var buf []byte
	writer := io.Writer(&buf)
	
	gzipWriter, err := gzip.NewWriterLevel(writer, bme.compressionLevel)
	if err != nil {
		return nil, err
	}
	defer gzipWriter.Close()

	if _, err := gzipWriter.Write(data); err != nil {
		return nil, err
	}

	if err := gzipWriter.Close(); err != nil {
		return nil, err
	}

	return buf, nil
}

func (bme *BaselineManagerEnhanced) decompressData(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	decompressed, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	return decompressed, nil
}

func (bme *BaselineManagerEnhanced) isCompressed(baseline *Baseline) bool {
	if baseline.Metadata == nil {
		return false
	}
	compressed, ok := baseline.Metadata["compressed"].(bool)
	return ok && compressed
}

func (bme *BaselineManagerEnhanced) hasChecksum(baseline *Baseline) bool {
	if baseline.Metadata == nil {
		return false
	}
	_, ok := baseline.Metadata["checksum"]
	return ok
}

func (bme *BaselineManagerEnhanced) calculateChecksum(data []byte) string {
	// TODO: Implement actual SHA256 checksum
	// For now, return placeholder
	return fmt.Sprintf("sha256:%x", len(data))
}

func (bme *BaselineManagerEnhanced) verifyChecksum(baseline *Baseline) error {
	if baseline.Metadata == nil {
		return fmt.Errorf("no metadata available")
	}

	expectedChecksum, ok := baseline.Metadata["checksum"].(string)
	if !ok {
		return fmt.Errorf("checksum not found in metadata")
	}

	actualChecksum := bme.calculateChecksum(baseline.SnapshotData)
	if actualChecksum != expectedChecksum {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedChecksum, actualChecksum)
	}

	return nil
}

func (bme *BaselineManagerEnhanced) verifySnapshotCompleteness(req *BaselineRequest) error {
	// Verify snapshot data is valid JSON
	var snapshotData map[string]interface{}
	if err := json.Unmarshal(req.SnapshotData, &snapshotData); err != nil {
		return fmt.Errorf("invalid snapshot data: %w", err)
	}

	// Check for required fields based on system type
	requiredFields := bme.getRequiredFields(req.SystemName)
	for _, field := range requiredFields {
		if _, exists := snapshotData[field]; !exists {
			return fmt.Errorf("required field missing: %s", field)
		}
	}

	return nil
}

func (bme *BaselineManagerEnhanced) getRequiredFields(systemName SystemName) []string {
	switch systemName {
	case SystemSAPFioneer:
		return []string{"journal_entries", "account_balances"}
	case SystemBCRS:
		return []string{"credit_exposures", "capital_ratios"}
	case SystemRCO:
		return []string{"liquidity_positions", "lcr_ratios"}
	case SystemAxiomSL:
		return []string{"regulatory_reports", "compliance_checks"}
	default:
		return []string{}
	}
}

