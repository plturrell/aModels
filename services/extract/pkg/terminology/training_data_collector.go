package terminology

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// TrainingDataCollector collects training data for the sap-rpt-1-oss classifier
type TrainingDataCollector struct {
	outputPath string
	mu         sync.Mutex
	logger     *log.Logger
	enabled    bool
}

// NewTrainingDataCollector creates a new training data collector
func NewTrainingDataCollector(outputPath string, logger *log.Logger) *TrainingDataCollector {
	collector := &TrainingDataCollector{
		outputPath: outputPath,
		logger:     logger,
		enabled:    os.Getenv("COLLECT_TRAINING_DATA") == "true",
	}

	// Ensure output directory exists
	if collector.enabled && outputPath != "" {
		if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
			logger.Printf("failed to create training data directory: %v", err)
			collector.enabled = false
		}
	}

	return collector
}

// CollectTableClassification collects training data for a table classification
func (tdc *TrainingDataCollector) CollectTableClassification(
	tableName string,
	columns []map[string]any,
	classification string,
	confidence float64,
	context string,
) error {
	if !tdc.enabled || tdc.outputPath == "" {
		return nil // Silently skip if not enabled
	}

	tdc.mu.Lock()
	defer tdc.mu.Unlock()

	// Prepare columns JSON
	columnsJSON, err := json.Marshal(columns)
	if err != nil {
		return fmt.Errorf("marshal columns: %w", err)
	}

	// Call Python script to collect training data
	cmd := exec.Command("python3", "./scripts/classify_table_sap_rpt_full.py",
		"--table-name", tableName,
		"--columns", string(columnsJSON),
		"--context", context,
		"--collect-training",
		"--known-classification", classification,
		"--training-output", tdc.outputPath,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			tdc.logger.Printf("failed to collect training data: %v, stderr: %s", err, string(exitErr.Stderr))
			return fmt.Errorf("collect training data: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return fmt.Errorf("collect training data: %w", err)
	}

	var result map[string]any
	if err := json.Unmarshal(output, &result); err != nil {
		tdc.logger.Printf("failed to parse training data collection result: %v", err)
		return fmt.Errorf("parse result: %w", err)
	}

	if status, ok := result["status"].(string); ok && status == "collected" {
		tdc.logger.Printf("Collected training data for table %s (classification: %s)", tableName, classification)
		return nil
	}

	return fmt.Errorf("training data collection failed: %v", result)
}

// ExportTrainingData exports collected training data to a file
func (tdc *TrainingDataCollector) ExportTrainingData(destinationPath string) error {
	if tdc.outputPath == "" {
		return fmt.Errorf("no training data path configured")
	}

	// Copy training data to destination
	sourceData, err := os.ReadFile(tdc.outputPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no training data collected yet")
		}
		return fmt.Errorf("read training data: %w", err)
	}

	// Ensure destination directory exists
	if err := os.MkdirAll(filepath.Dir(destinationPath), 0755); err != nil {
		return fmt.Errorf("create destination directory: %w", err)
	}

	if err := os.WriteFile(destinationPath, sourceData, 0644); err != nil {
		return fmt.Errorf("write training data: %w", err)
	}

	tdc.logger.Printf("Exported training data to %s", destinationPath)
	return nil
}

// GetTrainingDataStats returns statistics about collected training data
func (tdc *TrainingDataCollector) GetTrainingDataStats() (map[string]any, error) {
	if tdc.outputPath == "" || !tdc.enabled {
		return map[string]any{
			"enabled": false,
			"count":   0,
		}, nil
	}

	data, err := os.ReadFile(tdc.outputPath)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]any{
				"enabled": true,
				"count":   0,
				"path":    tdc.outputPath,
			}, nil
		}
		return nil, fmt.Errorf("read training data: %w", err)
	}

	var trainingData []map[string]any
	if err := json.Unmarshal(data, &trainingData); err != nil {
		return nil, fmt.Errorf("unmarshal training data: %w", err)
	}

	// Count by classification
	classificationCounts := make(map[string]int)
	for _, record := range trainingData {
		if classification, ok := record["classification"].(string); ok {
			classificationCounts[classification]++
		}
	}

	return map[string]any{
		"enabled":              true,
		"count":                 len(trainingData),
		"path":                 tdc.outputPath,
		"classification_counts": classificationCounts,
		"last_updated":         time.Now().UTC().Format(time.RFC3339),
	}, nil
}

