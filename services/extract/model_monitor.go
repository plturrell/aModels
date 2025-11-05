package main

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ModelPerformanceMetrics tracks model performance over time
type ModelPerformanceMetrics struct {
	TotalPredictions     int
	CorrectClassifications int
	ClassificationAccuracy float64
	QualityScoreMAE       float64 // Mean Absolute Error
	QualityScoreRMSE      float64 // Root Mean Squared Error
	UncertainPredictions  int
	LowConfidenceCount    int
	LastUpdated           time.Time
}

// ModelMonitor tracks model performance and enables active learning
type ModelMonitor struct {
	predictions     []MonitoredPrediction
	metrics         ModelPerformanceMetrics
	mu              sync.RWMutex
	logger          *log.Logger
	metricsFilePath string
	enabled         bool
}

// NewModelMonitor creates a new model performance monitor
func NewModelMonitor(metricsFilePath string, logger *log.Logger) *ModelMonitor {
	monitor := &ModelMonitor{
		predictions:     make([]MonitoredPrediction, 0),
		logger:          logger,
		metricsFilePath: metricsFilePath,
		enabled:        os.Getenv("MODEL_MONITORING_ENABLED") == "true",
	}

	if monitor.enabled && metricsFilePath != "" {
		if err := os.MkdirAll(filepath.Dir(metricsFilePath), 0755); err != nil {
			logger.Printf("failed to create metrics directory: %v", err)
			monitor.enabled = false
		} else {
			// Load existing metrics
			monitor.loadMetrics()
		}
	}

	return monitor
}

// MonitoredPrediction is the monitoring-focused view of a prediction
type MonitoredPrediction struct {
	PredictedClass      string
	ActualClass         string
	PredictedQuality    float64
	ActualQuality       float64
	PredictedConfidence float64
	Timestamp           time.Time
}

// RecordPrediction records a prediction for monitoring
func (mm *ModelMonitor) RecordPrediction(pred MonitoredPrediction) {
	if !mm.enabled {
		return
	}

	mm.mu.Lock()
	defer mm.mu.Unlock()

	pred.Timestamp = time.Now()
	mm.predictions = append(mm.predictions, pred)

	// Update metrics if we have ground truth
	if pred.ActualClass != "" || pred.ActualQuality > 0 {
		mm.updateMetrics(pred)
	}

	// Persist metrics periodically
	if len(mm.predictions)%10 == 0 {
		mm.persistMetrics()
	}
}

// updateMetrics updates performance metrics based on a prediction with ground truth
func (mm *ModelMonitor) updateMetrics(pred MonitoredPrediction) {
	mm.metrics.TotalPredictions++

	// Classification accuracy
	if pred.ActualClass != "" && pred.PredictedClass != "" {
		if pred.PredictedClass == pred.ActualClass {
			mm.metrics.CorrectClassifications++
		}
		mm.metrics.ClassificationAccuracy = float64(mm.metrics.CorrectClassifications) / float64(mm.metrics.TotalPredictions)
	}

	// Quality score errors
	if pred.ActualQuality > 0 && pred.PredictedQuality > 0 {
		diff := pred.PredictedQuality - pred.ActualQuality
		absDiff := diff
		if absDiff < 0 {
			absDiff = -absDiff
		}

		// Update MAE (simplified - in practice would use proper running average)
		mm.metrics.QualityScoreMAE = (mm.metrics.QualityScoreMAE*float64(mm.metrics.TotalPredictions-1) + absDiff) / float64(mm.metrics.TotalPredictions)
		
		// Update RMSE (simplified)
		sqDiff := diff * diff
		currentRMSE := mm.metrics.QualityScoreRMSE
		mm.metrics.QualityScoreRMSE = (currentRMSE*float64(mm.metrics.TotalPredictions-1) + sqDiff) / float64(mm.metrics.TotalPredictions)
	}

	// Uncertainty tracking
	if pred.PredictedConfidence < 0.7 {
		mm.metrics.LowConfidenceCount++
	}

	mm.metrics.LastUpdated = time.Now()
}

// ShouldReview determines if a prediction needs manual review (active learning)
func (mm *ModelMonitor) ShouldReview(predictedClass string, confidence float64) bool {
	if !mm.enabled {
		return false
	}

	// Low confidence predictions need review
	if confidence < 0.7 {
		return true
	}

	// Check if this class has low accuracy historically
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	// Count predictions for this class
	classCount := 0
	classCorrect := 0
	for _, p := range mm.predictions {
		if p.PredictedClass == predictedClass && p.ActualClass != "" {
			classCount++
			if p.PredictedClass == p.ActualClass {
				classCorrect++
			}
		}
	}

	// If we have enough data and accuracy is low, need review
	if classCount >= 5 {
		classAccuracy := float64(classCorrect) / float64(classCount)
		if classAccuracy < 0.6 {
			return true
		}
	}

	return false
}

// GetMetrics returns current performance metrics
func (mm *ModelMonitor) GetMetrics() ModelPerformanceMetrics {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.metrics
}

// GetUncertainPredictions returns predictions that need review
func (mm *ModelMonitor) GetUncertainPredictions(limit int) []MonitoredPrediction {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	uncertain := make([]MonitoredPrediction, 0)
	for _, pred := range mm.predictions {
		if pred.PredictedConfidence < 0.7 && pred.ActualClass == "" {
			uncertain = append(uncertain, pred)
			if len(uncertain) >= limit {
				break
			}
		}
	}

	return uncertain
}

// persistMetrics saves metrics to file
func (mm *ModelMonitor) persistMetrics() {
	if mm.metricsFilePath == "" {
		return
	}

	data, err := json.MarshalIndent(mm.metrics, "", "  ")
	if err != nil {
		mm.logger.Printf("failed to marshal metrics: %v", err)
		return
	}

	if err := os.WriteFile(mm.metricsFilePath, data, 0644); err != nil {
		mm.logger.Printf("failed to persist metrics: %v", err)
	}
}

// loadMetrics loads metrics from file
func (mm *ModelMonitor) loadMetrics() {
	if mm.metricsFilePath == "" {
		return
	}

	data, err := os.ReadFile(mm.metricsFilePath)
	if err != nil {
		if !os.IsNotExist(err) {
			mm.logger.Printf("failed to load metrics: %v", err)
		}
		return
	}

	if err := json.Unmarshal(data, &mm.metrics); err != nil {
		mm.logger.Printf("failed to unmarshal metrics: %v", err)
	}
}


