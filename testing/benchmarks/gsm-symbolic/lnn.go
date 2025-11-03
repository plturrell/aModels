package gsmsymbolic

import (
	"ai_benchmarks/internal/lnn"
	"strings"
)

// ModelAwareCalibrator wraps model-specific calibrators with automatic detection
type ModelAwareCalibrator struct {
	modelType string
	inner     lnn.Calibrator
}

func NewModelAwareCalibrator(taskID string, cfg lnn.Config) (lnn.Calibrator, error) {
	// Extract model type from task ID if present (e.g., "gsm-symbolic-gemmavault")
	parts := strings.Split(taskID, "-")
	modelType := "default"
	if len(parts) > 2 {
		modelType = parts[2]
	}

	// Try to get model-specific calibrator
	var inner lnn.Calibrator
	var err error

	switch strings.ToLower(modelType) {
	case "gemmavault", "gemma":
		inner, err = NewGemmaVaultCalibrator(cfg)
	case "phimini35", "phimini", "phi":
		inner, err = NewPhiMiniCalibrator(cfg)
	default:
		inner, err = lnn.NewDefaultCalibrator(lnn.DefaultConfig())
	}

	if err != nil {
		return nil, err
	}

	return &ModelAwareCalibrator{
		modelType: modelType,
		inner:     inner,
	}, nil
}

func (m *ModelAwareCalibrator) Generate(taskID string, prevMetrics map[string]float64) (lnn.GeneratedOutput, error) {
	return m.inner.Generate(taskID, prevMetrics)
}

func (m *ModelAwareCalibrator) UpdateFromFeedback(generatedParams map[string]float64, actualPerformance float64) error {
	return m.inner.UpdateFromFeedback(generatedParams, actualPerformance)
}

func (m *ModelAwareCalibrator) Save(path string) error {
	return m.inner.Save(path)
}

func (m *ModelAwareCalibrator) Load(path string) error {
	return m.inner.Load(path)
}

func init() {
	// Register the default GSM-Symbolic calibrator with model awareness
	lnn.RegisterCalibrator("gsm-symbolic", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewModelAwareCalibrator("gsm-symbolic", cfg)
	})
}
