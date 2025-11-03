package boolq

import "ai_benchmarks/internal/lnn"

// PhiMiniCalibrator learns optimal parameters for Phi-Mini-3.5 on BoolQ
type PhiMiniCalibrator struct {
	*lnn.DefaultCalibrator
	modelName string
}

func NewPhiMiniCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// Phi-Mini-3.5 configuration for BoolQ
	phiCfg := lnn.Config{
		InputSize:    64,
		HiddenSize:   96,
		OutputSize:   3, // Optimized for efficiency
		TimeSteps:    2,
		LearningRate: 0.002,
	}

	base, err := lnn.NewDefaultCalibrator(phiCfg)
	if err != nil {
		return nil, err
	}

	return &PhiMiniCalibrator{
		DefaultCalibrator: base.(*lnn.DefaultCalibrator),
		modelName:         "Phi-Mini-3.5",
	}, nil
}

func init() {
	lnn.RegisterCalibrator("boolq-phimini35", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("boolq-phimini", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("boolq-phi", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})
}
