package boolq

import "ai_benchmarks/internal/lnn"

// GemmaVaultCalibrator learns optimal parameters for GemmaVault on BoolQ
type GemmaVaultCalibrator struct {
	*lnn.DefaultCalibrator
	modelName string
}

func NewGemmaVaultCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// GemmaVault configuration for BoolQ
	gemmaCfg := lnn.Config{
		InputSize:    96,
		HiddenSize:   128,
		OutputSize:   4, // Parameters for yes/no confidence calibration
		TimeSteps:    3,
		LearningRate: 0.001,
	}

	base, err := lnn.NewDefaultCalibrator(gemmaCfg)
	if err != nil {
		return nil, err
	}

	return &GemmaVaultCalibrator{
		DefaultCalibrator: base.(*lnn.DefaultCalibrator),
		modelName:         "GemmaVault",
	}, nil
}

func init() {
	lnn.RegisterCalibrator("boolq-gemmavault", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})

	lnn.RegisterCalibrator("boolq-gemma", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})
}
