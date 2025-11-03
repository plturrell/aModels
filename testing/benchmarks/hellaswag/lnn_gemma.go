package hellaswag

import "ai_benchmarks/internal/lnn"

// GemmaVaultCalibrator learns optimal parameters for GemmaVault on HellaSwag
type GemmaVaultCalibrator struct {
	*lnn.DefaultCalibrator
	modelName string
}

func NewGemmaVaultCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// GemmaVault has high capacity for commonsense reasoning
	gemmaCfg := lnn.Config{
		InputSize:    128,
		HiddenSize:   256,
		OutputSize:   8, // Learn 8 parameters for HellaSwag
		TimeSteps:    6,
		LearningRate: 0.0005,
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
	lnn.RegisterCalibrator("hellaswag-gemmavault", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})

	lnn.RegisterCalibrator("hellaswag-gemma", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})
}
