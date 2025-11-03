package gsmsymbolic

import "ai_benchmarks/internal/lnn"

// GemmaVaultCalibrator learns optimal parameters for the GemmaVault model
type GemmaVaultCalibrator struct {
	*lnn.DefaultCalibrator
	modelName    string
	learningRate float64
}

func NewGemmaVaultCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// Use a larger architecture for mathematical reasoning
	gemmaCfg := lnn.Config{
		InputSize:    128,    // Larger capacity for complex math
		HiddenSize:   256,    // More neurons for reasoning patterns
		OutputSize:   12,     // Learn 12 parameters (temperature, top_p, etc.)
		TimeSteps:    7,      // Deeper temporal dynamics
		LearningRate: 0.0005, // Lower LR for stability
	}

	base, err := lnn.NewDefaultCalibrator(gemmaCfg)
	if err != nil {
		return nil, err
	}

	return &GemmaVaultCalibrator{
		DefaultCalibrator: base.(*lnn.DefaultCalibrator),
		modelName:         "GemmaVault",
		learningRate:      gemmaCfg.LearningRate,
	}, nil
}

func init() {
	// Register GemmaVault-specific calibrator
	lnn.RegisterCalibrator("gsm-symbolic-gemmavault", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})

	lnn.RegisterCalibrator("gsm-symbolic-gemma", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})
}
