package arc

import "ai_benchmarks/internal/lnn"

// GemmaVaultCalibrator learns optimal parameters for the GemmaVault model
// The LNN will discover the best configuration through training
type GemmaVaultCalibrator struct {
	*lnn.DefaultCalibrator
	modelName    string
	learningRate float64
}

func NewGemmaVaultCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// Use a larger architecture to learn complex parameter relationships
	// But don't hardcode the output parameters - let the LNN learn them
	gemmaCfg := lnn.Config{
		InputSize:    128,    // Larger capacity for learning
		HiddenSize:   256,    // More neurons to capture patterns
		OutputSize:   12,     // Learn 12 parameters (full ARC parameter space)
		TimeSteps:    7,      // Deeper temporal dynamics
		LearningRate: 0.0005, // Lower LR for stability with larger model
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
	lnn.RegisterCalibrator("arc-gemmavault", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})

	// Also register with common model name variants
	lnn.RegisterCalibrator("arc-gemma", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewGemmaVaultCalibrator(cfg)
	})
}
