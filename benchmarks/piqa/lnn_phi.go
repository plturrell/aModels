package piqa

import "ai_benchmarks/internal/lnn"

// PhiMiniCalibrator learns optimal parameters for the Phi-Mini-3.5 model
// The LNN will discover the best configuration through training
type PhiMiniCalibrator struct {
	*lnn.DefaultCalibrator
	modelName    string
	learningRate float64
}

func NewPhiMiniCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// Use a smaller architecture suitable for the model's capacity
	// But still let the LNN learn the optimal parameters
	phiCfg := lnn.Config{
		InputSize:    64,    // Appropriate for smaller model
		HiddenSize:   96,    // Balanced capacity
		OutputSize:   10,    // Learn 10 parameters (good coverage)
		TimeSteps:    4,     // Moderate temporal depth
		LearningRate: 0.001, // Standard learning rate
	}

	base, err := lnn.NewDefaultCalibrator(phiCfg)
	if err != nil {
		return nil, err
	}

	return &PhiMiniCalibrator{
		DefaultCalibrator: base.(*lnn.DefaultCalibrator),
		modelName:         "Phi-Mini-3.5",
		learningRate:      phiCfg.LearningRate,
	}, nil
}

func init() {
	// Register Phi-Mini-specific calibrator
	lnn.RegisterCalibrator("piqa-phimini35", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	// Also register with common model name variants
	lnn.RegisterCalibrator("piqa-phi", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("piqa-phimini", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})
}
