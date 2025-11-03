package gsmsymbolic

import "ai_benchmarks/internal/lnn"

// PhiMiniCalibrator learns optimal parameters for the Phi-Mini-3.5 model
type PhiMiniCalibrator struct {
	*lnn.DefaultCalibrator
	modelName    string
	learningRate float64
}

func NewPhiMiniCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// Use a smaller architecture suitable for Phi-Mini
	phiCfg := lnn.Config{
		InputSize:    64,    // Appropriate for smaller model
		HiddenSize:   96,    // Balanced capacity
		OutputSize:   10,    // Learn 10 parameters
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
	lnn.RegisterCalibrator("gsm-symbolic-phimini35", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("gsm-symbolic-phi", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("gsm-symbolic-phimini", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})
}
