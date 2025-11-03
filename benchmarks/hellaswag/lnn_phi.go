package hellaswag

import "ai_benchmarks/internal/lnn"

// PhiMiniCalibrator learns optimal parameters for Phi-Mini-3.5 on HellaSwag
type PhiMiniCalibrator struct {
	*lnn.DefaultCalibrator
	modelName string
}

func NewPhiMiniCalibrator(cfg lnn.Config) (lnn.Calibrator, error) {
	// Phi-Mini has smaller capacity, optimized for efficiency
	phiCfg := lnn.Config{
		InputSize:    64,
		HiddenSize:   96,
		OutputSize:   6, // Learn 6 core parameters
		TimeSteps:    4,
		LearningRate: 0.001,
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
	lnn.RegisterCalibrator("hellaswag-phimini35", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("hellaswag-phimini", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})

	lnn.RegisterCalibrator("hellaswag-phi", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return NewPhiMiniCalibrator(cfg)
	})
}
