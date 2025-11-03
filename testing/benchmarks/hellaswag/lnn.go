package hellaswag

import "ai_benchmarks/internal/lnn"

func init() {
	lnn.RegisterCalibrator("hellaswag", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return lnn.NewDefaultCalibrator(cfg)
	})
}
