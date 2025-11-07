package hellaswag

import "ai_benchmarks/pkg/lnn"

func init() {
	lnn.RegisterCalibrator("hellaswag", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return lnn.NewDefaultCalibrator(cfg)
	})
}
