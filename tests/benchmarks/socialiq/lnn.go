package socialiq

import "ai_benchmarks/pkg/lnn"

func init() {
	lnn.RegisterCalibrator("socialiqa", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return lnn.NewDefaultCalibrator(cfg)
	})
}
