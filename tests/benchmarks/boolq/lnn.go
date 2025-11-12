package boolq

import "ai_benchmarks/pkg/lnn"

func init() {
	lnn.RegisterCalibrator("boolq", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return lnn.NewDefaultCalibrator(cfg)
	})
}
