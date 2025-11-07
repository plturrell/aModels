package triviaqa

import "ai_benchmarks/pkg/lnn"

func init() {
	lnn.RegisterCalibrator("triviaqa", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return lnn.NewDefaultCalibrator(cfg)
	})
}
