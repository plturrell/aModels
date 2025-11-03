package triviaqa

import "ai_benchmarks/internal/lnn"

func init() {
	lnn.RegisterCalibrator("triviaqa", func(cfg lnn.Config) (lnn.Calibrator, error) {
		return lnn.NewDefaultCalibrator(cfg)
	})
}
