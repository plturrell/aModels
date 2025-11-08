package backend

import "os"

// AccelConfig controls coarse backend selection for benchmarks and tests.
// EnableFortran=false forces pure-Go via env override; true allows the registered
// provider (e.g., Fortran) if available.
type AccelConfig struct {
	EnableFortran bool
}

// SetAccelConfig applies the coarse selection by setting/unsetting
// INFRA_MATHS_BACKEND. Go is forced when set to "go"; otherwise the
// registered provider (if any) is used.
func SetAccelConfig(cfg AccelConfig) {
	if cfg.EnableFortran {
		_ = os.Unsetenv("INFRA_MATHS_BACKEND")
	} else {
		_ = os.Setenv("INFRA_MATHS_BACKEND", "go")
	}
}
