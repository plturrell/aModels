package storage

import (
	"math"
	"math/rand"
	"testing"
)

func TestDefaultPrivacyConfig(t *testing.T) {
	cfg := DefaultPrivacyConfig()
	if cfg.NoiseLevel != 0.1 {
		t.Fatalf("expected noise level 0.1, got %v", cfg.NoiseLevel)
	}
	if cfg.PrivacyBudget != 1.0 {
		t.Fatalf("expected privacy budget 1.0, got %v", cfg.PrivacyBudget)
	}
	if cfg.UsedBudget != 0.0 {
		t.Fatalf("expected used budget 0.0, got %v", cfg.UsedBudget)
	}
	if cfg.RetentionDays != 30 {
		t.Fatalf("expected retention days 30, got %d", cfg.RetentionDays)
	}
	if !cfg.EnableAnonymization {
		t.Fatal("expected anonymization to be enabled by default")
	}
	if cfg.PrivacyLevel != PrivacyLevelMedium {
		t.Fatalf("expected privacy level medium, got %s", cfg.PrivacyLevel)
	}
}

func TestNewPrivacyConfigLevels(t *testing.T) {
	tests := []struct {
		level             PrivacyLevel
		expectedBudget    float64
		expectedNoise     float64
		expectedRetention int
	}{
		{PrivacyLevelLow, 2.0, 0.05, 7},
		{PrivacyLevelMedium, 1.0, 0.1, 30},
		{PrivacyLevelHigh, 0.5, 0.2, 90},
	}

	for _, tt := range tests {
		cfg := NewPrivacyConfig(tt.level)
		if cfg.PrivacyBudget != tt.expectedBudget {
			t.Fatalf("level %s: expected budget %v, got %v", tt.level, tt.expectedBudget, cfg.PrivacyBudget)
		}
		if cfg.NoiseLevel != tt.expectedNoise {
			t.Fatalf("level %s: expected noise %v, got %v", tt.level, tt.expectedNoise, cfg.NoiseLevel)
		}
		if cfg.RetentionDays != tt.expectedRetention {
			t.Fatalf("level %s: expected retention %d, got %d", tt.level, tt.expectedRetention, cfg.RetentionDays)
		}
		if cfg.PrivacyLevel != tt.level {
			t.Fatalf("level %s: expected privacy level to match, got %s", tt.level, cfg.PrivacyLevel)
		}
	}
}

func TestPrivacyBudgetConsumption(t *testing.T) {
	cfg := DefaultPrivacyConfig()

	if !cfg.CanPerformOperation(0.5) {
		t.Fatal("expected to be able to perform operation costing 0.5")
	}

	if err := cfg.ConsumeBudget(0.5); err != nil {
		t.Fatalf("unexpected error consuming budget: %v", err)
	}

	if cfg.UsedBudget != 0.5 {
		t.Fatalf("expected used budget 0.5, got %v", cfg.UsedBudget)
	}

	if cfg.GetRemainingBudget() != 0.5 {
		t.Fatalf("expected remaining budget 0.5, got %v", cfg.GetRemainingBudget())
	}

	utilization := cfg.GetBudgetUtilization()
	if math.Abs(utilization-50.0) > 1e-6 {
		t.Fatalf("expected utilization 50%%, got %v", utilization)
	}

	if cfg.CanPerformOperation(0.6) {
		t.Fatal("expected operation exceeding budget to be disallowed")
	}

	if err := cfg.ConsumeBudget(0.6); err == nil {
		t.Fatal("expected error when consuming beyond budget")
	}

	cfg.ResetBudget()
	if cfg.UsedBudget != 0 {
		t.Fatalf("expected budget to reset to 0, got %v", cfg.UsedBudget)
	}
}

func TestValidatePrivacyConfig(t *testing.T) {
	cfg := DefaultPrivacyConfig()
	if err := ValidatePrivacyConfig(cfg); err != nil {
		t.Fatalf("expected valid config, got error: %v", err)
	}

	invalids := []PrivacyConfig{
		{NoiseLevel: -0.1, PrivacyBudget: 1, UsedBudget: 0},
		{NoiseLevel: 0.1, PrivacyBudget: 0, UsedBudget: 0},
		{NoiseLevel: 0.1, PrivacyBudget: 1, UsedBudget: -1},
		{NoiseLevel: 0.1, PrivacyBudget: 1, UsedBudget: 2},
	}

	for i, cfg := range invalids {
		if err := ValidatePrivacyConfig(&cfg); err == nil {
			t.Fatalf("expected validation error for invalid config %d", i)
		}
	}
}

func TestAnonymizeHelpers(t *testing.T) {
	hashed := AnonymizeString("user")
	if len(hashed) != 64 {
		t.Fatalf("expected SHA-256 hex string of length 64, got %d", len(hashed))
	}

	salted := AnonymizeWithSalt("user", "salt")
	if hashed == salted {
		t.Fatal("expected salted hash to differ from unsalted hash")
	}
}

func TestAddLaplacianNoiseDeterministic(t *testing.T) {
	rand.Seed(42)
	noisy1 := AddLaplacianNoise(10, 0.5)

	rand.Seed(42)
	noisy2 := AddLaplacianNoise(10, 0.5)

	if noisy1 != noisy2 {
		t.Fatalf("expected deterministic behaviour with fixed seed, got %v and %v", noisy1, noisy2)
	}

	if math.Abs(noisy1-10) < 1e-9 {
		t.Fatal("expected noise to change value when epsilon > 0")
	}

	// epsilon <= 0 should return original value
	if result := AddLaplacianNoise(5, 0); result != 5 {
		t.Fatalf("expected no change when epsilon <= 0, got %v", result)
	}
}

func TestAddGaussianNoiseDeterministic(t *testing.T) {
	rand.Seed(24)
	noisy1 := AddGaussianNoise(10, 0.5, 1e-5)

	rand.Seed(24)
	noisy2 := AddGaussianNoise(10, 0.5, 1e-5)

	if noisy1 != noisy2 {
		t.Fatalf("expected deterministic gaussian noise with fixed seed, got %v and %v", noisy1, noisy2)
	}

	if math.Abs(noisy1-10) < 1e-9 {
		t.Fatal("expected gaussian noise to adjust value when epsilon > 0")
	}

	if result := AddGaussianNoise(7, 0, 1e-5); result != 7 {
		t.Fatalf("expected no change when epsilon <= 0, got %v", result)
	}

	if result := AddGaussianNoise(7, 0.5, 0); result != 7 {
		t.Fatalf("expected no change when delta <= 0, got %v", result)
	}
}

func TestAddNoiseToVector(t *testing.T) {
	values := []float64{1, 2, 3}

	// Unknown noise type should act as no-op but still return new slice with same values
	noisy := AddNoiseToVector(values, 0.5, "unknown")
	if len(noisy) != len(values) {
		t.Fatalf("expected same length, got %d", len(noisy))
	}
	for i, v := range values {
		if noisy[i] != v {
			t.Fatalf("expected value %v at index %d, got %v", v, i, noisy[i])
		}
	}

	// Laplacian noise should be deterministic with seeded RNG
	rand.Seed(7)
	lap := AddNoiseToVector(values, 0.5, "laplacian")
	rand.Seed(7)
	expectedLap := []float64{
		AddLaplacianNoise(values[0], 0.5),
		AddLaplacianNoise(values[1], 0.5),
		AddLaplacianNoise(values[2], 0.5),
	}
	for i := range values {
		if lap[i] != expectedLap[i] {
			t.Fatalf("expected laplacian value %v at index %d, got %v", expectedLap[i], i, lap[i])
		}
	}

	// epsilon <= 0 should preserve values
	zeroNoise := AddNoiseToVector(values, 0, "gaussian")
	for i, v := range values {
		if zeroNoise[i] != v {
			t.Fatalf("expected zero epsilon to keep value %v at index %d, got %v", v, i, zeroNoise[i])
		}
	}
}
