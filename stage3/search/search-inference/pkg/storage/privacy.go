package storage

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// PrivacyLevel defines the level of privacy protection
type PrivacyLevel string

const (
	PrivacyLevelLow    PrivacyLevel = "low"
	PrivacyLevelMedium PrivacyLevel = "medium"
	PrivacyLevelHigh   PrivacyLevel = "high"
)

// PrivacyConfig defines differential privacy settings for search operations.
type PrivacyConfig struct {
	NoiseLevel          float64      `json:"noise_level"`
	PrivacyBudget       float64      `json:"privacy_budget"`
	UsedBudget          float64      `json:"used_budget"`
	RetentionDays       int          `json:"retention_days"`
	EnableAnonymization bool         `json:"enable_anonymization"`
	PrivacyLevel        PrivacyLevel `json:"privacy_level"`
}

// DefaultPrivacyConfig returns a default privacy configuration
func DefaultPrivacyConfig() *PrivacyConfig {
	return &PrivacyConfig{
		NoiseLevel:          0.1,
		PrivacyBudget:       1.0,
		UsedBudget:          0.0,
		RetentionDays:       30,
		EnableAnonymization: true,
		PrivacyLevel:        PrivacyLevelMedium,
	}
}

// NewPrivacyConfig creates a new privacy configuration
func NewPrivacyConfig(level PrivacyLevel) *PrivacyConfig {
	config := DefaultPrivacyConfig()

	switch level {
	case PrivacyLevelLow:
		config.NoiseLevel = 0.05
		config.PrivacyBudget = 2.0
		config.RetentionDays = 7
		config.PrivacyLevel = PrivacyLevelLow
	case PrivacyLevelMedium:
		config.NoiseLevel = 0.1
		config.PrivacyBudget = 1.0
		config.RetentionDays = 30
		config.PrivacyLevel = PrivacyLevelMedium
	case PrivacyLevelHigh:
		config.NoiseLevel = 0.2
		config.PrivacyBudget = 0.5
		config.RetentionDays = 90
		config.PrivacyLevel = PrivacyLevelHigh
	}

	return config
}

// CanPerformOperation checks if an operation can be performed within privacy budget
func (pc *PrivacyConfig) CanPerformOperation(cost float64) bool {
	return pc.UsedBudget+cost <= pc.PrivacyBudget
}

// ConsumeBudget consumes privacy budget for an operation
func (pc *PrivacyConfig) ConsumeBudget(cost float64) error {
	if !pc.CanPerformOperation(cost) {
		return fmt.Errorf("privacy budget would be exceeded: current=%.6f, cost=%.6f, budget=%.6f",
			pc.UsedBudget, cost, pc.PrivacyBudget)
	}

	pc.UsedBudget += cost
	return nil
}

// ResetBudget resets the privacy budget (typically called daily)
func (pc *PrivacyConfig) ResetBudget() {
	pc.UsedBudget = 0.0
}

// GetRemainingBudget returns the remaining privacy budget
func (pc *PrivacyConfig) GetRemainingBudget() float64 {
	return pc.PrivacyBudget - pc.UsedBudget
}

// GetBudgetUtilization returns the budget utilization percentage
func (pc *PrivacyConfig) GetBudgetUtilization() float64 {
	if pc.PrivacyBudget == 0 {
		return 0.0
	}
	return (pc.UsedBudget / pc.PrivacyBudget) * 100.0
}

// AnonymizeString anonymizes a string using SHA-256 hashing
func AnonymizeString(s string) string {
	hash := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", hash)
}

// AnonymizeWithSalt anonymizes a string with a salt for additional security
func AnonymizeWithSalt(s, salt string) string {
	hash := sha256.Sum256([]byte(s + salt))
	return fmt.Sprintf("%x", hash)
}

// AddLaplacianNoise adds Laplacian noise to a value for differential privacy
func AddLaplacianNoise(value, epsilon float64) float64 {
	if epsilon <= 0 {
		return value
	}

	// Generate Laplacian noise
	u := rand.Float64() - 0.5
	if u == 0 {
		u = 0.0001 // Avoid log(0)
	}

	sign := 1.0
	if u < 0 {
		sign = -1.0
	}
	noise := -sign * math.Log(1-2*math.Abs(u)) / epsilon
	return value + noise
}

// AddGaussianNoise adds Gaussian noise to a value for (ε,δ)-differential privacy
func AddGaussianNoise(value, epsilon, delta float64) float64 {
	if epsilon <= 0 || delta <= 0 {
		return value
	}

	// Calculate noise scale for Gaussian mechanism
	sigma := math.Sqrt(2*math.Log(1.25/delta)) / epsilon

	// Generate Gaussian noise using Box-Muller transform
	u1 := rand.Float64()
	u2 := rand.Float64()

	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	noise := sigma * z0

	return value + noise
}

// AddNoiseToVector adds noise to a vector of values
func AddNoiseToVector(values []float64, epsilon float64, noiseType string) []float64 {
	noisy := make([]float64, len(values))

	for i, val := range values {
		switch noiseType {
		case "laplacian":
			noisy[i] = AddLaplacianNoise(val, epsilon)
		case "gaussian":
			noisy[i] = AddGaussianNoise(val, epsilon, 1e-5) // Default delta
		default:
			noisy[i] = val // No noise
		}
	}

	return noisy
}

// PrivacyBudgetCosts defines the cost of various operations
var PrivacyBudgetCosts = struct {
	DocumentAdd    float64
	SearchQuery    float64
	SearchLog      float64
	ClickLog       float64
	AnalyticsQuery float64
}{
	DocumentAdd:    0.02,
	SearchQuery:    0.03,
	SearchLog:      0.01,
	ClickLog:       0.001,
	AnalyticsQuery: 0.005,
}

// ValidatePrivacyConfig validates a privacy configuration
func ValidatePrivacyConfig(config *PrivacyConfig) error {
	if config.NoiseLevel < 0 {
		return fmt.Errorf("noise level must be non-negative")
	}

	if config.PrivacyBudget <= 0 {
		return fmt.Errorf("privacy budget must be positive")
	}

	if config.UsedBudget < 0 {
		return fmt.Errorf("used budget must be non-negative")
	}

	if config.UsedBudget > config.PrivacyBudget {
		return fmt.Errorf("used budget cannot exceed total budget")
	}

	if config.RetentionDays <= 0 {
		return fmt.Errorf("retention days must be positive")
	}

	return nil
}

// GetPrivacyLevelFromString converts a string to PrivacyLevel
func GetPrivacyLevelFromString(s string) PrivacyLevel {
	switch s {
	case "low":
		return PrivacyLevelLow
	case "medium":
		return PrivacyLevelMedium
	case "high":
		return PrivacyLevelHigh
	default:
		return PrivacyLevelMedium
	}
}

// String returns the string representation of PrivacyLevel
func (pl PrivacyLevel) String() string {
	return string(pl)
}

// IsValid checks if the privacy level is valid
func (pl PrivacyLevel) IsValid() bool {
	return pl == PrivacyLevelLow || pl == PrivacyLevelMedium || pl == PrivacyLevelHigh
}

// GetNoiseLevel returns the noise level for a privacy level
func (pl PrivacyLevel) GetNoiseLevel() float64 {
	switch pl {
	case PrivacyLevelLow:
		return 0.05
	case PrivacyLevelMedium:
		return 0.1
	case PrivacyLevelHigh:
		return 0.2
	default:
		return 0.1
	}
}

// GetBudgetForLevel returns the privacy budget for a privacy level
func (pl PrivacyLevel) GetBudgetForLevel() float64 {
	switch pl {
	case PrivacyLevelLow:
		return 2.0
	case PrivacyLevelMedium:
		return 1.0
	case PrivacyLevelHigh:
		return 0.5
	default:
		return 1.0
	}
}

// GetRetentionDays returns the retention days for a privacy level
func (pl PrivacyLevel) GetRetentionDays() int {
	switch pl {
	case PrivacyLevelLow:
		return 7
	case PrivacyLevelMedium:
		return 30
	case PrivacyLevelHigh:
		return 90
	default:
		return 30
	}
}

// PrivacyAuditLog represents a privacy audit log entry
type PrivacyAuditLog struct {
	Timestamp    time.Time `json:"timestamp"`
	Operation    string    `json:"operation"`
	PrivacyCost  float64   `json:"privacy_cost"`
	BudgetBefore float64   `json:"budget_before"`
	BudgetAfter  float64   `json:"budget_after"`
	UserIDHash   string    `json:"user_id_hash"`
	SessionID    string    `json:"session_id"`
	Details      string    `json:"details"`
}

// LogPrivacyOperation logs a privacy operation for audit purposes
func LogPrivacyOperation(operation string, cost float64, budgetBefore, budgetAfter float64, userID, sessionID, details string) *PrivacyAuditLog {
	return &PrivacyAuditLog{
		Timestamp:    time.Now(),
		Operation:    operation,
		PrivacyCost:  cost,
		BudgetBefore: budgetBefore,
		BudgetAfter:  budgetAfter,
		UserIDHash:   AnonymizeString(userID),
		SessionID:    sessionID,
		Details:      details,
	}
}
