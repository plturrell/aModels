package server

import "time"

// RetryConfig defines retry behavior
type RetryConfig struct {
	MaxAttempts int           `json:"max_attempts"`
	BaseDelay   time.Duration `json:"base_delay"`
	MaxDelay    time.Duration `json:"max_delay"`
	Multiplier  float64       `json:"multiplier"`
	Jitter      bool          `json:"jitter"`
}
