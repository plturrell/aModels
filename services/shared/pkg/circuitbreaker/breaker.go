package circuitbreaker

import (
	"context"
	"time"

	"github.com/sony/gobreaker"
)

// CircuitBreaker wraps gobreaker.CircuitBreaker with a consistent interface.
type CircuitBreaker struct {
	cb     *gobreaker.CircuitBreaker
	name   string
	config Config
}

// Config configures the circuit breaker behavior.
type Config struct {
	Name              string        // Name of the circuit breaker (for metrics)
	MaxRequests       uint32        // Maximum requests in half-open state (default: 3)
	Interval          time.Duration // Time window for counting failures (default: 60s)
	Timeout           time.Duration // Time before attempting recovery (default: 30s)
	ReadyToTrip       func(counts gobreaker.Counts) bool // Function to determine when to open circuit
	OnStateChange     func(name string, from, to gobreaker.State) // Callback on state change
}

// DefaultConfig returns default circuit breaker configuration.
func DefaultConfig(name string) Config {
	return Config{
		Name:        name,
		MaxRequests: 3,
		Interval:    60 * time.Second,
		Timeout:     30 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > 5
		},
	}
}

// New creates a new circuit breaker with the given configuration.
func New(config Config) *CircuitBreaker {
	if config.Name == "" {
		config.Name = "circuit-breaker"
	}
	if config.MaxRequests == 0 {
		config.MaxRequests = 3
	}
	if config.Interval == 0 {
		config.Interval = 60 * time.Second
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.ReadyToTrip == nil {
		config.ReadyToTrip = func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > 5
		}
	}

	settings := gobreaker.Settings{
		Name:          config.Name,
		MaxRequests:   config.MaxRequests,
		Interval:      config.Interval,
		Timeout:       config.Timeout,
		ReadyToTrip:   config.ReadyToTrip,
		OnStateChange: config.OnStateChange,
	}

	return &CircuitBreaker{
		cb:     gobreaker.NewCircuitBreaker(settings),
		name:   config.Name,
		config: config,
	}
}

// Execute runs a function through the circuit breaker.
func (cb *CircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
	return cb.cb.Execute(fn)
}

// ExecuteWithContext runs a function through the circuit breaker with context support.
func (cb *CircuitBreaker) ExecuteWithContext(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
	// Check context before executing
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Execute through circuit breaker
	result, err := cb.cb.Execute(func() (interface{}, error) {
		// Check context again inside the function
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			return fn()
		}
	})

	return result, err
}

// Name returns the circuit breaker name.
func (cb *CircuitBreaker) Name() string {
	return cb.name
}

// State returns the current state of the circuit breaker.
func (cb *CircuitBreaker) State() gobreaker.State {
	return cb.cb.State()
}

// Counts returns the current counts for the circuit breaker.
func (cb *CircuitBreaker) Counts() gobreaker.Counts {
	return cb.cb.Counts()
}

// IsOpen returns true if the circuit breaker is in the open state.
func (cb *CircuitBreaker) IsOpen() bool {
	return cb.cb.State() == gobreaker.StateOpen
}

// IsHalfOpen returns true if the circuit breaker is in the half-open state.
func (cb *CircuitBreaker) IsHalfOpen() bool {
	return cb.cb.State() == gobreaker.StateHalfOpen
}

// IsClosed returns true if the circuit breaker is in the closed state.
func (cb *CircuitBreaker) IsClosed() bool {
	return cb.cb.State() == gobreaker.StateClosed
}

