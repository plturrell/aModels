package resilience

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// State represents the circuit breaker state.
type State int

const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

var (
	// ErrCircuitOpen is returned when the circuit breaker is open.
	ErrCircuitOpen = errors.New("circuit breaker is open")
	// ErrTooManyRequests is returned when too many requests are in flight in half-open state.
	ErrTooManyRequests = errors.New("too many requests in half-open state")
)

// CircuitBreaker implements the circuit breaker pattern.
type CircuitBreaker struct {
	mu              sync.RWMutex
	name            string
	maxRequests     uint32
	interval        time.Duration
	timeout         time.Duration
	failureThreshold uint32
	onStateChange   func(name string, from State, to State)

	state         State
	generation    uint64
	counts        *Counts
	expiry        time.Time
}

// Counts holds the statistics for the circuit breaker.
type Counts struct {
	Requests             uint32
	TotalSuccesses       uint32
	TotalFailures        uint32
	ConsecutiveSuccesses uint32
	ConsecutiveFailures  uint32
}

// Config holds the configuration for CircuitBreaker.
type Config struct {
	Name             string
	MaxRequests      uint32        // Max requests allowed in half-open state
	Interval         time.Duration // Rolling window for counts (closed state)
	Timeout          time.Duration // Time to wait before transitioning from open to half-open
	FailureThreshold uint32        // Number of consecutive failures to trigger open state
	OnStateChange    func(name string, from State, to State)
}

// DefaultConfig returns default configuration.
func DefaultConfig(name string) Config {
	return Config{
		Name:             name,
		MaxRequests:      1,
		Interval:         60 * time.Second,
		Timeout:          60 * time.Second,
		FailureThreshold: 5,
		OnStateChange:    nil,
	}
}

// NewCircuitBreaker creates a new CircuitBreaker.
func NewCircuitBreaker(config Config) *CircuitBreaker {
	cb := &CircuitBreaker{
		name:            config.Name,
		maxRequests:     config.MaxRequests,
		interval:        config.Interval,
		timeout:         config.Timeout,
		failureThreshold: config.FailureThreshold,
		onStateChange:   config.OnStateChange,
	}

	cb.toNewGeneration(time.Now())
	return cb
}

// Execute runs the given function if the circuit breaker is closed.
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func(context.Context) error) error {
	generation, err := cb.beforeRequest()
	if err != nil {
		return err
	}

	// Execute the function
	err = fn(ctx)

	// Record the result
	cb.afterRequest(generation, err == nil)

	return err
}

// beforeRequest checks if the request can proceed.
func (cb *CircuitBreaker) beforeRequest() (uint64, error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	state, generation := cb.currentState(now)

	if state == StateOpen {
		return generation, ErrCircuitOpen
	}

	if state == StateHalfOpen && cb.counts.Requests >= cb.maxRequests {
		return generation, ErrTooManyRequests
	}

	cb.counts.Requests++
	return generation, nil
}

// afterRequest records the result of the request.
func (cb *CircuitBreaker) afterRequest(generation uint64, success bool) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	state, currentGen := cb.currentState(now)

	// Ignore results from old generations
	if generation != currentGen {
		return
	}

	if success {
		cb.onSuccess(state, now)
	} else {
		cb.onFailure(state, now)
	}
}

// onSuccess handles successful request.
func (cb *CircuitBreaker) onSuccess(state State, now time.Time) {
	switch state {
	case StateClosed:
		cb.counts.TotalSuccesses++
		cb.counts.ConsecutiveSuccesses++
		cb.counts.ConsecutiveFailures = 0
	case StateHalfOpen:
		cb.counts.TotalSuccesses++
		cb.counts.ConsecutiveSuccesses++
		cb.counts.ConsecutiveFailures = 0
		
		// Transition to closed if we have enough successful requests
		if cb.counts.ConsecutiveSuccesses >= cb.maxRequests {
			cb.setState(StateClosed, now)
		}
	}
}

// onFailure handles failed request.
func (cb *CircuitBreaker) onFailure(state State, now time.Time) {
	switch state {
	case StateClosed:
		cb.counts.TotalFailures++
		cb.counts.ConsecutiveFailures++
		cb.counts.ConsecutiveSuccesses = 0
		
		// Transition to open if threshold exceeded
		if cb.counts.ConsecutiveFailures >= cb.failureThreshold {
			cb.setState(StateOpen, now)
		}
	case StateHalfOpen:
		// Any failure in half-open immediately opens the circuit
		cb.setState(StateOpen, now)
	}
}

// currentState returns the current state and generation.
func (cb *CircuitBreaker) currentState(now time.Time) (State, uint64) {
	switch cb.state {
	case StateClosed:
		if !cb.expiry.IsZero() && cb.expiry.Before(now) {
			cb.toNewGeneration(now)
		}
	case StateOpen:
		if cb.expiry.Before(now) {
			cb.setState(StateHalfOpen, now)
		}
	}

	return cb.state, cb.generation
}

// setState changes the circuit breaker state.
func (cb *CircuitBreaker) setState(state State, now time.Time) {
	if cb.state == state {
		return
	}

	prev := cb.state
	cb.state = state

	cb.toNewGeneration(now)

	if cb.onStateChange != nil {
		cb.onStateChange(cb.name, prev, state)
	}
}

// toNewGeneration starts a new generation.
func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
	cb.generation++
	cb.counts = &Counts{}

	var zero time.Time
	switch cb.state {
	case StateClosed:
		if cb.interval == 0 {
			cb.expiry = zero
		} else {
			cb.expiry = now.Add(cb.interval)
		}
	case StateOpen:
		cb.expiry = now.Add(cb.timeout)
	default:
		cb.expiry = zero
	}
}

// State returns the current state.
func (cb *CircuitBreaker) State() State {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	now := time.Now()
	state, _ := cb.currentState(now)
	return state
}

// Counts returns the current counts.
func (cb *CircuitBreaker) Counts() Counts {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	return *cb.counts
}

// Reset resets the circuit breaker to closed state.
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.toNewGeneration(time.Now())
	cb.state = StateClosed
}

// String returns a string representation of the circuit breaker.
func (cb *CircuitBreaker) String() string {
	return fmt.Sprintf("CircuitBreaker[%s](state=%s, requests=%d, successes=%d, failures=%d)",
		cb.name, cb.State(), cb.counts.Requests, cb.counts.TotalSuccesses, cb.counts.TotalFailures)
}
