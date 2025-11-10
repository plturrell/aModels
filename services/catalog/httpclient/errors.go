package httpclient

import (
	"fmt"
)

// IntegrationError is the base error type for integration errors.
type IntegrationError struct {
	Message       string
	Service       string
	CorrelationID string
	StatusCode    int
	Err           error
}

func (e *IntegrationError) Error() string {
	msg := fmt.Sprintf("integration error: %s", e.Message)
	if e.Service != "" {
		msg += fmt.Sprintf(" (service: %s)", e.Service)
	}
	if e.CorrelationID != "" {
		msg += fmt.Sprintf(" (correlation_id: %s)", e.CorrelationID)
	}
	if e.StatusCode > 0 {
		msg += fmt.Sprintf(" (status: %d)", e.StatusCode)
	}
	if e.Err != nil {
		msg += fmt.Sprintf(": %v", e.Err)
	}
	return msg
}

func (e *IntegrationError) Unwrap() error {
	return e.Err
}

// ServiceUnavailableError indicates the service is unavailable.
type ServiceUnavailableError struct {
	*IntegrationError
}

// NewServiceUnavailableError creates a new service unavailable error.
func NewServiceUnavailableError(service, correlationID string, err error) *ServiceUnavailableError {
	return &ServiceUnavailableError{
		IntegrationError: &IntegrationError{
			Message:       "service unavailable",
			Service:       service,
			CorrelationID: correlationID,
			Err:           err,
		},
	}
}

// TimeoutError indicates a request timeout.
type TimeoutError struct {
	*IntegrationError
}

// NewTimeoutError creates a new timeout error.
func NewTimeoutError(service, correlationID string, err error) *TimeoutError {
	return &TimeoutError{
		IntegrationError: &IntegrationError{
			Message:       "request timeout",
			Service:       service,
			CorrelationID: correlationID,
			Err:           err,
		},
	}
}

// ValidationError indicates a response validation error.
type ValidationError struct {
	*IntegrationError
}

// NewValidationError creates a new validation error.
func NewValidationError(service, correlationID string, err error) *ValidationError {
	return &ValidationError{
		IntegrationError: &IntegrationError{
			Message:       "response validation failed",
			Service:       service,
			CorrelationID: correlationID,
			Err:           err,
		},
	}
}

// AuthenticationError indicates an authentication error.
type AuthenticationError struct {
	*IntegrationError
}

// NewAuthenticationError creates a new authentication error.
func NewAuthenticationError(service, correlationID string, err error) *AuthenticationError {
	return &AuthenticationError{
		IntegrationError: &IntegrationError{
			Message:       "authentication failed",
			Service:       service,
			CorrelationID: correlationID,
			StatusCode:    401,
			Err:           err,
		},
	}
}

