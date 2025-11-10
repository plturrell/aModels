"""Structured error types for integrations."""

from __future__ import annotations


class IntegrationError(Exception):
    """Base exception for integration errors."""
    
    def __init__(self, message: str, service: str | None = None, correlation_id: str | None = None):
        """Initialize integration error.
        
        Args:
            message: Error message
            service: Service name that caused the error
            correlation_id: Correlation ID for tracing
        """
        super().__init__(message)
        self.service = service
        self.correlation_id = correlation_id


class ServiceUnavailableError(IntegrationError):
    """Service is unavailable (circuit breaker open, health check failed, etc.)."""
    pass


class TimeoutError(IntegrationError):
    """Request timeout."""
    pass


class ValidationError(IntegrationError):
    """Response validation error."""
    pass


class AuthenticationError(IntegrationError):
    """Authentication error."""
    pass

