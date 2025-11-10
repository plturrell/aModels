"""
Structured logging configuration with correlation IDs.
"""
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Optional
import json

# Context variable for correlation ID (thread-safe in async)
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""
    
    def filter(self, record):
        record.correlation_id = correlation_id_ctx.get() or "no-correlation-id"
        return True


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, 'correlation_id', 'no-correlation-id'),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger with correlation ID support.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def _log(self, level: int, message: str, **kwargs):
        """Log with extra fields."""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)
        
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
):
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatter for structured logs
        log_file: Optional log file path
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    console_handler.addFilter(correlation_filter)
    
    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(correlation_id)s] - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.addFilter(correlation_filter)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    # Set uvicorn loggers to use same config
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    corr_id = correlation_id_ctx.get()
    if corr_id is None:
        corr_id = str(uuid.uuid4())
        correlation_id_ctx.set(corr_id)
    return corr_id


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context."""
    correlation_id_ctx.set(correlation_id)


def clear_correlation_id():
    """Clear correlation ID from current context."""
    correlation_id_ctx.set(None)


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger(name)
