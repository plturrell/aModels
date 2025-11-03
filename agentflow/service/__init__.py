"""
AgentFlow FastAPI service package.

This module wires the FastAPI application, persistence layers, and Langflow
integration used to expose flow synchronisation and execution APIs.
"""

from .config import get_settings  # noqa: F401
