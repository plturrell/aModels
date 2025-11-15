"""OpenTelemetry instrumentation for DeepAgents service."""

import os
import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resource import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.semconv.resource import ResourceAttributes

# OpenLLMetry - LLM observability
try:
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    OPENLLMETRY_AVAILABLE = True
except ImportError:
    OPENLLMETRY_AVAILABLE = False
    logger.warning("OpenLLMetry LangChain instrumentation not available. Install opentelemetry-instrumentation-langchain")

logger = logging.getLogger(__name__)

_tracer_provider: Optional[TracerProvider] = None


def init_otel_instrumentation(app=None):
    """Initialize OpenTelemetry instrumentation."""
    global _tracer_provider
    
    # Check if tracing is enabled
    if os.getenv("OTEL_TRACES_ENABLED", "false").lower() != "true":
        logger.info("OpenTelemetry tracing disabled")
        return
    
    try:
        # Create resource with service attributes
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: "deepagents-service",
            ResourceAttributes.SERVICE_VERSION: os.getenv("SERVICE_VERSION", "0.1.0"),
            ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("SERVICE_INSTANCE_ID", ""),
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
            "agent.framework.type": "deepagents",
        })
        
        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)
        
        # Configure span limits for full attribute capture
        _tracer_provider._span_limits.max_attributes = -1  # No limit
        _tracer_provider._span_limits.max_events = -1
        _tracer_provider._span_limits.max_links = -1
        _tracer_provider._span_limits.max_event_attributes = -1
        _tracer_provider._span_limits.max_link_attributes = -1
        _tracer_provider._span_limits.max_attribute_value_length = -1
        
        # Set up OTLP exporter
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://telemetry-exporter:4318")
        exporter = OTLPSpanExporter(
            endpoint=f"{otlp_endpoint}/v1/traces",
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(exporter)
        _tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)
        
        # Instrument FastAPI if app is provided
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented with OpenTelemetry")
        
        # Instrument HTTPX client
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX client instrumented with OpenTelemetry")
        
        # Instrument LangChain with OpenLLMetry for LLM observability
        if OPENLLMETRY_AVAILABLE:
            try:
                LangchainInstrumentor().instrument()
                logger.info("LangChain instrumented with OpenLLMetry")
            except Exception as e:
                logger.warning(f"Failed to instrument LangChain with OpenLLMetry: {e}")
        else:
            logger.warning("OpenLLMetry LangChain instrumentation not available")
        
        logger.info(f"OpenTelemetry instrumentation initialized (endpoint: {otlp_endpoint})")
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)


def get_tracer(name: str = "deepagents-service"):
    """Get a tracer instance."""
    if _tracer_provider is None:
        return trace.NoOpTracerProvider().get_tracer(name)
    return trace.get_tracer(name)


def shutdown_otel():
    """Shutdown OpenTelemetry instrumentation."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        logger.info("OpenTelemetry instrumentation shut down")

