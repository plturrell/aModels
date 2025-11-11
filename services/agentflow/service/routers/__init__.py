"""
FastAPI routers for the AgentFlow service.
"""

from .flows import router as flows_router  # noqa: F401
from .health import router as health_router  # noqa: F401
from .sgmi import router as sgmi_router  # noqa: F401
