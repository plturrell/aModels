"""LangSmith instrumentation for the training pipeline."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from langsmith import Client
except ImportError:  # pragma: no cover - dependency optional
    Client = None  # type: ignore[assignment]


class LangSmithTracer:
    """Lightweight helper that records pipeline runs to LangSmith."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.project_name = os.getenv("LANGSMITH_PROJECT", "aModels-training-pipeline")
        self.enabled = bool(
            Client
            and os.getenv("LANGSMITH_API_KEY")
            and os.getenv("ENABLE_LANGSMITH_LOGGING", "false").lower() == "true"
        )
        self._client: Optional[Client] = None
        if self.enabled and Client is not None:
            try:
                self._client = Client()
            except Exception as exc:  # pragma: no cover - client init failure
                self.logger.warning("Failed to initialize LangSmith client: %s", exc)
                self.enabled = False

    def record_run(
        self,
        project_id: str,
        system_id: Optional[str],
        results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a single run record to LangSmith."""
        if not self.enabled or self._client is None:
            return

        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        tags = ["amodels", "training"]
        if system_id:
            tags.append(f"system:{system_id}")

        try:
            self._client.create_run(
                id=run_id,
                name=f"training-pipeline-{project_id}",
                run_type="pipeline",
                project_name=self.project_name,
                inputs={
                    "project_id": project_id,
                    "system_id": system_id,
                },
                outputs={
                    "steps": results.get("steps", {}),
                    "domain_metrics": results.get("domain_metrics"),
                },
                tags=tags,
                start_time=now,
                end_time=now,
                metadata={
                    "dataset_files": (dataset_info or {}).get("files"),
                    "digital_twin": results.get("steps", {}).get("digital_twin"),
                },
            )
            self.logger.info("📡 Logged training run to LangSmith project %s", self.project_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to record LangSmith run: %s", exc)

