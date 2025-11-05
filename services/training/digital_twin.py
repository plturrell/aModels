"""Digital twin simulation support for the training pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx


class DigitalTwinSimulator:
    """Facade for invoking the digital twin simulation environment.

    The simulator can operate in two modes:
    1. Remote invocation via ``DIGITAL_TWIN_SERVICE_URL``.
    2. Local heuristic simulation if no remote endpoint is configured.
    """

    def __init__(
        self,
        simulator_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.simulator_url = simulator_url or os.getenv("DIGITAL_TWIN_SERVICE_URL")
        self.logger = logger or logging.getLogger(__name__)
        self._client: Optional[httpx.Client] = None

        if self.simulator_url:
            self._client = httpx.Client(timeout=60.0)

    @property
    def is_enabled(self) -> bool:
        return bool(self.simulator_url or os.getenv("ENABLE_DIGITAL_TWIN", "false").lower() == "true")

    def simulate(
        self,
        pipeline_results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
        signavio_metadata: Optional[Dict[str, Any]] = None,
        scenario: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a simulation and return a summary record."""
        if self.simulator_url and self._client:
            return self._simulate_remote(
                pipeline_results, dataset_info, signavio_metadata, scenario
            )

        return self._simulate_local(pipeline_results, dataset_info, signavio_metadata, scenario)

    def _simulate_remote(
        self,
        pipeline_results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]],
        signavio_metadata: Optional[Dict[str, Any]],
        scenario: Optional[str],
    ) -> Dict[str, Any]:
        payload = {
            "scenario": scenario or os.getenv("DIGITAL_TWIN_SCENARIO", "default"),
            "pipeline": pipeline_results,
            "dataset": dataset_info or {},
            "signavio": signavio_metadata or {},
        }

        try:
            response = self._client.post(
                f"{self.simulator_url.rstrip('/')}/simulate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "status": "success",
                "mode": "remote",
                "impact_projection": data.get("impact_projection"),
                "anomalies": data.get("anomalies", []),
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Digital twin remote simulation failed: %s", exc)
            return {"status": "failed", "mode": "remote", "error": str(exc)}

    def _simulate_local(
        self,
        pipeline_results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]],
        signavio_metadata: Optional[Dict[str, Any]],
        scenario: Optional[str],
    ) -> Dict[str, Any]:
        """Fallback simulation driven by heuristic scoring."""
        extract_step = pipeline_results.get("steps", {}).get("extract", {})
        node_count = extract_step.get("nodes", 0)
        edge_count = extract_step.get("edges", 0)
        process_count = (signavio_metadata or {}).get("process_count", 0)
        dataset_files = (dataset_info or {}).get("files", [])

        coverage_score = 0.0
        if node_count:
            coverage_score = min(1.0, edge_count / max(node_count, 1))

        simulated_latency = round(0.5 + node_count * 0.0002 + process_count * 0.0005, 3)
        estimated_effort = max(1.0, len(dataset_files) * 2.5)

        return {
            "status": "success",
            "mode": "local",
            "scenario": scenario or os.getenv("DIGITAL_TWIN_SCENARIO", "default"),
            "coverage_score": coverage_score,
            "simulated_latency_hours": simulated_latency,
            "estimated_validation_effort_hours": estimated_effort,
            "processes_simulated": process_count,
        }

