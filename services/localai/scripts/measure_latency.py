#!/usr/bin/env python3
"""Measure latency for key LocalAI transformer domains."""

import argparse
import json
import time
import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_TIMEOUT = 30
DEFAULT_OUTPUT = "logs/localai_router_latency_latest.json"

FINANCE_DOMAINS = {"0x5D1A-SubledgerAgent"}
FINANCE_MAX_TOKENS = 128
VECTOR_DOMAINS = {"0x3579-VectorProcessingAgent"}
VECTOR_MAX_TOKENS = 192
DEFAULT_MAX_TOKENS = 256

ENABLE_GEMMA7B = os.getenv("ENABLE_GEMMA7B", "").lower() in {"1", "true", "yes"}

DEFAULT_TESTS: List[Dict[str, str]] = [
    {
        "domain": "0x3579-VectorProcessingAgent",
        "prompt": "Summarize why vector databases help retrieval pipelines in two sentences.",
    },
    {
        "domain": "0x5D1A-SubledgerAgent",
        "prompt": "List three key considerations for subledger reconciliation.",
    },
    {
        "domain": "0xG2B-GemmaAssistant",
        "prompt": "Give a three-word greeting.",
    },
]

if ENABLE_GEMMA7B:
    DEFAULT_TESTS.append(
        {
            "domain": "0xG7B-Gemma7BAssistant",
            "prompt": "Offer a succinct two-word greeting.",
        }
    )


def _max_tokens_for_domain(domain: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    if domain in FINANCE_DOMAINS:
        return FINANCE_MAX_TOKENS
    if domain in VECTOR_DOMAINS:
        return VECTOR_MAX_TOKENS
    return DEFAULT_MAX_TOKENS


def _build_payload(domain: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": domain,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }


def run_probe(
    endpoint: str,
    timeout: float,
    tests: List[Dict[str, str]],
    default_max_tokens: Optional[int],
) -> Dict[str, Any]:
    session = requests.Session()
    session.verify = False
    results = []

    for spec in tests:
        domain = spec["domain"]
        prompt = spec["prompt"]
        max_tokens = _max_tokens_for_domain(domain, default_max_tokens)
        payload = _build_payload(domain, prompt, max_tokens)
        started = time.time()
        status_code: Optional[int] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        tokens_per_second: Optional[float] = None
        timings_ms: Optional[Dict[str, float]] = None
        error: Optional[str] = None

        try:
            response = session.post(endpoint, json=payload, timeout=timeout)
            status_code = response.status_code
            if response.ok:
                data = response.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                total_tokens = usage.get("total_tokens")
                timings = data.get("timings") or {}
                timings_ms = {k: float(v) for k, v in timings.items()}
                predicted_ms = timings.get("predicted_ms")
                if isinstance(predicted_ms, (int, float)) and completion_tokens:
                    seconds = predicted_ms / 1000.0
                    if seconds > 0:
                        tokens_per_second = completion_tokens / seconds
                elif completion_tokens:
                    elapsed = time.time() - started
                    if elapsed > 0:
                        tokens_per_second = completion_tokens / elapsed
            else:
                error = response.text
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)

        latency_sec = time.time() - started
        results.append(
            {
                "domain": domain,
                "prompt": prompt,
                "status_code": status_code,
                "latency_sec": latency_sec,
                "max_tokens": max_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "timings_ms": timings_ms,
                "error": error,
            }
        )

    return {"timestamp": time.time(), "endpoint": endpoint, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help="LocalAI chat completions endpoint"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout per request (seconds)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the latest latency sample JSON",
    )
    parser.add_argument(
        "--history",
        help="Optional path to append the new sample into an array history",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        dest="max_tokens_override",
        help="Override max tokens for all domains (otherwise per-domain defaults apply)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(
        endpoint=args.endpoint,
        timeout=args.timeout,
        tests=DEFAULT_TESTS,
        default_max_tokens=args.max_tokens_override,
    )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if args.history:
        try:
            with open(args.history, "r", encoding="utf-8") as handle:
                history = json.load(handle)
                if not isinstance(history, list):
                    history = []
        except FileNotFoundError:
            history = []
        history.append(payload)
        with open(args.history, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
