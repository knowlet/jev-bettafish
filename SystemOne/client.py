"""Minimal System One HTTP client with fail-open behavior.

Endpoint, model and key are env-driven (breaking change: no legacy
fallbacks):

  key:   DECISION_MODEL_API_KEY (unset -> client disabled -> legacy LLM path)
  url:   DECISION_MODEL_BASE_URL (default: OpenRouter Decisions API)
  model: DECISION_MODEL_NAME (default: inception/mercury-decide:free)
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import httpx
from loguru import logger

from .config import load_environment


_TRACE_LOCK = threading.Lock()
_SINGLETON: Optional["SystemOneClient"] = None


def _config_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


class SystemOneClient:
    """Small wrapper around the OpenRouter Decisions API.

    This layer is intentionally non-critical: missing credentials, transport
    errors, response changes, or low-confidence decisions must not break the
    original BettaFish reasoning path.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._config = load_environment()
        # An explicit empty key disables the client, even when ambient
        # credentials exist. Never borrow credentials from another service.
        configured_key = (
            api_key
            if api_key is not None
            else self._config.get("DECISION_MODEL_API_KEY", "")
        )
        self.api_key = configured_key.strip()
        self.url = url if url is not None else (
            self._config.get("DECISION_MODEL_BASE_URL")
            or "https://openrouter.ai/api/alpha/decisions"
        )
        self.model = model if model is not None else (
            self._config.get("DECISION_MODEL_NAME")
            or "inception/mercury-decide:free"
        )
        configured_timeout = (
            timeout
            if timeout is not None
            else self._config.get("SYSTEM_ONE_TIMEOUT", "30")
        )
        try:
            self.timeout = float(configured_timeout)
        except (TypeError, ValueError):
            self.timeout = 30.0
        self._stats_lock = threading.Lock()
        self._request_stats = {"attempted": 0, "succeeded": 0, "failed": 0}
        if self.requested and not self.api_key:
            logger.warning(
                "System One requested but DECISION_MODEL_API_KEY is missing or blank; "
                "using the legacy LLM path. Legacy credential names are not read. "
                "Configure a credential issued for DECISION_MODEL_BASE_URL; "
                "do not reuse a TypeSafe credential at OpenRouter."
            )

    @property
    def requested(self) -> bool:
        return _config_bool(self._config, "SYSTEM_ONE_ENABLED", True)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and self.requested

    @property
    def request_stats(self) -> Dict[str, int]:
        """Snapshot transport counts; a valid response is not an accepted decision."""
        with self._stats_lock:
            return dict(self._request_stats)

    def _count_request(self, outcome: str) -> None:
        with self._stats_lock:
            self._request_stats[outcome] += 1

    def evaluate(
        self,
        *,
        state: Dict[str, Any],
        questions: Mapping[str, Dict[str, Any]],
        decision_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "state": state,
            "questions": dict(questions),
        }
        started = time.perf_counter()
        self._count_request("attempted")
        try:
            response = httpx.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
                follow_redirects=False,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("System One response is not a JSON object")
            self._count_request("succeeded")
            self._trace(
                decision_id=decision_id,
                state=state,
                result=result,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return result
        except Exception as exc:
            self._count_request("failed")
            # Transport errors can contain Authorization values or URL
            # credentials. Keep useful status without copying raw exceptions
            # into logs or downloadable research traces.
            error = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                error += f" (HTTP {exc.response.status_code})"
            logger.warning(
                f"System One decision '{decision_id}' failed; using LLM fallback: {error}"
            )
            self._trace(
                decision_id=decision_id,
                state=state,
                result=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )
            return None

    def _trace(
        self,
        *,
        decision_id: str,
        state: Dict[str, Any],
        result: Optional[Dict[str, Any]],
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        # Optional diagnostics must not discard a valid decision or prevent
        # the caller from falling back after a failed request.
        try:
            self._write_trace(
                decision_id=decision_id,
                state=state,
                result=result,
                duration_ms=duration_ms,
                error=error,
            )
        except Exception as exc:
            logger.warning(
                f"System One trace for '{decision_id}' could not be written: {exc}"
            )

    def _write_trace(
        self,
        *,
        decision_id: str,
        state: Dict[str, Any],
        result: Optional[Dict[str, Any]],
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        trace_path = self._config.get("SYSTEM_ONE_TRACE_PATH")
        if not trace_path:
            return

        state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_id": decision_id,
            "model": self.model,
            "ok": error is None,
            "duration_ms": round(duration_ms, 2),
            "state_sha256": hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
            "state_keys": sorted(state.keys()),
        }
        if result is not None:
            answers = result.get("answers")
            if answers is None and isinstance(result.get("data"), dict):
                answers = result["data"].get("answers")
            record["answers"] = answers
            if "usage" in result:
                record["usage"] = result["usage"]
            for response_field, trace_field in (
                ("model", "response_model"), ("provider", "provider"),
            ):
                value = result.get(response_field)
                if isinstance(value, str):
                    record[trace_field] = value
        if error:
            record["error"] = error[:500]
        if _config_bool(self._config, "SYSTEM_ONE_TRACE_STATE", False):
            record["state_preview"] = state_json[:4000]

        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_system_one_client() -> SystemOneClient:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SystemOneClient()
    return _SINGLETON

