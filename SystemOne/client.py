"""Minimal System One HTTP client with fail-open behavior.

Default endpoint is the OpenRouter Decisions API
(https://openrouter.ai/api/alpha/decisions) with
model `inception/mercury-decide:free`.

Resolution order (first set wins):
  key:   explicit arg > DECISION_MODEL_API_KEY > OPENROUTER_API_KEY
         > TYPESAFE_API_KEY (legacy)
  url:   explicit arg > SYSTEM_ONE_URL (legacy) > DECISION_MODEL_BASE_URL
         > OpenRouter default
  model: explicit arg > SYSTEM_ONE_MODEL (legacy) > DECISION_MODEL_NAME
         > mercury-decide default
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import httpx
from loguru import logger


_TRACE_LOCK = threading.Lock()
_SINGLETON: Optional["SystemOneClient"] = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


class SystemOneClient:
    """Small wrapper around POST /v1/systemone.

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
        self.api_key = (
            api_key
            or os.getenv("DECISION_MODEL_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("TYPESAFE_API_KEY", "")
        )
        self.url = (
            url
            or os.getenv("SYSTEM_ONE_URL")
            or os.getenv("DECISION_MODEL_BASE_URL")
            or "https://openrouter.ai/api/alpha/decisions"
        )
        self.model = (
            model
            or os.getenv("SYSTEM_ONE_MODEL")
            or os.getenv("DECISION_MODEL_NAME")
            or "inception/mercury-decide:free"
        )
        try:
            self.timeout = float(
                timeout if timeout is not None else os.getenv("SYSTEM_ONE_TIMEOUT", "30")
            )
        except (TypeError, ValueError):
            self.timeout = 30.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and _env_bool("SYSTEM_ONE_ENABLED", True)

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
        try:
            response = httpx.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("System One response is not a JSON object")
            self._trace(
                decision_id=decision_id,
                state=state,
                result=result,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return result
        except Exception as exc:
            logger.warning(
                f"System One decision '{decision_id}' failed; using LLM fallback: {exc}"
            )
            self._trace(
                decision_id=decision_id,
                state=state,
                result=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
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
        trace_path = os.getenv("SYSTEM_ONE_TRACE_PATH")
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
        if error:
            record["error"] = error[:500]
        if _env_bool("SYSTEM_ONE_TRACE_STATE", False):
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
