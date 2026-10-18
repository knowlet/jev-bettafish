#!/usr/bin/env python3
"""Headless QueryEngine runner used by workflow_dispatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path

# When executed as "python scripts/run_workflow_research.py", Python puts
# scripts/ (not the repository root) on sys.path. Add the repo root explicitly
# so package imports such as QueryEngine work both locally and in GitHub Actions.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _apply_generic_model_env(environment: MutableMapping[str, str] | None = None) -> None:
    environment = os.environ if environment is None else environment
    aliases = {
        "QUERY_ENGINE_API_KEY": "LLM_MODEL_API_KEY",
        "QUERY_ENGINE_BASE_URL": "LLM_MODEL_BASE_URL",
        "QUERY_ENGINE_MODEL_NAME": "LLM_MODEL_NAME",
    }
    # A key, URL and model belong to the same provider configuration. Filling
    # individual holes could send an engine-specific key to the generic URL.
    # Empty .env.example placeholders still permit the generic configuration.
    if any(environment.get(target) for target in aliases):
        # Pin an omitted URL to the SDK default before QueryEngine Settings
        # independently reads .env, which may contain an unrelated endpoint.
        environment.setdefault("QUERY_ENGINE_BASE_URL", "")
        return
    for target, source in aliases.items():
        environment[target] = environment.get(source, "")


def _preflight(environment: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environment is None else environment
    required = ["QUERY_ENGINE_API_KEY", "QUERY_ENGINE_MODEL_NAME", "TAVILY_API_KEY"]
    # System One is optional; a missing key uses the legacy LLM path.
    missing = [name for name in required if not environment.get(name, "").strip()]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))


def _prepare_environment(output_dir: Path) -> dict[str, str]:
    from SystemOne.config import load_environment

    environment = load_environment()
    _apply_generic_model_env(environment)
    _preflight(environment)
    environment.setdefault("OUTPUT_DIR", str(output_dir / "query_reports"))

    # QueryEngine constructs its Settings at import time. Promote only its
    # resolved configuration, after validation, so .env-only launches work
    # without importing unrelated .env credentials into the process.
    query_names = {
        "QUERY_ENGINE_API_KEY", "QUERY_ENGINE_BASE_URL", "QUERY_ENGINE_MODEL_NAME",
        "QUERY_ENGINE_PROVIDER", "TAVILY_API_KEY", "SEARCH_TIMEOUT",
        "SEARCH_CONTENT_MAX_LENGTH", "MAX_REFLECTIONS", "MAX_PARAGRAPHS",
        "MAX_SEARCH_RESULTS", "OUTPUT_DIR", "SAVE_INTERMEDIATE_STATES",
        "LLM_REQUEST_TIMEOUT", "QUERY_ENGINE_REQUEST_TIMEOUT",
        "OPENCODE_SESSION_ID", "OPENCODE_USER_AGENT",
    }
    for name in query_names:
        if name in environment:
            os.environ[name] = environment[name]
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    environment = _prepare_environment(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from QueryEngine.agent import DeepSearchAgent
    from SystemOne.client import get_system_one_client

    # Use the same dotenv-aware singleton as the decision call sites. These
    # flags describe configuration, not successful decision usage.
    system_one_client = get_system_one_client()
    stats_before = system_one_client.request_stats

    started = datetime.now(timezone.utc)
    agent = DeepSearchAgent()
    report = agent.research(args.query, save_report=False)
    finished = datetime.now(timezone.utc)
    stats_after = system_one_client.request_stats
    request_stats = {name: stats_after[name] - stats_before[name] for name in stats_before}
    agent_config = getattr(agent, "config", None)

    (output_dir / "result.md").write_text(report, encoding="utf-8")
    metadata = {
        "query": args.query,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "system_one_requested": system_one_client.requested,
        "system_one_available": bool(system_one_client.api_key),
        "system_one_enabled": system_one_client.enabled,
        "system_one_model": system_one_client.model,
        "system_one_requests_attempted": request_stats["attempted"],
        # HTTP/JSON success does not imply acceptance by a decision's gates.
        "system_one_responses_succeeded": request_stats["succeeded"],
        "system_one_requests_failed": request_stats["failed"],
        "llm_model": getattr(
            agent_config, "QUERY_ENGINE_MODEL_NAME", environment["QUERY_ENGINE_MODEL_NAME"],
        ),
        "max_reflections": int(getattr(
            agent_config, "MAX_REFLECTIONS", environment.get("MAX_REFLECTIONS", "2"),
        )),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
