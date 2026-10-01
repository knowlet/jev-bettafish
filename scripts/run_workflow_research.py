#!/usr/bin/env python3
"""Headless QueryEngine runner used by workflow_dispatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# When executed as "python scripts/run_workflow_research.py", Python puts
# scripts/ (not the repository root) on sys.path. Add the repo root explicitly
# so package imports such as QueryEngine work both locally and in GitHub Actions.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _apply_generic_model_env() -> None:
    aliases = {
        "QUERY_ENGINE_API_KEY": "MODEL_API_KEY",
        "QUERY_ENGINE_BASE_URL": "MODEL_BASE_URL",
        "QUERY_ENGINE_MODEL_NAME": "MODEL_NAME",
    }
    for target, source in aliases.items():
        if not os.getenv(target) and os.getenv(source):
            os.environ[target] = os.environ[source]


def _is_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _preflight() -> None:
    required = ["QUERY_ENGINE_API_KEY", "QUERY_ENGINE_MODEL_NAME", "TAVILY_API_KEY"]
    if _is_enabled("SYSTEM_ONE_ENABLED", True):
        if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("TYPESAFE_API_KEY"):
            required.append("OPENROUTER_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OUTPUT_DIR", str(output_dir / "query_reports"))

    _apply_generic_model_env()
    _preflight()

    from QueryEngine.agent import DeepSearchAgent

    started = datetime.now(timezone.utc)
    agent = DeepSearchAgent()
    report = agent.research(args.query, save_report=False)
    finished = datetime.now(timezone.utc)

    (output_dir / "result.md").write_text(report, encoding="utf-8")
    metadata = {
        "query": args.query,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "system_one_enabled": _is_enabled("SYSTEM_ONE_ENABLED", True),
        "system_one_model": os.getenv(
            "SYSTEM_ONE_MODEL", "inception/mercury-decide:free"
        ),
        "llm_model": os.getenv("QUERY_ENGINE_MODEL_NAME"),
        "max_reflections": int(os.getenv("MAX_REFLECTIONS", "2")),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
