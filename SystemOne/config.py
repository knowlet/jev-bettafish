"""Read System One configuration without changing the process environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv.main import DotEnv


def load_environment() -> dict[str, str]:
    """Return the selected dotenv values overlaid by the process environment.

    Select the working-directory .env when it exists, otherwise the project
    .env. The files are alternatives, not a per-key fallback chain. Parsing
    must remain read-only: exporting unrelated engine credentials would turn
    old file values into process overrides after a configuration reload.
    """
    if os.getenv("PYTHON_DOTENV_DISABLED", "").casefold() in {
        "1", "true", "t", "yes", "y",
    }:
        return dict(os.environ)
    cwd_env = Path.cwd() / ".env"
    env_file = (
        cwd_env
        if cwd_env.exists()
        else Path(__file__).resolve().parents[1] / ".env"
    )
    # dotenv_values() hardcodes override=True during ${VAR} interpolation.
    # DotEnv.dict() preserves load_dotenv(override=False)'s process priority
    # inside references, without calling set_as_environment_variables().
    values = DotEnv(env_file, override=False, encoding="utf-8").dict()
    return {
        **{name: value for name, value in values.items() if value is not None},
        **os.environ,
    }


def env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a tuning setting with the same dotenv/process precedence."""
    return load_environment().get(name, default)
