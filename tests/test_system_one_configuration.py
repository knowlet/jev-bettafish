"""Offline regressions for decision credentials and workflow preflight."""

import os
from pathlib import Path
import subprocess

import pytest

from SystemOne import client
from scripts import run_workflow_research as runner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.chdir(tmp_path)
    # Keep the project fallback isolated from any developer credentials.
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(client, "__file__", str(project / "SystemOne" / "client.py"))
    return tmp_path, project


@pytest.mark.parametrize("location", ["cwd", "project"])
def test_client_loads_canonical_dotenv_values(isolated_env, location):
    cwd, project = isolated_env
    directory = cwd if location == "cwd" else project
    (directory / ".env").write_text(
        "DECISION_MODEL_API_KEY=dotenv-key\n"
        "DECISION_MODEL_BASE_URL=https://example.invalid/decisions\n"
        "DECISION_MODEL_NAME=test-model\n"
        "SYSTEM_ONE_TIMEOUT=12\n", encoding="utf-8"
    )
    configured = client.SystemOneClient()
    assert configured.api_key == "dotenv-key"
    assert configured.url == "https://example.invalid/decisions"
    assert configured.model == "test-model"
    assert configured.timeout == 12
    assert configured.enabled


def test_cwd_dotenv_and_process_environment_take_precedence(isolated_env):
    cwd, project = isolated_env
    (project / ".env").write_text("DECISION_MODEL_API_KEY=project-key\n")
    (cwd / ".env").write_text(
        "DECISION_MODEL_API_KEY=cwd-key\nDECISION_MODEL_NAME=cwd-model\n"
        "SYSTEM_ONE_ENABLED=false\n"
    )
    os.environ["DECISION_MODEL_API_KEY"] = "process-key"
    configured = client.SystemOneClient()
    assert configured.api_key == "process-key"
    assert configured.model == "cwd-model"
    assert not configured.enabled
    assert client.SystemOneClient(api_key="explicit-key").api_key == "explicit-key"


def test_legacy_credentials_are_not_sent_to_new_provider(isolated_env, monkeypatch):
    cwd, _ = isolated_env
    (cwd / ".env").write_text(
        "TYPESAFE_API_KEY=legacy-key\nOPENROUTER_API_KEY=old-name\n"
    )
    def unexpected_request(*args, **kwargs):
        pytest.fail("A missing canonical key must not make a request")
    monkeypatch.setattr(client.httpx, "post", unexpected_request)
    configured = client.SystemOneClient()
    assert not configured.enabled
    assert configured.evaluate(state={}, questions={}, decision_id="test") is None


@pytest.mark.parametrize("enabled", [None, "true", "false"])
def test_runner_allows_missing_optional_decision_key(isolated_env, enabled):
    os.environ.update(QUERY_ENGINE_API_KEY="llm-key", QUERY_ENGINE_MODEL_NAME="model", TAVILY_API_KEY="search-key")
    if enabled is not None:
        os.environ["SYSTEM_ONE_ENABLED"] = enabled
    runner._preflight()


@pytest.mark.parametrize("missing", ["QUERY_ENGINE_API_KEY", "QUERY_ENGINE_MODEL_NAME", "TAVILY_API_KEY"])
def test_runner_still_requires_llm_and_search_configuration(isolated_env, missing):
    os.environ.update(QUERY_ENGINE_API_KEY="llm-key", QUERY_ENGINE_MODEL_NAME="model", TAVILY_API_KEY="search-key")
    del os.environ[missing]
    with pytest.raises(RuntimeError, match=missing):
        runner._preflight()


def workflow_preflight_script():
    workflow = (ROOT / ".github/workflows/manual-research.yml").read_text()
    step = workflow.split("      - name: Validate required configuration\n", 1)[1]
    run = step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
    return "\n".join(line[10:] for line in run.splitlines())


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_workflow_allows_missing_optional_decision_key(enabled):
    result = subprocess.run(["bash", "-e", "-c", workflow_preflight_script()], env={
        "LLM_MODEL_API_KEY": "llm-key", "LLM_MODEL_NAME": "model",
        "TAVILY_API_KEY": "search-key", "SYSTEM_ONE_ENABLED": enabled,
    }, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing", ["LLM_MODEL_API_KEY", "LLM_MODEL_NAME", "TAVILY_API_KEY"])
def test_workflow_still_requires_llm_and_search_configuration(missing):
    env = {"LLM_MODEL_API_KEY": "llm-key", "LLM_MODEL_NAME": "model", "TAVILY_API_KEY": "search-key"}
    del env[missing]
    result = subprocess.run(["bash", "-e", "-c", workflow_preflight_script()], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert missing in result.stdout
