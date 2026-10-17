"""Offline regressions for decision credentials and workflow preflight."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from SystemOne import client, decisions
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


def run_offline_research(monkeypatch, output_dir, outcome="accepted"):
    """Exercise the real runner/client/decision with only research and HTTP mocked."""
    os.environ.update(
        QUERY_ENGINE_API_KEY="llm-key", QUERY_ENGINE_MODEL_NAME="model",
        TAVILY_API_KEY="search-key",
    )
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        request = client.httpx.Request("POST", url)
        if outcome == "http_error":
            return client.httpx.Response(503, request=request)
        confidence = 0.1 if outcome == "low_confidence" else 0.99
        return client.httpx.Response(200, request=request, json={"answers": {
            "search_tool": {"choice": "basic_search_news", "confidence": confidence},
        }})

    class FakeAgent:
        def research(self, query, save_report):
            assert save_report is False
            decision = decisions.choose_query_search_tool(
                input_data={"title": query}, generated_query=query, phase="initial",
            )
            return "system_one" if decision is not None else "legacy"

    module = ModuleType("QueryEngine.agent")
    module.DeepSearchAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "QueryEngine.agent", module)
    monkeypatch.setattr(client, "_SINGLETON", None)
    monkeypatch.setattr(client.httpx, "post", post)
    monkeypatch.setattr(sys, "argv", [
        "run_workflow_research.py", "--query", "offline-query",
        "--output-dir", str(output_dir),
    ])
    assert runner.main() == 0
    metadata_text = (output_dir / "metadata.json").read_text(encoding="utf-8")
    assert "llm-key" not in metadata_text
    assert "decision-test-key" not in metadata_text
    assert "search-key" not in metadata_text
    return (
        json.loads(metadata_text),
        (output_dir / "result.md").read_text(encoding="utf-8"),
        requests,
    )


@pytest.mark.parametrize("toggle,key,requested,available,enabled", [
    (None, None, True, False, False),
    ("true", None, True, False, False),
    ("true", "", True, False, False),
    ("true", "decision-test-key", True, True, True),
    ("false", "decision-test-key", False, True, False),
    ("false", None, False, False, False),
])
def test_runner_metadata_records_effective_client_availability(
    isolated_env, monkeypatch, toggle, key, requested, available, enabled,
):
    cwd, _ = isolated_env
    if toggle is not None:
        os.environ["SYSTEM_ONE_ENABLED"] = toggle
    if key is not None:
        os.environ["DECISION_MODEL_API_KEY"] = key
    metadata, report, requests = run_offline_research(monkeypatch, cwd / "artifacts")
    assert metadata["system_one_enabled"] is enabled
    assert metadata["system_one_requested"] is requested
    assert metadata["system_one_available"] is available
    assert metadata["system_one_model"] == "inception/mercury-decide:free"
    assert report == ("system_one" if enabled else "legacy")
    assert len(requests) == int(enabled)


@pytest.mark.parametrize("location", ["cwd", "project"])
@pytest.mark.parametrize("toggle", ["true", "false"])
def test_runner_metadata_uses_dotenv_client_configuration(
    isolated_env, monkeypatch, location, toggle,
):
    cwd, project = isolated_env
    directory = cwd if location == "cwd" else project
    (directory / ".env").write_text(
        "DECISION_MODEL_API_KEY=decision-test-key\n"
        "DECISION_MODEL_NAME=dotenv-model\n"
        f"SYSTEM_ONE_ENABLED={toggle}\n", encoding="utf-8",
    )
    metadata, report, requests = run_offline_research(monkeypatch, cwd / "artifacts")
    assert metadata["system_one_requested"] is (toggle == "true")
    assert metadata["system_one_available"] is True
    assert metadata["system_one_enabled"] is (toggle == "true")
    assert metadata["system_one_model"] == "dotenv-model"
    assert report == ("system_one" if toggle == "true" else "legacy")
    assert all(request["model"] == "dotenv-model" for request in requests)


@pytest.mark.parametrize("outcome", ["http_error", "low_confidence"])
def test_runner_enabled_metadata_does_not_claim_successful_usage(
    isolated_env, monkeypatch, outcome,
):
    cwd, _ = isolated_env
    os.environ["DECISION_MODEL_API_KEY"] = "decision-test-key"
    metadata, report, requests = run_offline_research(
        monkeypatch, cwd / "artifacts", outcome=outcome,
    )
    assert metadata["system_one_requested"] is True
    assert metadata["system_one_available"] is True
    assert metadata["system_one_enabled"] is True
    assert report == "legacy"
    assert len(requests) == 1


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



@pytest.mark.parametrize("http_error", [False, True])
@pytest.mark.parametrize("trace_failure", ["directory", "parent_file", "permission"])
def test_trace_failure_preserves_decision_or_fallback(
    monkeypatch, tmp_path, http_error, trace_failure,
):
    monkeypatch.setattr(client, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("SYSTEM_ONE_ENABLED", "true")
    if trace_failure == "directory":
        trace_path = tmp_path
    elif trace_failure == "parent_file":
        parent = tmp_path / "not-a-directory"
        parent.write_text("occupied")
        trace_path = parent / "trace.jsonl"
    else:
        trace_path = tmp_path / "trace.jsonl"
        original_open = client.Path.open

        def denied_open(path, *args, **kwargs):
            if path == trace_path:
                raise PermissionError("trace write denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(client.Path, "open", denied_open)
    monkeypatch.setenv("SYSTEM_ONE_TRACE_PATH", str(trace_path))
    expected = {"answers": {"continue": {"noul": 0.8}}}

    def post(url, **kwargs):
        return client.httpx.Response(
            503 if http_error else 200,
            request=client.httpx.Request("POST", url),
            json=expected,
        )

    monkeypatch.setattr(client.httpx, "post", post)
    configured = client.SystemOneClient(api_key="offline-test-key")
    result = configured.evaluate(state={}, questions={}, decision_id="test.trace")
    assert result == (None if http_error else expected)


@pytest.mark.parametrize("http_error", [False, True])
def test_trace_still_records_success_and_failure(monkeypatch, tmp_path, http_error):
    import json

    monkeypatch.setattr(client, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("SYSTEM_ONE_ENABLED", "true")
    trace_path = tmp_path / "nested" / "trace.jsonl"
    monkeypatch.setenv("SYSTEM_ONE_TRACE_PATH", str(trace_path))
    expected = {"answers": {"continue": {"noul": 0.8}}}

    def post(url, **kwargs):
        return client.httpx.Response(
            503 if http_error else 200,
            request=client.httpx.Request("POST", url),
            json=expected,
        )

    monkeypatch.setattr(client.httpx, "post", post)
    result = client.SystemOneClient(api_key="offline-test-key").evaluate(
        state={}, questions={}, decision_id="test.trace",
    )
    assert result == (None if http_error else expected)
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["ok"] is (not http_error)
    assert records[0]["decision_id"] == "test.trace"
