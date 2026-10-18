"""Offline headless-startup regressions using the real QueryEngine settings/LLM."""

import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import ModuleType

import httpx
import openai
import pytest

from SystemOne import client, config as environment_config, decisions
from scripts import run_workflow_research as runner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(client, "_SINGLETON", None)
    # Any unexpected provider dispatch fails before opening a real connection.
    def forbidden_connect(*args, **kwargs):
        pytest.fail("Offline runner regression attempted a real connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    return tmp_path


def write_env(directory, **values):
    (directory / ".env").write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def install_offline_agent(monkeypatch, *, outcome="accepted"):
    """Retain settings resolution and both credential-bearing HTTP boundaries."""
    captured = {"llm": [], "decision": [], "configs": []}
    real_openai = openai.OpenAI

    def llm_response(request):
        captured["llm"].append(request)
        return httpx.Response(200, request=request, json={
            "id": "offline-response", "object": "chat.completion", "created": 0,
            "model": "offline-model",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "offline report"},
            }],
        })

    def offline_openai(**kwargs):
        return real_openai(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(llm_response)),
        )

    def decision_response(url, **kwargs):
        request = httpx.Request(
            "POST", url, headers=kwargs["headers"], json=kwargs["json"],
        )
        captured["decision"].append(request)
        if outcome == "http_error":
            return httpx.Response(503, request=request)
        if outcome == "malformed":
            return httpx.Response(200, request=request, json={"answers": {}})
        confidence = 0.1 if outcome == "low_confidence" else 0.99
        return httpx.Response(200, request=request, json={"answers": {
            "search_tool": {"choice": "basic_search_news", "confidence": confidence},
        }})

    monkeypatch.setattr(openai, "OpenAI", offline_openai)
    monkeypatch.setattr(client.httpx, "post", decision_response)

    class OfflineAgent:
        def __init__(self):
            # The actual settings file validates import-time .env/env behavior.
            self.config = runpy.run_path(
                str(ROOT / "QueryEngine/utils/config.py"),
                run_name="offline_query_settings",
            )["settings"]
            captured["configs"].append(self.config)
            llm_class = runpy.run_path(
                str(ROOT / "QueryEngine/llms/base.py"), run_name="offline_query_llm",
            )["LLMClient"]
            self.llm = llm_class(
                api_key=self.config.QUERY_ENGINE_API_KEY,
                base_url=self.config.QUERY_ENGINE_BASE_URL,
                model_name=self.config.QUERY_ENGINE_MODEL_NAME,
            )

        def research(self, query, save_report):
            assert save_report is False
            decision = decisions.choose_query_search_tool(
                input_data={"title": query}, generated_query=query, phase="initial",
            )
            report = self.llm.invoke("Write an offline report", query)
            self.llm.client.close()
            return ("decision: " if decision is not None else "fallback: ") + report

    module = ModuleType("QueryEngine.agent")
    module.DeepSearchAgent = OfflineAgent
    monkeypatch.setitem(sys.modules, "QueryEngine.agent", module)
    return captured


def run_research(monkeypatch, directory):
    output = directory / "artifacts"
    monkeypatch.setattr(sys, "argv", [
        "run_workflow_research.py", "--query", "offline query",
        "--output-dir", str(output),
    ])
    assert runner.main() == 0
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    report = (output / "result.md").read_text(encoding="utf-8")
    return metadata, report


@pytest.mark.parametrize("file_output", ["", "dotenv-reports"])
@pytest.mark.parametrize("process_output,expected_directory", [
    (None, "artifacts/query_reports"),
    ("", "artifacts/query_reports"),
    ("   ", "artifacts/query_reports"),
    ("process-reports", "process-reports"),
])
def test_runner_report_directory_uses_cli_default_or_nonblank_process_override(
    offline_runner, monkeypatch, file_output, process_output, expected_directory,
):
    write_env(
        offline_runner, QUERY_ENGINE_API_KEY="offline-llm-key",
        QUERY_ENGINE_MODEL_NAME="offline-model", TAVILY_API_KEY="offline-search-key",
        OUTPUT_DIR=file_output,
    )
    if process_output is not None:
        os.environ["OUTPUT_DIR"] = process_output
    captured = install_offline_agent(monkeypatch)
    metadata, report = run_research(monkeypatch, offline_runner)

    # Assert the real consumer's Settings, not just the runner's merged dict.
    # A blank value would make DeepSearchAgent's directory creation fail.
    config, = captured["configs"]
    assert config.OUTPUT_DIR.strip()
    assert Path(config.OUTPUT_DIR).resolve() == offline_runner / expected_directory
    assert metadata["llm_model"] == "offline-model"
    assert report == "fallback: offline report"


@pytest.mark.parametrize("group", ["engine", "generic"])
def test_dotenv_only_startup_uses_matching_llm_and_decision_credentials(
    offline_runner, monkeypatch, group,
):
    values = {
        "QUERY_ENGINE_API_KEY": "", "QUERY_ENGINE_BASE_URL": "", "QUERY_ENGINE_MODEL_NAME": "",
        "TAVILY_API_KEY": "offline-search-key",
        "DECISION_MODEL_API_KEY": "offline-decision-key",
        "DECISION_MODEL_BASE_URL": "https://decisions.example.invalid/evaluate",
        "DECISION_MODEL_NAME": "offline-decision-model",
        "SYSTEM_ONE_ENABLED": "true", "MAX_REFLECTIONS": "3",
        # Loading a decision key must not inject unrelated SDK credentials.
        "OPENAI_API_KEY": "unrelated-legacy-key",
        "OPENAI_BASE_URL": "https://unrelated.example.invalid/v1",
    }
    if group == "engine":
        values.update(
            QUERY_ENGINE_API_KEY="offline-llm-key",
            QUERY_ENGINE_BASE_URL="https://llm.example.invalid/v1",
            QUERY_ENGINE_MODEL_NAME="offline-llm-model",
        )
    else:
        values.update(
            LLM_MODEL_API_KEY="offline-llm-key",
            LLM_MODEL_BASE_URL="https://llm.example.invalid/v1",
            LLM_MODEL_NAME="offline-llm-model",
        )
    write_env(offline_runner, **values)
    captured = install_offline_agent(monkeypatch)
    metadata, report = run_research(monkeypatch, offline_runner)

    assert report == "decision: offline report"
    assert metadata["llm_model"] == "offline-llm-model"
    assert metadata["max_reflections"] == 3
    assert metadata["system_one_model"] == "offline-decision-model"
    assert metadata["system_one_requests_attempted"] == 1
    assert metadata["system_one_responses_succeeded"] == 1
    assert metadata["system_one_requests_failed"] == 0
    llm_request, = captured["llm"]
    decision_request, = captured["decision"]
    assert str(llm_request.url) == "https://llm.example.invalid/v1/chat/completions"
    assert llm_request.headers["Authorization"] == "Bearer offline-llm-key"
    assert str(decision_request.url) == "https://decisions.example.invalid/evaluate"
    assert decision_request.headers["Authorization"] == "Bearer offline-decision-key"
    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_BASE_URL" not in os.environ
    assert "DECISION_MODEL_API_KEY" not in os.environ
    assert not any(value in json.dumps(metadata) for value in (
        "offline-llm-key", "offline-decision-key", "offline-search-key", "unrelated-legacy-key",
    ))


@pytest.mark.parametrize("engine_values", [
    {"QUERY_ENGINE_API_KEY": "engine-only-key"},
    {"QUERY_ENGINE_MODEL_NAME": "engine-only-model"},
    {"QUERY_ENGINE_BASE_URL": "https://engine.example.invalid/v1"},
    {"QUERY_ENGINE_API_KEY": "engine-only-key", "QUERY_ENGINE_BASE_URL": "https://engine.example.invalid/v1"},
])
def test_partial_engine_configuration_never_borrows_generic_provider_fields(
    offline_runner, monkeypatch, engine_values,
):
    write_env(
        offline_runner, **engine_values,
        LLM_MODEL_API_KEY="other-provider-key",
        LLM_MODEL_BASE_URL="https://other-provider.example.invalid/v1",
        LLM_MODEL_NAME="other-provider-model", TAVILY_API_KEY="offline-search-key",
    )
    captured = install_offline_agent(monkeypatch)
    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        run_research(monkeypatch, offline_runner)
    assert not captured["configs"]
    assert not captured["llm"]
    assert not captured["decision"]


@pytest.mark.parametrize("engine_url", [None, ""])
def test_engine_default_endpoint_never_inherits_generic_url(
    offline_runner, monkeypatch, engine_url,
):
    values = {
        "QUERY_ENGINE_API_KEY": "engine-key", "QUERY_ENGINE_MODEL_NAME": "engine-model",
        "LLM_MODEL_API_KEY": "other-provider-key", "LLM_MODEL_NAME": "other-provider-model",
        "LLM_MODEL_BASE_URL": "https://other-provider.example.invalid/v1",
        "TAVILY_API_KEY": "offline-search-key",
        "TYPESAFE_API_KEY": "legacy-decision-key", "OPENROUTER_API_KEY": "legacy-openrouter-key",
    }
    if engine_url is not None:
        values["QUERY_ENGINE_BASE_URL"] = engine_url
    write_env(offline_runner, **values)
    captured = install_offline_agent(monkeypatch)
    metadata, report = run_research(monkeypatch, offline_runner)
    request, = captured["llm"]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer engine-key"
    assert not captured["decision"]
    assert report == "fallback: offline report"
    assert metadata["system_one_requested"] is True
    assert metadata["system_one_available"] is False
    assert metadata["system_one_enabled"] is False
    assert metadata["system_one_requests_attempted"] == 0


def test_process_environment_overrides_dotenv_including_empty_decision_key(
    offline_runner, monkeypatch,
):
    write_env(
        offline_runner,
        QUERY_ENGINE_API_KEY="dotenv-llm-key",
        QUERY_ENGINE_BASE_URL="https://dotenv.example.invalid/v1",
        QUERY_ENGINE_MODEL_NAME="dotenv-model", TAVILY_API_KEY="offline-search-key",
        DECISION_MODEL_API_KEY="dotenv-decision-key", MAX_REFLECTIONS="3",
    )
    os.environ.update(
        QUERY_ENGINE_API_KEY="process-llm-key", QUERY_ENGINE_BASE_URL="",
        QUERY_ENGINE_MODEL_NAME="process-model", DECISION_MODEL_API_KEY="", MAX_REFLECTIONS="0",
    )
    captured = install_offline_agent(monkeypatch)
    metadata, report = run_research(monkeypatch, offline_runner)
    request, = captured["llm"]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer process-llm-key"
    assert metadata["llm_model"] == "process-model"
    assert metadata["max_reflections"] == 0
    assert metadata["system_one_enabled"] is False
    assert not captured["decision"]
    assert report == "fallback: offline report"


def test_disabled_dotenv_cannot_supply_a_stale_url_for_process_credentials(
    offline_runner, monkeypatch,
):
    write_env(
        offline_runner, QUERY_ENGINE_BASE_URL="https://stale-provider.example.invalid/v1",
    )
    os.environ.update(
        PYTHON_DOTENV_DISABLED="1", QUERY_ENGINE_API_KEY="process-only-key",
        QUERY_ENGINE_MODEL_NAME="process-only-model", TAVILY_API_KEY="offline-search-key",
    )
    captured = install_offline_agent(monkeypatch)
    run_research(monkeypatch, offline_runner)
    request, = captured["llm"]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer process-only-key"
    assert captured["configs"][0].QUERY_ENGINE_BASE_URL == ""


@pytest.mark.parametrize("cwd_present", [False, True])
def test_runner_uses_one_dotenv_file_with_cwd_before_project(
    offline_runner, monkeypatch, cwd_present,
):
    project = offline_runner / "project"
    project.mkdir()
    monkeypatch.setattr(
        environment_config, "__file__", str(project / "SystemOne" / "config.py"),
    )
    write_env(
        project, QUERY_ENGINE_API_KEY="project-key",
        QUERY_ENGINE_BASE_URL="https://project.example.invalid/v1",
        QUERY_ENGINE_MODEL_NAME="project-model", TAVILY_API_KEY="project-search-key",
        DECISION_MODEL_API_KEY="project-decision-key", MAX_REFLECTIONS="3",
    )
    if cwd_present:
        write_env(
            offline_runner, QUERY_ENGINE_API_KEY="cwd-key",
            QUERY_ENGINE_BASE_URL="https://cwd.example.invalid/v1",
            QUERY_ENGINE_MODEL_NAME="cwd-model", TAVILY_API_KEY="cwd-search-key",
            MAX_REFLECTIONS="1",
        )
    captured = install_offline_agent(monkeypatch)
    metadata, _ = run_research(monkeypatch, offline_runner)
    selected = "cwd" if cwd_present else "project"
    request, = captured["llm"]
    assert str(request.url) == f"https://{selected}.example.invalid/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {selected}-key"
    assert metadata["llm_model"] == f"{selected}-model"
    assert metadata["max_reflections"] == (1 if cwd_present else 3)
    # A CWD file does not inherit the project's decision key.
    assert metadata["system_one_available"] is (not cwd_present)
    assert len(captured["decision"]) == int(not cwd_present)


def test_metadata_uses_effective_query_settings_instead_of_reparsing_environment(
    offline_runner, monkeypatch,
):
    write_env(
        offline_runner, QUERY_ENGINE_API_KEY="offline-llm-key",
        QUERY_ENGINE_MODEL_NAME="offline-llm-model", TAVILY_API_KEY="offline-search-key",
        # QueryEngine Settings explicitly supports case-insensitive env names.
        max_reflections="3",
    )
    captured = install_offline_agent(monkeypatch)
    metadata, _ = run_research(monkeypatch, offline_runner)
    assert "MAX_REFLECTIONS" not in os.environ
    assert captured["configs"][0].MAX_REFLECTIONS == 3
    assert metadata["max_reflections"] == 3


@pytest.mark.parametrize("outcome,succeeded,failed", [
    ("http_error", 0, 1), ("low_confidence", 1, 0), ("malformed", 1, 0),
])
def test_runtime_counts_expose_failed_attempts_without_claiming_decision_usage(
    offline_runner, monkeypatch, outcome, succeeded, failed,
):
    write_env(
        offline_runner, QUERY_ENGINE_API_KEY="offline-llm-key",
        QUERY_ENGINE_MODEL_NAME="offline-model", TAVILY_API_KEY="offline-search-key",
        DECISION_MODEL_API_KEY="offline-decision-key",
    )
    captured = install_offline_agent(monkeypatch, outcome=outcome)
    # Seed an earlier request to ensure metadata describes this run, not the
    # lifetime of the shared client when the runner is used programmatically.
    client.get_system_one_client().evaluate(state={}, questions={}, decision_id="earlier")
    metadata, report = run_research(monkeypatch, offline_runner)
    assert len(captured["decision"]) == 2
    assert report == "fallback: offline report"
    assert metadata["system_one_requested"] is True
    assert metadata["system_one_available"] is True
    assert metadata["system_one_enabled"] is True
    assert metadata["system_one_requests_attempted"] == 1
    assert metadata["system_one_responses_succeeded"] == succeeded
    assert metadata["system_one_requests_failed"] == failed
