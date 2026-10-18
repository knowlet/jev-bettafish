"""Offline credential routing and non-mutating configuration regressions."""

import json
import os

import httpx
import pytest

from SystemOne import client, config as decision_config, decisions


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(
        decision_config, "__file__", str(project / "SystemOne" / "config.py")
    )
    return tmp_path, project


def test_decision_client_does_not_pin_unrelated_engine_credentials(isolated_config):
    """A later provider change must load its new key and URL together."""
    from config import Settings

    cwd, _ = isolated_config
    env_file = cwd / ".env"
    env_file.write_text(
        "QUERY_ENGINE_API_KEY=old-provider-dummy-key\n"
        "QUERY_ENGINE_MODEL_NAME=old-model\n",
        encoding="utf-8",
    )
    before = dict(os.environ)
    configured = client.SystemOneClient()
    assert not configured.enabled
    assert os.environ == before

    env_file.write_text(
        "QUERY_ENGINE_API_KEY=new-provider-dummy-key\n"
        "QUERY_ENGINE_BASE_URL=https://new-provider.invalid/v1\n"
        "QUERY_ENGINE_MODEL_NAME=new-model\n",
        encoding="utf-8",
    )
    reloaded = Settings(_env_file=env_file)
    assert reloaded.QUERY_ENGINE_API_KEY == "new-provider-dummy-key"
    assert reloaded.QUERY_ENGINE_BASE_URL == "https://new-provider.invalid/v1"
    assert reloaded.QUERY_ENGINE_MODEL_NAME == "new-model"


def test_selected_dotenv_is_not_a_per_key_project_fallback(isolated_config):
    cwd, project = isolated_config
    (project / ".env").write_text(
        "DECISION_MODEL_API_KEY=project-dummy-key\n", encoding="utf-8"
    )
    (cwd / ".env").write_text("SYSTEM_ONE_ENABLED=true\n", encoding="utf-8")
    before = dict(os.environ)
    assert not client.SystemOneClient().enabled
    assert os.environ == before


def test_explicit_dotenv_disable_does_not_load_file_credentials(isolated_config):
    cwd, _ = isolated_config
    (cwd / ".env").write_text(
        "DECISION_MODEL_API_KEY=file-dummy-key\n", encoding="utf-8"
    )
    os.environ["PYTHON_DOTENV_DISABLED"] = "true"
    assert not client.SystemOneClient().enabled


def test_process_values_win_inside_dotenv_references(isolated_config):
    cwd, _ = isolated_config
    (cwd / ".env").write_text(
        "SERVICE_KEY=file-dummy-key\n"
        "SERVICE_URL=https://file-provider.invalid/decisions\n"
        "DECISION_MODEL_API_KEY=${SERVICE_KEY}\n"
        "DECISION_MODEL_BASE_URL=${SERVICE_URL}\n",
        encoding="utf-8",
    )
    os.environ.update(
        SERVICE_KEY="process-dummy-key",
        SERVICE_URL="https://process-provider.invalid/decisions",
    )
    before = dict(os.environ)
    configured = client.SystemOneClient()
    assert configured.api_key == "process-dummy-key"
    assert configured.url == "https://process-provider.invalid/decisions"
    assert os.environ == before


@pytest.mark.parametrize("key", ["", "   \t"])
@pytest.mark.parametrize("source", ["explicit", "process"])
def test_blank_key_disables_without_borrowing_credentials(
    isolated_config, monkeypatch, key, source,
):
    cwd, _ = isolated_config
    (cwd / ".env").write_text(
        "DECISION_MODEL_API_KEY=dotenv-dummy-key\n", encoding="utf-8"
    )
    os.environ["DECISION_MODEL_API_KEY"] = (
        "process-dummy-key" if source == "explicit" else key
    )
    configured = client.SystemOneClient(
        api_key=key if source == "explicit" else None,
        url="https://explicit-provider.invalid/decisions",
    )
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"answers": {}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(client.httpx, "post", transport.post)
        assert configured.evaluate(state={}, questions={}, decision_id="blank-key") is None
    assert not configured.enabled
    assert configured.api_key == ""
    assert configured.request_stats == {"attempted": 0, "succeeded": 0, "failed": 0}
    assert seen == []


@pytest.mark.parametrize("canonical_url", [None, "https://custom-provider.invalid/decisions"])
def test_request_uses_only_the_canonical_credential_and_endpoint(
    isolated_config, monkeypatch, canonical_url,
):
    os.environ.update(
        DECISION_MODEL_API_KEY="decision-dummy-key",
        DECISION_MODEL_NAME="selected-model",
        TYPESAFE_API_KEY="typesafe-dummy-key",
        OPENROUTER_API_KEY="legacy-openrouter-dummy-key",
        LLM_MODEL_API_KEY="llm-dummy-key",
        SYSTEM_ONE_URL="https://legacy-provider.invalid/decisions",
        SYSTEM_ONE_MODEL="legacy-model",
    )
    if canonical_url is not None:
        os.environ["DECISION_MODEL_BASE_URL"] = canonical_url
    configured = client.SystemOneClient()
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"answers": {}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(client.httpx, "post", transport.post)
        assert configured.evaluate(
            state={"prompt": "offline"}, questions={}, decision_id="canonical",
        ) == {"answers": {}}
    assert len(seen) == 1
    assert str(seen[0].url) == (
        canonical_url or "https://openrouter.ai/api/alpha/decisions"
    )
    assert seen[0].headers["authorization"] == "Bearer decision-dummy-key"
    assert json.loads(seen[0].content)["model"] == "selected-model"
    assert configured.request_stats == {"attempted": 1, "succeeded": 1, "failed": 0}


def test_only_legacy_and_llm_credentials_make_no_request(isolated_config, monkeypatch):
    os.environ.update(
        TYPESAFE_API_KEY="typesafe-dummy-key",
        OPENROUTER_API_KEY="legacy-openrouter-dummy-key",
        LLM_MODEL_API_KEY="llm-dummy-key",
        QUERY_ENGINE_API_KEY="query-dummy-key",
    )
    configured = client.SystemOneClient()

    def unexpected_request(*args, **kwargs):
        pytest.fail("Other providers' credentials must never enable a decision request")

    monkeypatch.setattr(client.httpx, "post", unexpected_request)
    assert configured.requested
    assert not configured.enabled
    assert configured.evaluate(state={}, questions={}, decision_id="legacy-only") is None


def test_explicit_values_take_precedence_over_process_and_dotenv(isolated_config):
    cwd, _ = isolated_config
    (cwd / ".env").write_text(
        "DECISION_MODEL_API_KEY=file-dummy-key\n"
        "DECISION_MODEL_BASE_URL=https://file-provider.invalid/decisions\n"
        "DECISION_MODEL_NAME=file-model\nSYSTEM_ONE_TIMEOUT=20\n",
        encoding="utf-8",
    )
    os.environ.update(
        DECISION_MODEL_API_KEY="process-dummy-key",
        DECISION_MODEL_BASE_URL="https://process-provider.invalid/decisions",
        DECISION_MODEL_NAME="process-model",
        SYSTEM_ONE_TIMEOUT="10",
    )
    configured = client.SystemOneClient(
        api_key="explicit-dummy-key",
        url="https://explicit-provider.invalid/decisions",
        model="explicit-model",
        timeout=5,
    )
    assert configured.api_key == "explicit-dummy-key"
    assert configured.url == "https://explicit-provider.invalid/decisions"
    assert configured.model == "explicit-model"
    assert configured.timeout == 5


def test_redirect_does_not_forward_credentials_or_research_state(
    isolated_config, monkeypatch,
):
    os.environ.update(DECISION_MODEL_API_KEY="decision-dummy-key")
    configured = client.SystemOneClient()
    seen = []

    def handle(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(307, headers={
                "Location": "https://other-provider.invalid/collect",
            })
        return httpx.Response(200, json={"answers": {}})

    # Even a transport whose default follows redirects must honor the call's
    # explicit prohibition. This exercises HTTPX redirect handling itself.
    with httpx.Client(
        transport=httpx.MockTransport(handle), follow_redirects=True,
    ) as transport:
        monkeypatch.setattr(client.httpx, "post", transport.post)
        assert configured.evaluate(
            state={"private_research": "offline-only"},
            questions={}, decision_id="redirect",
        ) is None
    assert len(seen) == 1
    assert seen[0].url.host == "openrouter.ai"
    assert configured.request_stats == {"attempted": 1, "succeeded": 0, "failed": 1}


def test_dotenv_toggles_thresholds_and_trace_remain_effective(
    isolated_config, monkeypatch,
):
    cwd, _ = isolated_config
    trace_path = cwd / "trace.jsonl"
    (cwd / ".env").write_text(
        "DECISION_MODEL_API_KEY=decision-dummy-key\n"
        "SYSTEM_ONE_ENABLED=true\n"
        "SYSTEM_ONE_CHOICE_CONFIDENCE=0.9\n"
        "SYSTEM_ONE_EVIDENCE_CANDIDATES=4\n"
        f"SYSTEM_ONE_TRACE_PATH={trace_path}\n",
        encoding="utf-8",
    )
    before = dict(os.environ)
    configured = client.SystemOneClient()
    monkeypatch.setattr(decisions, "get_system_one_client", lambda: configured)

    def handle(request):
        return httpx.Response(200, json={
            "model": "served-model",
            "provider": "served-provider",
            "answers": {"search_tool": {
                "choice": "deep_search_news", "confidence": 0.8,
            }},
        })

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(client.httpx, "post", transport.post)
        choice = decisions.choose_query_search_tool(
            input_data={"title": "offline"}, generated_query="offline", phase="initial",
        )
    assert choice is None  # The .env confidence threshold is actually applied.
    assert decisions._int_env("SYSTEM_ONE_EVIDENCE_CANDIDATES", 10) == 4
    assert configured.requested and configured.enabled
    assert os.environ == before
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["response_model"] == "served-model"
    assert record["provider"] == "served-provider"
    assert "state_preview" not in record


@pytest.mark.parametrize("failure", ["transport", "timeout", "http", "429", "json", "not_object"])
def test_failure_counters_and_logs_do_not_expose_credentials(
    isolated_config, monkeypatch, failure,
):
    cwd, _ = isolated_config
    trace_path = cwd / "trace.jsonl"
    os.environ.update(
        DECISION_MODEL_API_KEY="decision-dummy-key",
        SYSTEM_ONE_TRACE_PATH=str(trace_path),
    )
    configured = client.SystemOneClient(
        url="https://provider.invalid/decisions?access_token=url-dummy-secret",
    )
    warnings = []
    monkeypatch.setattr(client.logger, "warning", warnings.append)

    def handle(request):
        if failure == "transport":
            raise httpx.RequestError("invalid Bearer decision-dummy-key", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout for decision-dummy-key", request=request)
        if failure == "http":
            return httpx.Response(503)
        if failure == "429":
            return httpx.Response(429)
        if failure == "json":
            return httpx.Response(200, text="invalid decision-dummy-key")
        return httpx.Response(200, json=["decision-dummy-key"])

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(client.httpx, "post", transport.post)
        assert configured.evaluate(state={}, questions={}, decision_id="failure") is None
    assert configured.request_stats == {"attempted": 1, "succeeded": 0, "failed": 1}
    exposed = trace_path.read_text(encoding="utf-8") + "\n".join(warnings)
    assert "decision-dummy-key" not in exposed
    assert "url-dummy-secret" not in exposed
    assert json.loads(trace_path.read_text(encoding="utf-8"))["ok"] is False
