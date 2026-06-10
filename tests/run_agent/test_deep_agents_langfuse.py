"""Unit tests for Langfuse self-hosted tracing wiring in deep_agents_runtime.

Verifies that ``DeepAgentsAIAgent`` injects a Langfuse ``CallbackHandler`` into
the LangGraph invocation config when (and only when) Langfuse keys are present,
without breaking the existing LangSmith / streaming paths.

All tests build the agent via ``object.__new__`` (bypassing ``__init__`` and the
real SDK) and mock ``_agent.invoke`` so no live model or Langfuse server is hit.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_langfuse_env(monkeypatch):
    """Keep tests hermetic: the handler mirrors keys into os.environ as a
    side effect, and LANGFUSE_* names are not credential-shaped so the
    global conftest filter doesn't clear them between tests in this file."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)


def _make_agent(callbacks=None, langfuse_handler=None, **overrides):
    """Build a DeepAgentsAIAgent in the sync (non-streaming) path.

    ``callbacks`` populates the captured-attribute dict the gateway would set
    (e.g. langfuse_public_key). ``langfuse_handler`` pre-seeds the memoized
    handler so we can assert injection without constructing a real one.
    """
    from agent.deep_agents_runtime import DeepAgentsAIAgent

    defaults = dict(
        mode="deepagents",
        _quiet_mode=False,
        _skip_memory=True,
        _platform=None,
        _session_id="test-session",
        _max_iterations=90,
        provider="",
        _api_key=None,
        _base_url=None,
        _callbacks=callbacks or {},
        _agent=MagicMock(),
        _langfuse_handler=langfuse_handler,
    )
    defaults.update(overrides)
    agent = object.__new__(DeepAgentsAIAgent)
    for k, v in defaults.items():
        object.__setattr__(agent, k, v)
    return agent


def _invoke_and_get_config(agent):
    """Run a sync conversation and return the config passed to invoke."""
    agent._agent.invoke.return_value = {"messages": []}
    with patch("agent.deep_agents_runtime._HadesStreamingBridge") as mock_br:
        mock_br.return_value.any_callbacks_set.return_value = False
        agent.run_conversation(user_message="hi")
    return agent._agent.invoke.call_args.kwargs["config"]


class TestLangfuseInjection:
    def test_handler_injected_when_keys_present(self):
        """With Langfuse keys, a CallbackHandler is added to config['callbacks']."""
        fake_handler = object()
        agent = _make_agent(
            callbacks={
                "langfuse_public_key": "pk-lf-test",
                "langfuse_secret_key": "sk-lf-test",
            },
        )
        with (
            patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", True),
            patch(
                "agent.deep_agents_runtime.CallbackHandler",
                return_value=fake_handler,
            ),
        ):
            config = _invoke_and_get_config(agent)

        assert fake_handler in config.get("callbacks", [])

    def test_no_handler_without_keys(self):
        """Without keys, no callbacks are injected (config stays clean)."""
        agent = _make_agent(callbacks={})
        with patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", True):
            config = _invoke_and_get_config(agent)

        assert not config.get("callbacks")

    def test_no_handler_when_package_unavailable(self):
        """Keys present but langfuse not installed -> no crash, no handler."""
        agent = _make_agent(
            callbacks={
                "langfuse_public_key": "pk-lf-test",
                "langfuse_secret_key": "sk-lf-test",
            },
        )
        with patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", False):
            config = _invoke_and_get_config(agent)

        assert not config.get("callbacks")

    def test_handler_construction_failure_is_swallowed(self):
        """If CallbackHandler() raises, the conversation still runs."""
        agent = _make_agent(
            callbacks={
                "langfuse_public_key": "pk-lf-test",
                "langfuse_secret_key": "sk-lf-test",
            },
        )
        with (
            patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", True),
            patch(
                "agent.deep_agents_runtime.CallbackHandler",
                side_effect=RuntimeError("boom"),
            ),
        ):
            config = _invoke_and_get_config(agent)

        assert not config.get("callbacks")
        agent._agent.invoke.assert_called_once()

    def test_handler_is_memoized(self):
        """The handler is constructed once and reused across calls."""
        agent = _make_agent(
            callbacks={
                "langfuse_public_key": "pk-lf-test",
                "langfuse_secret_key": "sk-lf-test",
            },
        )
        with (
            patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", True),
            patch(
                "agent.deep_agents_runtime.CallbackHandler",
                return_value=object(),
            ) as mock_ctor,
        ):
            agent._get_langfuse_handler()
            agent._get_langfuse_handler()

        assert mock_ctor.call_count == 1

    def test_langsmith_path_unaffected_without_langfuse(self):
        """Existing LangSmith tags path still works when no Langfuse keys set."""
        agent = _make_agent(callbacks={}, _ls_api_key="ls-key", _ls_tags=["hades"])
        with patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", True):
            config = _invoke_and_get_config(agent)

        assert config["tags"] == ["hades"]


class TestLangfuseProperties:
    def test_has_langfuse_tracing_true_when_handler_present(self):
        agent = _make_agent(langfuse_handler=object())
        assert agent.has_langfuse_tracing is True

    def test_has_langfuse_tracing_false_when_absent(self):
        agent = _make_agent(langfuse_handler=None, callbacks={})
        with patch("agent.deep_agents_runtime.LANGFUSE_AVAILABLE", True):
            assert agent.has_langfuse_tracing is False

    def test_get_tracing_config_reports_langfuse(self):
        agent = _make_agent(
            langfuse_handler=object(),
            _langgraph_checkpointer=True,
            _langgraph_store=False,
            _debug=False,
            _ls_project="hades",
            _ls_tags=["hades"],
            _ls_api_key=None,
        )
        cfg = agent.get_tracing_config()
        assert cfg["langfuse_enabled"] is True
