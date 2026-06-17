"""Regression: ``AIAgent(runtime=...)`` must accept the selector and dispatch.

The gateway (and ACP / TUI gateway) splat a ``runtime`` key into
``AIAgent(**...)`` when deep-agents mode is enabled.  A refactor that
extracted ``__init__`` into ``agent.agent_init`` dropped the ``runtime``
parameter and the ``if runtime == "deepagents": self._init_deepagents(...)``
dispatch, leaving ``_init_deepagents`` and the gateway call sites orphaned.
The result was ``TypeError: AIAgent.__init__() got an unexpected keyword
argument 'runtime'`` on every gateway/dashboard session with deep-agents on.

These tests pin the contract documented in ``resolve_agent_runtime``:
``runtime="native"`` (default) -> normal init; ``runtime="deepagents"`` ->
``_init_deepagents``.
"""

import inspect

import run_agent


def test_aiagent_init_accepts_runtime_kwarg():
    """The ``runtime`` selector must be a real ``__init__`` parameter."""
    params = inspect.signature(run_agent.AIAgent.__init__).parameters
    assert "runtime" in params
    assert params["runtime"].default == "native"


def test_runtime_deepagents_dispatches_to_init_deepagents(monkeypatch):
    captured = {}

    def fake_init_deepagents(self, **kwargs):
        captured["kwargs"] = kwargs
        self._runtime_mode = "deepagents"
        self._deep_agents_impl = object()

    def fail_native(*args, **kwargs):
        raise AssertionError("native init_agent must not run for deepagents runtime")

    monkeypatch.setattr(
        run_agent.AIAgent, "_init_deepagents", fake_init_deepagents, raising=True
    )
    monkeypatch.setattr("agent.agent_init.init_agent", fail_native, raising=True)

    agent = run_agent.AIAgent(
        runtime="deepagents",
        model="test/model",
        skip_memory=True,
        skip_context_files=True,
        quiet_mode=True,
    )

    assert captured, "expected _init_deepagents to be called"
    assert captured["kwargs"].get("model") == "test/model"
    assert agent._runtime_mode == "deepagents"


def test_runtime_native_uses_standard_init(monkeypatch):
    seen = {}

    def fake_init_agent(self, **kwargs):
        seen["model"] = kwargs.get("model")

    def fail_deep(self, **kwargs):
        raise AssertionError("_init_deepagents must not run for native runtime")

    monkeypatch.setattr("agent.agent_init.init_agent", fake_init_agent, raising=True)
    monkeypatch.setattr(
        run_agent.AIAgent, "_init_deepagents", fail_deep, raising=True
    )

    run_agent.AIAgent(model="test/model")

    assert seen.get("model") == "test/model"
