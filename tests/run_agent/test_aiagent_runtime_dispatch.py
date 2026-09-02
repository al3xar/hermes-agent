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


def test_active_runtime_reflects_instantiated_impl(monkeypatch):
    """``active_runtime`` validates the *live* impl, not the config flag: it
    reports 'deepagents' only when a DeepAgentsAIAgent impl self-reports it."""
    import types

    def fake_init_deepagents(self, **kwargs):
        self._runtime_mode = "deepagents"
        self._deep_agents_impl = types.SimpleNamespace(mode="deepagents")

    monkeypatch.setattr(
        run_agent.AIAgent, "_init_deepagents", fake_init_deepagents, raising=True
    )
    monkeypatch.setattr(
        "agent.agent_init.init_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("native must not run")),
        raising=True,
    )

    agent = run_agent.AIAgent(runtime="deepagents", model="m", skip_memory=True)
    assert agent.active_runtime == "deepagents"


def test_active_runtime_falls_back_to_native_without_live_impl(monkeypatch):
    """A deepagents flag with no real impl (failed/partial init) must NOT
    masquerade as deepagents — there's no impl to claim it."""

    def fake_init_deepagents(self, **kwargs):
        # Flag set, but no DeepAgentsAIAgent self-reporting its mode.
        self._runtime_mode = "deepagents"
        self._deep_agents_impl = object()

    monkeypatch.setattr(
        run_agent.AIAgent, "_init_deepagents", fake_init_deepagents, raising=True
    )

    agent = run_agent.AIAgent(runtime="deepagents", model="m", skip_memory=True)
    assert agent.active_runtime == "native"


def test_active_runtime_native_by_default(monkeypatch):
    monkeypatch.setattr(
        "agent.agent_init.init_agent", lambda *a, **k: None, raising=True
    )
    agent = run_agent.AIAgent(model="m")
    assert agent.active_runtime == "native"


def test_deepagents_branch_forwards_display_callbacks(monkeypatch):
    """Callbacks passed to ``AIAgent(...)`` at construction (as the TUI and CLI
    do) must reach the deepagents impl — otherwise the streaming bridge gets
    ``None`` and the UI shows no tool / thinking chrome (text streams only via
    run_conversation's stream_callback). The gateway sets them post-init so it
    was unaffected; the TUI/CLI construction path was not."""
    import types

    def fake_init_deepagents(self, **kwargs):
        self._runtime_mode = "deepagents"
        # SimpleNamespace records forwarded attrs like the real impl's
        # __setattr__ capture into _callbacks.
        self._deep_agents_impl = types.SimpleNamespace(mode="deepagents")

    monkeypatch.setattr(
        run_agent.AIAgent, "_init_deepagents", fake_init_deepagents, raising=True
    )

    cb_start = lambda *a: None
    cb_complete = lambda *a: None
    cb_gen = lambda *a: None
    cb_think = lambda *a: None
    cb_progress = lambda *a, **k: None

    agent = run_agent.AIAgent(
        runtime="deepagents",
        model="m",
        skip_memory=True,
        tool_start_callback=cb_start,
        tool_complete_callback=cb_complete,
        tool_gen_callback=cb_gen,
        thinking_callback=cb_think,
        tool_progress_callback=cb_progress,
    )

    impl = agent._deep_agents_impl
    assert impl.tool_start_callback is cb_start
    assert impl.tool_complete_callback is cb_complete
    assert impl.tool_gen_callback is cb_gen
    assert impl.thinking_callback is cb_think
    assert impl.tool_progress_callback is cb_progress


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
