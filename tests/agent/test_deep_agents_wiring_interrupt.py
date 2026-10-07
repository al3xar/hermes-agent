"""Deep-agents facade: interrupt()/clear_interrupt() must not AttributeError.

``AIAgent(runtime="deepagents")`` skips ``init_agent``, but the facade still
inherits ``InterruptControlMixin``; the gateway calls ``agent.interrupt(text)``
on a busy session and the mixin reads control state (``_execution_thread_id``,
``_active_children_lock`` ...) that only ``init_agent`` used to seed.
"""

import sys
import types

import pytest


class _StubImpl:
    """Stand-in for DeepAgentsAIAgent: no LangGraph, no control state."""

    mode = "deepagents"
    quiet_mode = True

    def __init__(self, **kwargs):
        self._callbacks = {}

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


@pytest.fixture
def deepagents_facade(monkeypatch):
    fake = types.ModuleType("agent.deep_agents_runtime")
    fake.DeepAgentsAIAgent = _StubImpl
    monkeypatch.setitem(sys.modules, "agent.deep_agents_runtime", fake)
    from run_agent import AIAgent

    return AIAgent(runtime="deepagents", quiet_mode=True, session_id="s")


def test_interrupt_does_not_raise(deepagents_facade):
    assert deepagents_facade.interrupt("new message") is True
    assert deepagents_facade._interrupt_requested is True
    # No execution thread bound yet: the tool-level signal is deferred.
    assert deepagents_facade._interrupt_thread_signal_pending is True


def test_clear_interrupt_does_not_raise(deepagents_facade):
    deepagents_facade.interrupt("x")
    deepagents_facade.clear_interrupt()
    assert deepagents_facade._interrupt_requested is False
    assert deepagents_facade._interrupt_thread_signal_pending is False


def test_hard_interrupt_does_not_raise(deepagents_facade):
    deepagents_facade.hard_interrupt("stop")
    assert deepagents_facade._hard_interrupt_requested.is_set()
