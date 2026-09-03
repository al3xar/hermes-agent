"""Deep-agents wiring — Strategy/Factory extraction from run_agent.py.

This module holds the *live* deep-agents integration so that ``run_agent.py``
keeps only stable SEAMS (the ``runtime`` constructor param, the ``_runtime_mode``
attribute and the ``if runtime == "deepagents":`` branch in ``__init__``/
``run_conversation``).  When upstream churns ``run_agent.py`` the seam contract
stays small; the behaviour here can re-synchronise without re-embedding the
wiring into the core file.

Contract preserved (do NOT change):
  * AIAgent(**_agent_cbs(sid), **_runtime_kwargs) construction from webui/tui.
  * run_conversation(...) native contract and the native callback surface
    (tool_progress/start/complete, thinking, reasoning, clarify, stream_delta,
    interim_assistant, tool_gen, status, notice, setup_mcp ...).
  * The ``_StreamBridge`` in ``agent.deep_agents_runtime`` translates LangGraph
    events to that SAME native callback signature.

The forwarding __setattr__/__getattr__ of ``_CAPTURED_NAMES`` lives in this mixin
instead of being inlined into ``AIAgent.__init__`` / ``__setattr__`` / ``__getattr__``.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Callback / config names that the deep-agents impl captures.  Kept in the
# separate file so the core only imports the symbol, never inlines the set.
# ---------------------------------------------------------------------------
_CAPTURED_NAMES = frozenset((
    "tool_progress_callback",
    "tool_start_callback",
    "tool_complete_callback",
    "thinking_callback",
    "reasoning_callback",
    "clarify_callback",
    "step_callback",
    "stream_delta_callback",
    "interim_assistant_callback",
    "tool_gen_callback",
    "status_callback",
    "notice_callback",
    "notice_clear_callback",
    "reasoning_config",
    "service_tier",
    "request_overrides",
    "background_review_callback",
    # Tracing / observability
    "debug",
    "langsmith_api_key",
    "langsmith_project",
    "langsmith_tags",
    # Langfuse credentials — set by the gateway (or seeded from
    # HERMES_LANGFUSE_* env at init) and read by _get_langfuse_handler.
    "langfuse_public_key",
    "langfuse_secret_key",
    "langfuse_base_url",
))


def _forward_callbacks_to_impl(agent: Any, impl: Any) -> None:
    """Forward construction-time display callbacks / request config to the
    deep-agents impl.

    The native path routes these through ``init_agent``; here they would
    otherwise be dropped, so the streaming bridge gets ``None`` and the UI
    shows no tool / thinking / status chrome (only text streams, via
    run_conversation's stream_callback). The TUI and CLI pass these as
    constructor kwargs; the gateway sets them post-init, so only TUI/CLI were
    affected. ``__setattr__`` captures _CAPTURED_NAMES into the impl.
    """
    forwarded = (
        ("tool_progress_callback", agent._tool_progress_callback),
        ("tool_start_callback", agent._tool_start_callback),
        ("tool_complete_callback", agent._tool_complete_callback),
        ("thinking_callback", agent._thinking_callback),
        ("reasoning_callback", agent._reasoning_callback),
        ("clarify_callback", agent._clarify_callback),
        ("step_callback", agent._step_callback),
        ("stream_delta_callback", agent._stream_delta_callback),
        ("interim_assistant_callback", agent._interim_assistant_callback),
        ("tool_gen_callback", agent._tool_gen_callback),
        ("status_callback", agent._status_callback),
        ("notice_callback", agent._notice_callback),
        ("notice_clear_callback", agent._notice_clear_callback),
        ("reasoning_config", agent._reasoning_config),
        ("service_tier", agent._service_tier),
        ("request_overrides", agent._request_overrides),
    )
    for name, val in forwarded:
        if val is not None:
            agent.__setattr__(name, val)


def _deepagents_init(
    agent: Any,
    *,
    base_url: Any = None,
    api_key: Any = None,
    provider: Any = None,
    model: Any = "",
    max_iterations: Any = 90,
    enabled_toolsets: Any = None,
    disabled_toolsets: Any = None,
    quiet_mode: bool = False,
    skip_memory: bool = False,
    skip_context_files: bool = False,
    session_id: Any = None,
    platform: Any = None,
    # Gateway / TUI / CLI extras:
    reasoning_config: Any = None,
    service_tier: Any = None,
    request_overrides: Any = None,
    ephemeral_system_prompt: Any = None,
    credential_pool: Any = None,
) -> None:
    """Initialize ``agent`` as a DeepAgents-backed runtime.

    Only the base config plus the reasoning/config kwargs that the committed
    ``DeepAgentsAIAgent`` accepts are forwarded; the impl captures every other
    callback via ``__setattr__`` after construction.
    """
    from agent.deep_agents_runtime import DeepAgentsAIAgent

    agent._runtime_mode = "deepagents"
    agent._deep_agents_impl = DeepAgentsAIAgent(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        model=model,
        max_iterations=max_iterations,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=quiet_mode,
        skip_memory=skip_memory,
        skip_context_files=skip_context_files,
        session_id=session_id,
        platform=platform,
        reasoning_config=reasoning_config,
        service_tier=service_tier,
        request_overrides=request_overrides,
        ephemeral_system_prompt=ephemeral_system_prompt,
        credential_pool=credential_pool,
    )
    # Expose gateway-facing attrs via the impl's __setattr__ forwarders.
    agent.__setattr__("reasoning_config", reasoning_config)
    agent.__setattr__("service_tier", service_tier)
    agent.__setattr__("request_overrides", request_overrides)


def build_agent_impl(agent: Any) -> Any:
    """Strategy/Factory seam.

    Given a half-constructed ``AIAgent`` facade whose constructor kwargs have
    been stashed as ``_da_*`` private attrs, build the live runtime impl and
    return it.  Returns ``None`` for the native runtime (the facade then falls
    through to ``init_agent``).
    """
    if getattr(agent, "_runtime_mode", None) != "deepagents":
        return None
    _deepagents_init(
        agent,
        base_url=getattr(agent, "_da_base_url", None),
        api_key=getattr(agent, "_da_api_key", None),
        provider=getattr(agent, "_da_provider", None),
        model=getattr(agent, "_da_model", ""),
        max_iterations=getattr(agent, "_da_max_iterations", 90),
        enabled_toolsets=getattr(agent, "_da_enabled_toolsets", None),
        disabled_toolsets=getattr(agent, "_da_disabled_toolsets", None),
        quiet_mode=getattr(agent, "_da_quiet_mode", False),
        skip_memory=getattr(agent, "_da_skip_memory", False),
        skip_context_files=getattr(agent, "_da_skip_context_files", False),
        session_id=getattr(agent, "_da_session_id", None),
        platform=getattr(agent, "_da_platform", None),
        reasoning_config=getattr(agent, "_da_reasoning_config", None),
        service_tier=getattr(agent, "_da_service_tier", None),
        request_overrides=getattr(agent, "_da_request_overrides", None),
        ephemeral_system_prompt=getattr(agent, "_da_ephemeral_system_prompt", None),
        credential_pool=getattr(agent, "_da_credential_pool", None),
    )
    return agent._deep_agents_impl


def _is_deepagents(agent: Any) -> bool:
    try:
        mode = object.__getattribute__(agent, "_runtime_mode")
    except AttributeError:
        return False
    return mode == "deepagents"


class DeepAgentsWiringMixin:
    """Mixin that adds deep-agents callback forwarding to ``AIAgent``.

    The forwarding __setattr__/__getattr__ of ``_CAPTURED_NAMES`` used to live
    inline in ``AIAgent.__init__`` / ``__setattr__`` / ``__getattr__``.  Moving
    them here keeps ``run_agent.py`` to a small, stable seam while the live
    behaviour stays in this file (which can re-synchronise without touching the
    core).
    """

    # ------------------------------------------------------------------
    # Callback forwarding for deepagents runtime mode: the gateway sets
    # ``agent.tool_progress_callback = cb`` etc. These must reach
    # self._deep_agents_impl (which has __setattr__ forwarding of its own).
    # ------------------------------------------------------------------
    def _da_setattr(self, name: str, value: Any) -> None:
        if name in (
            "_deep_agents_impl",
            "_runtime_mode",
        ):
            # These are set during construction — allow normally.
            object.__setattr__(self, name, value)
            return
        if not _is_deepagents(self):
            object.__setattr__(self, name, value)
            return
        impl = getattr(self, "_deep_agents_impl", None)
        if impl is not None and name in _CAPTURED_NAMES:
            impl.__setattr__(name, value)
            return
        object.__setattr__(self, name, value)

    def _da_getattr(self, name: str) -> Any:
        if not _is_deepagents(self):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        impl = getattr(self, "_deep_agents_impl", None)
        if impl is not None:
            try:
                return getattr(impl, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def _da_active_runtime(self) -> str:
        """Return the execution backend that was *actually instantiated*.

        Unlike the ``deepagents_mode`` config flag (which only states intent),
        this reflects the live object graph: it returns ``"deepagents`` only
        when a ``DeepAgentsAIAgent`` impl was built and self-reports its mode,
        and ``"native`` otherwise.
        """
        if not _is_deepagents(self):
            return "native"
        impl = getattr(self, "_deep_agents_impl", None)
        if impl is not None and getattr(impl, "mode", None) == "deepagents":
            return "deepagents"
        return "native"


# ---------------------------------------------------------------------------
# Module-level seams.
#
# ``AIAgent`` (in ``run_agent.py``) cannot inherit from ``DeepAgentsWiringMixin``
# without disturbing its MRO / other tests, so it keeps its dunder methods and
# the ``active_runtime`` property as thin, stable seams that delegate here.
# The live behaviour (what these do) lives in this module, which can
# re-synchronise against an upstream change to ``run_agent.py`` without touching
# the core. The first arg is ``Any`` because the seam is invoked with the
# ``AIAgent`` facade, not a ``DeepAgentsWiringMixin``.
# ---------------------------------------------------------------------------


def apply_setattr(self: Any, name: str, value: Any) -> None:
    return DeepAgentsWiringMixin._da_setattr(self, name, value)


def apply_getattr(self: Any, name: str) -> Any:
    return DeepAgentsWiringMixin._da_getattr(self, name)


def resolve_active_runtime(self: Any) -> str:
    return DeepAgentsWiringMixin._da_active_runtime(self)
