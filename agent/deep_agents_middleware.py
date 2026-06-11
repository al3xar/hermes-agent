"""Middleware that bridges DeepAgents/LangGraph to Hades turn lifecycle.

Provides:
- _HadesMiddleware: a LangGraph AgentMiddleware that syncs Hades memory
  and tracks interruptions across turns.
- _HadesInterruptSignal: a lightweight interrupt signal that can be
  set/cleared from the gateway/CLI.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from deepagents.graph import AgentMiddleware, AgentState
from langchain_core.messages import messages_from_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _HadesInterruptSignal:
    """Thread-safe interrupt signal shared between gateway and agent."""

    _flag: bool = field(default=False, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def is_set(self) -> bool:
        with self._lock:
            return self._flag

    def set(self):
        with self._lock:
            self._flag = True

    def clear(self):
        with self._lock:
            self._flag = False


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class _HadesMiddleware(AgentMiddleware):
    """LangGraph middleware that hooks into turn lifecycle.

    - ``before_agent``: preresolve memory context for this turn and set
      the interrupt flag if one was requested by the gateway.
    - ``after_agent``: save new conversation context to memory and clear
      the interrupt signal.
    """

    def __init__(self, hades_agent):
        """Store reference to the native AIAgent so we can read/write memory."""
        self._agent = hades_agent
        self._interrupt = _HadesInterruptSignal()

    def before_agent(self, state: dict, **kwargs) -> dict:
        """Run before the model is called.

        - Prefetch memory context (Hades memory_manager).
        - Propagate any interrupt that was requested from the gateway.
        """
        agent = self._agent
        # Prefetch memory context
        if not getattr(agent, "skip_memory", True):
            try:
                mem_ctx = agent.get_memory_context()
                if mem_ctx:
                    state["memory"] = state.get("memory", {})
                    state["memory"]["context"] = mem_ctx
            except Exception:
                logger.exception("Memory prefetch failed in middleware")

        # Check interrupt signal
        if self._interrupt.is_set():
            logger.info("Interrupt signal present at middleware before_agent")

        return state

    def after_agent(self, state: dict, **kwargs) -> dict:
        """Run after the model is called.

        - Save new context to memory.
        - Clear the interrupt signal.
        """
        agent = self._agent
        if not getattr(agent, "skip_memory", True):
            try:
                msgs = state.get("_messages", [])
                if msgs:
                    agent.save_memory(msgs)
            except Exception:
                logger.exception("Memory save failed in middleware")

        self._interrupt.clear()
        return state

    # ── Interrupt API ────────────────────────────────────────────────

    def request_interrupt(self):
        """Signal an interrupt (callable from gateway/CLI)."""
        self._interrupt.set()

    def clear_interrupt(self):
        """Clear the interrupt after handling."""
        self._interrupt.clear()

    @property
    def interrupt_signal(self) -> _HadesInterruptSignal:
        return self._interrupt
