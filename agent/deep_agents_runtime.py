"""DeepAgents runtime - LangGraph-backed agent backed by Hermes internals.

Removes all child AIAgent instantiation. Config is accepted directly from
the AIAgent facade parameters instead of delegating to a full runtime.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from typing import Any, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import (
        HumanMessage,
        AIMessage,
        AIMessageChunk,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    from deepagents.graph import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.backends import langsmith

    DEEPAGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning("DeepAgents SDK not available: %s", e)
    DEEPAGENTS_AVAILABLE = False
    FilesystemMiddleware = None  # type: ignore[misc,assignment]
    langsmith = None  # type: ignore[misc,assignment]
    MemorySaver = None  # type: ignore[misc,assignment]
    InMemoryStore = None  # type: ignore[misc,assignment]

# Langfuse (SDK v3) — optional, lazy-installed via the ``deepagents`` extra.
# In v3 the LangChain ``CallbackHandler`` carries no credentials: they live on
# a ``Langfuse`` client (singleton), and the handler reads from it. Both are
# imported at module level (not inside the method) so tests can patch
# ``Langfuse`` / ``CallbackHandler`` / ``LANGFUSE_AVAILABLE`` and so handler
# construction is gated on a single, inspectable flag. The langchain
# integration also needs the ``langchain`` package present, not just
# ``langchain-core``.
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None  # type: ignore[misc,assignment]
    CallbackHandler = None  # type: ignore[misc,assignment]
    LANGFUSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LANGCHAIN_FINAL_RESPONSE_KEY = "__hermes_final_response"


# ---------------------------------------------------------------------------
# Message Converters
# ---------------------------------------------------------------------------


def _convert_messages_to_langchain(messages):
    """Convert Hermes message list to LangChain messages."""
    if not messages:
        return []

    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            ai = AIMessage(content=content or "")
            if tool_calls:
                ai.tool_calls = tool_calls
            result.append(ai)
        elif role == "tool":
            tool_id = msg.get("tool_call_id", "")
            result.append(ToolMessage(content=content, tool_call_id=tool_id))
    return result


def _convert_langchain_to_hermes(messages):
    """Convert LangChain messages to Hermes message format."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry = {"role": "assistant", "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
    return result


# ---------------------------------------------------------------------------
# Result Parsing
# ---------------------------------------------------------------------------


def _parse_langgraph_result(result: dict, task_id: str = None) -> dict:
    """Parse LangGraph agent result dict into Hermes result dict shape."""
    raw_messages = result.get("messages", [])
    hermes_messages = _convert_langchain_to_hermes(raw_messages)

    final_response = ""
    last_reasoning = None
    for msg in reversed(raw_messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        final_response = part.get("text", "")
                        break
                    elif isinstance(part, str):
                        final_response = part
                        break
            elif msg.content:
                final_response = msg.content
            last_reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "additional_kwargs", {}
            ).get("reasoning")
            break

    # Reasoning the model leaked into the content channel: keep the stored body
    # clean but preserve the thought as ``last_reasoning`` (mirrors the live
    # stream router, which sends it to the thinking channel).
    if isinstance(final_response, str):
        final_response, _leaked_reasoning = _split_reasoning_from_content(final_response)
        if _leaked_reasoning:
            last_reasoning = (
                f"{last_reasoning}\n{_leaked_reasoning}"
                if last_reasoning
                else _leaked_reasoning
            )

    api_calls = sum(
        1 for m in raw_messages if isinstance(m, AIMessage) and m.tool_calls
    )
    completed = bool(final_response)

    return {
        "final_response": final_response,
        "messages": hermes_messages,
        "api_calls": api_calls,
        "completed": completed,
        "failed": False,
        "interrupted": False,
        "partial": False,
        "turn_exit_reason": "completed" if completed else "no_response",
        "last_reasoning": last_reasoning,
        "model": "",
    }


def _parse_error_result(e: Exception) -> dict:
    """Return an error result dict matching run_conversation shape."""
    return {
        "final_response": f"Error: {e}",
        "messages": [],
        "api_calls": 0,
        "completed": False,
        "failed": True,
        "interrupted": False,
        "partial": False,
        "turn_exit_reason": f"deepagents_error: {e}",
    }


# ---------------------------------------------------------------------------
# Environment Injection
# ---------------------------------------------------------------------------


def _inject_provider_env(provider, base_url, api_key):
    """Set provider-specific env vars for LangChain auto-loading.

    LangChain model bindings read API keys from env vars based on provider.
    This maps Hermes' model/provider to the env var names LangChain expects.
    """
    env_map = {
        "": ("OPENAI_API_KEY", "OPENAI_API_BASE"),  # fallback to OpenAI
        "openai": ("OPENAI_API_KEY", "OPENAI_API_BASE"),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_BASE"),
        "google": ("GOOGLE_API_KEY", "GOOGLE_API_BASE"),
        "xai": ("XAI_API_KEY", "XAI_API_BASE"),
        "cohere": ("COHERE_API_KEY", "COHERE_API_BASE"),
        "groq": ("GROQ_API_KEY", "GROQ_API_BASE"),
    }

    env_vars = env_map.get(provider, ("API_KEY", "BASE_URL"))

    if api_key:
        os.environ[env_vars[0]] = api_key

    if base_url:
        os.environ[env_vars[1]] = base_url


# ---------------------------------------------------------------------------
# Reasoning / extended-thinking
# ---------------------------------------------------------------------------

# Hermes effort → token budget, mirroring agent.anthropic_adapter.THINKING_BUDGET
# so the deepagents runtime reasons with the same depth as native for a given
# effort. ``medium`` is the default when an effort isn't recognized.
_REASONING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}


def _reasoning_enabled(reasoning_config) -> bool:
    """True when reasoning_config asks for extended thinking.

    Shape mirrors the native runtime: ``{"enabled": bool, "effort": str}``.
    Absent/``None`` config means "leave the model at its default" (treated as
    not explicitly enabled here).
    """
    if not isinstance(reasoning_config, dict):
        return False
    return reasoning_config.get("enabled") is not False


def _reasoning_effort(reasoning_config) -> str:
    effort = ""
    if isinstance(reasoning_config, dict):
        effort = str(reasoning_config.get("effort", "") or "").strip().lower()
    return effort or "medium"


def _looks_like_openai_reasoning_model(name: str) -> bool:
    """OpenAI exposes ``reasoning_effort`` only on its reasoning families."""
    n = (name or "").lower()
    return n.startswith(("o1", "o3", "o4", "gpt-5")) or "-o1" in n or "-o3" in n


def _build_reasoning_model(provider, model_name, reasoning_config):
    """Return a provider chat-model with extended thinking enabled, or ``None``.

    The deepagents runtime otherwise hands ``create_deep_agent`` a bare model
    string, so ``reasoning_config`` (captured from the gateway but never applied)
    was silently dropped — the model never enabled thinking, so the streaming
    bridge's reasoning path had nothing to surface. This maps the same
    effort→budget the native Anthropic adapter uses onto each provider's
    LangChain binding.

    Best-effort: returns ``None`` (caller keeps the plain string/default model)
    when reasoning is disabled, the provider isn't reasoning-capable here, or the
    binding can't be constructed — enabling thinking must never brick the agent.
    """
    if not _reasoning_enabled(reasoning_config):
        return None

    p = (provider or "").lower()
    name = model_name or ""
    effort = _reasoning_effort(reasoning_config)
    budget = _REASONING_BUDGET.get(effort, 8000)

    try:
        if "anthropic" in p:
            if "haiku" in name.lower():  # Haiku has no extended thinking
                return None
            from langchain_anthropic import ChatAnthropic

            # Anthropic requires max_tokens > budget_tokens when thinking is on.
            return ChatAnthropic(
                model=name,
                thinking={"type": "enabled", "budget_tokens": budget},
                max_tokens=budget + 8192,
            )
        if "google" in p:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # include_thoughts surfaces reasoning so the bridge can stream it.
            return ChatGoogleGenerativeAI(
                model=name,
                thinking_budget=budget,
                include_thoughts=True,
            )
        if p in ("openai", "") and _looks_like_openai_reasoning_model(name):
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=name, reasoning_effort=effort)
    except Exception:
        logger.warning(
            "deepagents: could not enable reasoning for provider=%s model=%s; "
            "falling back to default model binding",
            p,
            name,
            exc_info=True,
        )
    return None


# ---------------------------------------------------------------------------
# Tool Adapters
# ---------------------------------------------------------------------------


def _structured_tool_from_schema(name, description, parameters):
    """Build a LangChain ``StructuredTool`` that routes to Hermes dispatch.

    The tool's parameter schema (OpenAI/JSON-schema object) MUST be forwarded
    as ``args_schema``: the closure below takes only ``**kwargs`` and exposes
    no typed signature, so without an explicit ``args_schema`` LangChain
    infers an *empty* argument schema and the model is handed a tool it can't
    pass arguments to (e.g. ``run_terminal_command`` with no ``command``
    field). langchain 1.x accepts a JSON-schema dict directly as
    ``args_schema``.

    Execution always goes through ``model_tools.handle_function_call``, which
    is the native runtime's dispatch entrypoint — it covers registry tools,
    MCP tools, and the synthetic ``tool_search`` / ``tool_describe`` /
    ``tool_call`` bridge tools alike.
    """
    from model_tools import handle_function_call

    def _execute_sync(**kwargs):
        try:
            return handle_function_call(function_name=name, function_args=kwargs)
        except Exception as e:
            error_str = str(e)
            return json.dumps({"error": error_str}, ensure_ascii=False)

    if not isinstance(parameters, dict) or not parameters:
        parameters = {"type": "object", "properties": {}}

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_execute_sync,
        args_schema=parameters,
        infer_schema=False,
    )


class _HermesToolAdapter:
    """Adapts a single Hermes registry entry to a LangChain StructuredTool.

    Invokes handle_function_call from model_tools atomically.
    """

    def __init__(self, tool_entry):
        self._entry = tool_entry
        self._tool = self._build_langchain_tool()

    def _build_langchain_tool(self):
        entry = self._entry
        schema = entry.schema
        func_name = schema.get("name", entry.name)
        func_desc = entry.description or schema.get("description", "")
        return _structured_tool_from_schema(
            func_name, func_desc, schema.get("parameters")
        )

    @property
    def langchain_tool(self):
        return self._tool

    @property
    def name(self):
        return self._entry.name

    @property
    def schema(self):
        return self._entry.schema

    @property
    def toolset(self):
        return self._entry.toolset


def build_hermes_tools(enabled_toolsets, disabled_toolsets):
    """Build LangChain StructuredTool list from Hermes tool definitions.

    Toolset resolution goes through ``model_tools.get_tool_definitions`` —
    the *same* path the native runtime uses — so that toolset names
    (``terminal``, ``web``, …), composite/legacy toolsets, and MCP server
    aliases (e.g. ``nyxstrike`` → the dynamically registered ``mcp-nyxstrike``
    toolset) all expand to concrete tool definitions. Calling ``registry.
    get_definitions`` directly would bypass that expansion (its argument is a
    pre-resolved *set of tool names*, not toolset names), which silently
    dropped every tool — built-in and MCP alike.

    Tools are built straight from the returned definitions rather than via a
    ``registry.get_entry`` lookup: that keeps full parity with the native
    runtime, including dynamic schema overrides (already applied by
    ``get_tool_definitions``) and the synthetic ``tool_search`` /
    ``tool_describe`` / ``tool_call`` bridge tools — which have no registry
    entry and would otherwise be dropped exactly when a large catalog (e.g.
    NyxStrike's 185+ tools) triggers tool-search assembly.
    """
    from model_tools import get_tool_definitions

    tools = []
    try:
        definitions = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=True,
        )
    except Exception:
        logger.exception("build_hermes_tools: failed to resolve tool definitions")
        definitions = []

    for tool_def in definitions:
        # get_tool_definitions returns OpenAI-format entries:
        # ``{"type": "function", "function": {"name": ..., ...}}``. Fall back to
        # a flat dict so a future schema shape (or a test double) still works.
        fn = tool_def.get("function") if isinstance(tool_def, dict) else None
        if not isinstance(fn, dict):
            fn = tool_def if isinstance(tool_def, dict) else {}
        tool_name = fn.get("name", "")
        if not tool_name:
            continue
        tools.append(
            _structured_tool_from_schema(
                tool_name, fn.get("description", ""), fn.get("parameters")
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Streaming Bridge
# ---------------------------------------------------------------------------


def _split_stream_item(item):
    """Normalize a LangGraph ``stream`` item into ``(mode, payload)``.

    With ``stream_mode=["updates", "messages"]`` and ``subgraphs=False`` each
    item is a ``(mode, payload)`` 2-tuple. Be liberal so a single-mode stream
    (a bare ``updates`` dict) — or a test double — still routes: an unmarked
    dict is treated as an ``updates`` payload.
    """
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
    ):
        return item[0], item[1]
    if isinstance(item, dict):
        return "updates", item
    return None, None


def _split_chunk_content(chunk):
    """Return ``(visible_text, thinking_text)`` for an ``AIMessageChunk``.

    ``content`` is a plain string for OpenAI-style providers and a list of
    content-block dicts for Anthropic-style ones (text vs. extended-thinking
    blocks arrive interleaved). Reasoning may also ride ``additional_kwargs.
    reasoning_content`` (DeepSeek/vLLM). Visible text and reasoning are split so
    they can drive ``stream_delta`` and ``thinking`` separately, matching the
    native runtime (which never leaks think-block text into the assistant body).
    """
    content = getattr(chunk, "content", None)
    text_parts: list[str] = []
    think_parts: list[str] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype in ("text", "text_delta"):
                    text_parts.append(part.get("text", "") or "")
                elif ptype in (
                    "thinking",
                    "thinking_delta",
                    "reasoning",
                    "reasoning_content",
                ):
                    think_parts.append(
                        part.get("thinking")
                        or part.get("reasoning")
                        or part.get("text", "")
                        or ""
                    )

    # Separate reasoning channel. Different providers surface it under
    # different keys: ``reasoning_content`` (DeepSeek/most vLLM builds) or
    # ``reasoning`` (some vLLM builds / OpenAI-style). Also accept a top-level
    # ``reasoning`` attr the integration may attach.
    ak = getattr(chunk, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        rc = ak.get(key)
        if isinstance(rc, str) and rc:
            think_parts.append(rc)
    top = getattr(chunk, "reasoning", None)
    if isinstance(top, str) and top:
        think_parts.append(top)

    return "".join(t for t in text_parts if t), "".join(t for t in think_parts if t)


def _stringify_tool_content(content):
    """Flatten a ToolMessage ``content`` (str or content-block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", "") or "")
        return "".join(parts)
    return "" if content is None else str(content)


# Reasoning markers some models (Qwen3.x on vLLM under a rich agentic prompt)
# leak into the *content* channel — the live reasoning already rides the
# thinking channel, so these regions must be suppressed from the visible body.
_REASON_OPEN = ("<think>", "<|mask_start|>", "<mask_start>")
_REASON_CLOSE = ("</think>", "<|mask_end|>", "<mask_end>")
_REASON_MAXTOK = max(len(t) for t in _REASON_OPEN + _REASON_CLOSE)


def _split_reasoning_from_content(text: str) -> tuple[str, str]:
    """Split a finished string into ``(clean_body, reasoning)`` by extracting
    ``<think>…</think>`` / ``<|mask_start|>…<mask_end>`` regions. Used for the
    stored final response, where the whole content is available — the body
    stays clean and the extracted reasoning is preserved as the turn's thought."""
    if not text:
        return text, ""
    import re

    pattern = re.compile(
        r"<\|?(?:think|mask_start)\|?>(.*?)<\|?(?:/think|mask_end)\|?>",
        re.DOTALL,
    )
    reasoning = "\n".join(m.group(1).strip() for m in pattern.finditer(text) if m.group(1).strip())
    body = pattern.sub("", text)
    # Drop any unpaired stray markers left behind.
    for mk in (*_REASON_OPEN, *_REASON_CLOSE):
        body = body.replace(mk, "")
    return body, reasoning


class _ReasoningContentRouter:
    """Stateful, streaming-safe router that splits a content stream into
    ``(visible, reasoning)``. Some models leak their reasoning into the content
    channel wrapped in ``<think>…</think>`` / ``<|mask_start|>…<mask_end>``;
    rather than drop it, we route the inside-marker text to the *thinking*
    channel (so it shows as reasoning) and keep the body clean. Markers may be
    split across chunks, so a short tail is held back until it can't be the
    start of a marker."""

    def __init__(self):
        self._buf = ""
        self._in_reasoning = False

    @staticmethod
    def _earliest(buf, markers):
        best, best_mk = -1, ""
        for mk in markers:
            i = buf.find(mk)
            if i >= 0 and (best < 0 or i < best):
                best, best_mk = i, mk
        return best, best_mk

    @staticmethod
    def _partial_suffix_len(buf, markers):
        """Length of the longest tail of *buf* that is a proper prefix of some
        marker — the only part that could still grow into a real marker and so
        must be held back. Clean text yields 0 (emit immediately)."""
        best = 0
        for mk in markers:
            for k in range(min(len(buf), len(mk) - 1), 0, -1):
                if buf.endswith(mk[:k]):
                    best = max(best, k)
                    break
        return best

    def feed(self, text: str) -> tuple[str, str]:
        """Split *text* into ``(visible, reasoning)`` across the marker state."""
        self._buf += text
        visible: list[str] = []
        reasoning: list[str] = []
        while True:
            if not self._in_reasoning:
                # Look for ANY marker: an open switches us into reasoning, while
                # a stray (unpaired) close marker is just a reasoning delimiter
                # the model emitted without its open — drop it from the body.
                pos, mk = self._earliest(self._buf, _REASON_OPEN + _REASON_CLOSE)
                if pos < 0:
                    hold = self._partial_suffix_len(
                        self._buf, _REASON_OPEN + _REASON_CLOSE
                    )
                    safe = len(self._buf) - hold
                    if safe > 0:
                        visible.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                visible.append(self._buf[:pos])
                self._buf = self._buf[pos + len(mk):]
                if mk in _REASON_OPEN:
                    self._in_reasoning = True
                # else: stray close marker — already dropped, stay visible.
            else:
                pos, mk = self._earliest(self._buf, _REASON_CLOSE)
                if pos < 0:
                    hold = self._partial_suffix_len(self._buf, _REASON_CLOSE)
                    safe = len(self._buf) - hold
                    if safe > 0:
                        reasoning.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                reasoning.append(self._buf[:pos])
                self._buf = self._buf[pos + len(mk):]
                self._in_reasoning = False
        return "".join(visible), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        """Emit any held-back tail at stream end as visible or reasoning."""
        tail, self._buf = self._buf, ""
        return ("", tail) if self._in_reasoning else (tail, "")


# ---------------------------------------------------------------------------
# Text-embedded tool-call recovery
# ---------------------------------------------------------------------------
#
# Some OpenAI-compatible backends (vLLM started without a matching
# ``--tool-call-parser``, or aggressively quantized models like NVFP4 that drift
# from the trained tool-call grammar) return tool calls inline as text instead
# of structured ``tool_calls``. The model emits, e.g.::
#
#     <model_tool_calls>
#     <tool name="web_search"><query>spain results</query></tool>
#     </model_tool_calls>
#
# The agent graph then sees no tool to run and the raw XML lands on screen. We
# parse that text back into real tool calls (and hide the XML) so the tool
# actually executes — a client-side mitigation for a serving-side gap.

_MODEL_TOOL_CALLS_OPEN = "<model_tool_calls>"
_MODEL_TOOL_CALLS_CLOSE = "</model_tool_calls>"
_MODEL_TOOL_CALLS_BLOCK_RE = re.compile(
    r"<model_tool_calls>(.*?)</model_tool_calls>", re.DOTALL | re.IGNORECASE
)
_MODEL_TOOL_CALLS_OPEN_RE = re.compile(
    r"<model_tool_calls>.*\Z", re.DOTALL | re.IGNORECASE
)
_TOOL_BLOCK_RE = re.compile(
    r'<tool\b[^>]*\bname\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</tool>',
    re.DOTALL | re.IGNORECASE,
)
_TOOL_ARG_RE = re.compile(r"<([A-Za-z_][\w.-]*)\s*>(.*?)</\1>", re.DOTALL)


def _parse_text_tool_calls(text: str) -> list[dict]:
    """Best-effort parse of text-embedded ``<model_tool_calls>`` blocks into
    ``[{"name", "args", "id"}, ...]``. Returns ``[]`` when none are present or
    parsing fails — callers fall back to the model's structured tool_calls."""
    if not text or "<tool" not in text.lower():
        return []
    blocks = _MODEL_TOOL_CALLS_BLOCK_RE.findall(text)
    if not blocks:
        # Unclosed wrapper or bare <tool> blocks the model emitted without the
        # outer marker — scan the whole text as one block.
        blocks = [text]
    calls: list[dict] = []
    for block in blocks:
        for name, body in _TOOL_BLOCK_RE.findall(block):
            args = {
                arg_name: arg_val.strip()
                for arg_name, arg_val in _TOOL_ARG_RE.findall(body)
            }
            calls.append(
                {
                    "name": name.strip(),
                    "args": args,
                    "id": f"tcrepair_{uuid.uuid4().hex[:12]}",
                }
            )
    return calls


def _strip_text_tool_calls(text: str) -> str:
    """Remove ``<model_tool_calls>`` blocks (closed or trailing-unclosed) from
    visible content once they've been parsed into structured tool calls."""
    out = _MODEL_TOOL_CALLS_BLOCK_RE.sub("", text)
    out = _MODEL_TOOL_CALLS_OPEN_RE.sub("", out)
    return out.strip()


def _repair_chat_result(result) -> None:
    """Populate ``tool_calls`` from text-embedded blocks when the server
    returned none, stripping the XML from the visible content. Mutates the
    ChatResult's AIMessages in place; no-op when real tool_calls already exist."""
    for gen in getattr(result, "generations", None) or []:
        msg = getattr(gen, "message", None)
        if not isinstance(msg, AIMessage) or getattr(msg, "tool_calls", None):
            continue
        if not isinstance(msg.content, str):
            continue
        parsed = _parse_text_tool_calls(msg.content)
        if not parsed:
            continue
        msg.tool_calls = parsed
        msg.content = _strip_text_tool_calls(msg.content)


class _ToolCallContentRouter:
    """Streaming-safe filter that removes ``<model_tool_calls>`` blocks from a
    content token stream and captures their bodies for later parsing. Mirrors
    :class:`_ReasoningContentRouter`: markers may be split across chunks, so a
    short tail that could still grow into the open marker is held back."""

    def __init__(self):
        self._buf = ""
        self._capturing = False
        self.captured: list[str] = []

    @staticmethod
    def _partial_suffix_len(buf: str, marker: str) -> int:
        for k in range(min(len(buf), len(marker) - 1), 0, -1):
            if buf.endswith(marker[:k]):
                return k
        return 0

    def feed(self, text: str) -> str:
        """Return the visible portion of *text*, capturing tool-call blocks."""
        self._buf += text
        visible: list[str] = []
        while True:
            if not self._capturing:
                pos = self._buf.find(_MODEL_TOOL_CALLS_OPEN)
                if pos < 0:
                    hold = self._partial_suffix_len(self._buf, _MODEL_TOOL_CALLS_OPEN)
                    safe = len(self._buf) - hold
                    if safe > 0:
                        visible.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                visible.append(self._buf[:pos])
                self._buf = self._buf[pos + len(_MODEL_TOOL_CALLS_OPEN):]
                self._capturing = True
            else:
                pos = self._buf.find(_MODEL_TOOL_CALLS_CLOSE)
                if pos < 0:
                    break  # keep buffering the block body until it closes
                self.captured.append(self._buf[:pos])
                self._buf = self._buf[pos + len(_MODEL_TOOL_CALLS_CLOSE):]
                self._capturing = False
        return "".join(visible)

    def flush(self) -> str:
        """At stream end: a still-open block is treated as captured (unclosed);
        any leftover non-block tail is returned as visible text."""
        if self._capturing:
            self.captured.append(self._buf)
            self._buf = ""
            self._capturing = False
            return ""
        tail, self._buf = self._buf, ""
        return tail

    def parsed_tool_calls(self) -> list[dict]:
        return _parse_text_tool_calls("".join(self.captured)) if self.captured else []


def _tool_calls_to_chunks(parsed: list[dict]) -> list[dict]:
    """Convert parsed tool calls into ``tool_call_chunks`` for an
    ``AIMessageChunk`` (args serialized to JSON, one index per call)."""
    return [
        {
            "name": call["name"],
            "args": json.dumps(call["args"], ensure_ascii=False),
            "id": call["id"],
            "index": i,
        }
        for i, call in enumerate(parsed)
    ]


def _repair_stream(chunks):
    """Wrap a sync chat stream: hide ``<model_tool_calls>`` text and, at the
    end, emit a synthetic chunk carrying the recovered tool calls. Passes the
    stream through untouched once the server emits real structured tool calls."""
    router = _ToolCallContentRouter()
    saw_structured = False
    for chunk in chunks:
        msg = getattr(chunk, "message", None)
        if msg is not None and getattr(msg, "tool_call_chunks", None):
            saw_structured = True
        if not saw_structured and msg is not None and isinstance(msg.content, str):
            msg.content = router.feed(msg.content)
        yield chunk
    if saw_structured:
        return
    tail = router.flush()
    parsed = router.parsed_tool_calls()
    if tail:
        yield ChatGenerationChunk(message=AIMessageChunk(content=tail))
    if parsed:
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", tool_call_chunks=_tool_calls_to_chunks(parsed)
            )
        )


async def _repair_astream(chunks):
    """Async counterpart of :func:`_repair_stream`."""
    router = _ToolCallContentRouter()
    saw_structured = False
    async for chunk in chunks:
        msg = getattr(chunk, "message", None)
        if msg is not None and getattr(msg, "tool_call_chunks", None):
            saw_structured = True
        if not saw_structured and msg is not None and isinstance(msg.content, str):
            msg.content = router.feed(msg.content)
        yield chunk
    if saw_structured:
        return
    tail = router.flush()
    parsed = router.parsed_tool_calls()
    if tail:
        yield ChatGenerationChunk(message=AIMessageChunk(content=tail))
    if parsed:
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", tool_call_chunks=_tool_calls_to_chunks(parsed)
            )
        )


_REPAIR_CHAT_CLASS = None


def _make_repair_chat_openai(**kwargs):
    """Build a ChatOpenAI that recovers text-embedded tool calls on both the
    invoke and streaming paths. The subclass is defined lazily (ChatOpenAI is an
    optional import) and cached so the type is stable across calls."""
    global _REPAIR_CHAT_CLASS
    if _REPAIR_CHAT_CLASS is None:
        from langchain_openai import ChatOpenAI

        class _ToolCallRepairChatOpenAI(ChatOpenAI):
            """ChatOpenAI variant that parses ``<model_tool_calls>`` text the
            server failed to surface as structured tool_calls (see module
            notes), keeping live token streaming for ordinary text."""

            def _generate(self, messages, stop=None, run_manager=None, **kw):
                result = super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kw
                )
                try:
                    _repair_chat_result(result)
                except Exception:
                    logger.debug("tool-call text repair failed (_generate)", exc_info=True)
                return result

            async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
                result = await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kw
                )
                try:
                    _repair_chat_result(result)
                except Exception:
                    logger.debug("tool-call text repair failed (_agenerate)", exc_info=True)
                return result

            def _stream(self, *args, **kw):
                yield from _repair_stream(super()._stream(*args, **kw))

            async def _astream(self, *args, **kw):
                async for chunk in _repair_astream(super()._astream(*args, **kw)):
                    yield chunk

        _REPAIR_CHAT_CLASS = _ToolCallRepairChatOpenAI
    return _REPAIR_CHAT_CLASS(**kwargs)


class _HermesStreamingBridge:
    """Wires LangGraph stream events to Hermes callback forwarding.

    The bridge holds the Hermes callbacks captured from the agent (populated by
    the gateway via ``__setattr__``) and translates *real* LangGraph stream
    items into the same callback signatures the native runtime emits, so a
    gateway (TUI, ACP, messaging) sees an identical event stream regardless of
    runtime:

      * ``messages`` mode → token-level ``AIMessageChunk`` → ``stream_delta``
        (visible text) and ``thinking`` (reasoning), plus a ``tool_gen``
        "preparing…" beat as a tool name first streams in.
      * ``updates`` mode → completed node output → ``tool_start`` for each
        ``AIMessage.tool_calls`` entry and ``tool_complete`` for each
        ``ToolMessage`` (the authoritative tool chrome the TUI renders).

    The earlier implementation matched ``{"events": …}`` / ``{"output": …}``
    shapes that LangGraph never emits in ``updates``/``messages`` mode, so no
    callback ever fired mid-turn and the whole reply landed at once.
    """

    def __init__(
        self,
        agent,
        stream_delta=None,
        tool_progress=None,
        thinking=None,
        step=None,
        tool_start=None,
        tool_complete=None,
        tool_gen=None,
        input_tool_ids=None,
    ):
        self._agent = agent
        self._stream_delta = stream_delta
        self._tool_progress = tool_progress
        self._thinking = thinking
        self._step = step
        self._tool_start = tool_start
        self._tool_complete = tool_complete
        self._tool_gen = tool_gen
        # Tool-call ids already present in the INPUT history. In ``updates`` mode
        # the real graph re-surfaces the input messages (so the model keeps
        # context), which would otherwise make the bridge re-emit tool_start/
        # tool_complete for every prior turn's tools — the whole tool trail
        # reprints on each new turn. Skipping these ids renders only THIS turn's
        # new tools, while leaving the no-echo (pure-delta) case untouched.
        self._input_tool_ids = set(input_tool_ids or ())
        # Tool-call id → (name, args), so a ToolMessage can echo the original
        # call's name/args on completion (ToolMessage carries neither reliably).
        self._tool_args: dict[str, tuple[str, dict]] = {}
        # Tool names already announced via ``tool_gen`` this turn, cleared when
        # the call actually starts so a later same-named call re-announces.
        self._gen_announced: set[str] = set()
        # Route reasoning the model leaks into the content channel (see
        # _ReasoningContentRouter) to the thinking channel instead of the body.
        self._leak = _ReasoningContentRouter()

    def any_callbacks_set(self):
        return any(
            cb is not None
            for cb in (
                self._stream_delta,
                self._tool_progress,
                self._thinking,
                self._step,
                self._tool_start,
                self._tool_complete,
                self._tool_gen,
            )
        )

    @staticmethod
    def _safe_call(cb, *args):
        """Invoke a forwarded callback, swallowing any exception.

        Forwarded callbacks (TUI websocket emit, CLI render, etc.) run inside
        the stream-consumption loop. An unguarded raise would propagate out of
        ``process_stream_item`` and abort the whole turn — the graph stops, the
        tool never executes, and the final answer never streams, leaving only
        the tool box on screen. The native runtime isolates these the same way.
        """
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            logger.debug("deepagents stream callback failed", exc_info=True)

    def flush_visible(self):
        """Flush any text the reasoning router held back. Called once after the
        stream drains so a trailing partial-marker tail isn't lost."""
        tail_visible, tail_reasoning = self._leak.flush()
        if tail_reasoning:
            self._safe_call(self._thinking, tail_reasoning)
        if tail_visible:
            self._safe_call(self._stream_delta, tail_visible)

    def process_stream_item(self, item):
        """Route one LangGraph ``stream`` item to the Hermes callbacks."""
        mode, payload = _split_stream_item(item)
        if mode == "messages":
            self._handle_messages(payload)
        elif mode == "updates":
            self._handle_updates(payload)

    # -- messages mode: token-level streaming --------------------------------

    def _handle_messages(self, payload):
        """Handle a ``messages`` item: ``(AIMessageChunk, metadata)``."""
        chunk = payload
        if isinstance(payload, tuple) and payload:
            chunk = payload[0]
        if chunk is None:
            return

        # ``messages`` mode yields *every* message a node produces, not just
        # the model's tokens — the tool node's ToolMessage (whose ``content``
        # is the tool *result*) flows through here too. Only AI output is
        # visible assistant text; tool chrome is owned by ``updates`` mode.
        # ToolMessage/HumanMessage/SystemMessage are not AIMessage subclasses,
        # while AIMessageChunk is — so this guard keeps live token streaming.
        if not isinstance(chunk, AIMessage):
            return

        text, thinking = _split_chunk_content(chunk)
        if thinking:
            self._safe_call(self._thinking, thinking)
        if text:
            # Reasoning the model leaks into the content channel is routed to
            # the thinking channel (so it still shows as the agent's thought)
            # while the visible body stays clean.
            visible, leaked_reasoning = self._leak.feed(text)
            if leaked_reasoning:
                self._safe_call(self._thinking, leaked_reasoning)
            if visible:
                self._safe_call(self._step, 1, [])  # minimal step marker
                self._safe_call(self._stream_delta, visible)

        # "preparing <tool>…" beat as the tool name first streams in.
        if self._tool_gen:
            for tc in getattr(chunk, "tool_call_chunks", None) or []:
                name = (tc or {}).get("name")
                if name and name not in self._gen_announced:
                    self._gen_announced.add(name)
                    self._safe_call(self._tool_gen, name)

    # -- updates mode: completed tool calls / results ------------------------

    def _handle_updates(self, payload):
        """Handle an ``updates`` item: ``{node_name: {"messages": [...]}}``."""
        if not isinstance(payload, dict):
            return
        for node_update in payload.values():
            if not isinstance(node_update, dict):
                continue
            for msg in node_update.get("messages") or []:
                self._handle_update_message(msg)

    def _handle_update_message(self, msg):
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                self._emit_tool_start(tc)
        elif isinstance(msg, ToolMessage):
            self._emit_tool_complete(msg)

    def _emit_tool_start(self, tc):
        if not isinstance(tc, dict):
            return
        name = tc.get("name") or ""
        args = tc.get("args")
        if not isinstance(args, dict):
            args = {}
        tc_id = tc.get("id") or f"call_{len(self._tool_args)}"
        # Historical tool call re-surfaced from the input — already rendered on
        # the turn that produced it. Skip so the trail doesn't reprint.
        if tc_id in self._input_tool_ids:
            return
        self._tool_args[tc_id] = (name, args)
        # Fallback "preparing…" beat: LangGraph commonly surfaces a tool call
        # only as a completed AIMessage here in ``updates`` mode, never as an
        # incremental ``tool_call_chunk`` in ``messages`` mode — so the
        # _handle_messages announce path never ran. Emit it once before
        # tool_start so the generating indicator shows like the native runtime.
        # If chunks already announced this name, ``_gen_announced`` holds it and
        # we skip, avoiding a double beat.
        if self._tool_gen and name and name not in self._gen_announced:
            self._safe_call(self._tool_gen, name)
        self._gen_announced.discard(name)
        self._safe_call(self._tool_start, tc_id, name, args)
        self._safe_call(
            self._tool_progress, "tool.started", name, str(args)[:200], args
        )

    def _emit_tool_complete(self, msg):
        tc_id = getattr(msg, "tool_call_id", "") or ""
        # Historical tool result re-surfaced from the input — its start was
        # skipped above, so skip the completion to keep the trail in sync.
        if tc_id in self._input_tool_ids:
            return
        name, args = self._tool_args.pop(
            tc_id, (getattr(msg, "name", "") or "", {})
        )
        if not name:
            name = getattr(msg, "name", "") or ""
        result = _stringify_tool_content(getattr(msg, "content", ""))
        self._safe_call(self._tool_complete, tc_id, name, args, result)
        self._safe_call(
            self._tool_progress, "tool.completed", name, result[:200], args
        )


# ---------------------------------------------------------------------------
# Main Agent Class
# ---------------------------------------------------------------------------


class DeepAgentsAIAgent:
    """Hermes agent backed by DeepAgents SDK (LangGraph).

    Accepts config directly — no child AIAgent instantiation.
    Public API matches run_agent.AIAgent:
      - run_conversation(user_message, system_message, conversation_history, task_id)
      - chat(message)

    Callback forwarding: known callback attributes are stored in ``_callbacks``
    and served via ``__getattr__`` so gateway attribute-setting still works.
    """

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

    # Attributes initialized during __init__ that must not raise
    # AttributeError when read on partially-constructed or mock agents.
    _DEFAULT_ATTRS = frozenset((
        "_quiet_mode",
        "_skip_memory",
        "_platform",
        "_session_id",
        "_max_iterations",
        "_langgraph_checkpointer",
        "_langgraph_store",
        "_langsmith_api_key",
        "_langsmith_project",
        "_langsmith_tags",
        "_ls_api_key",
        "_ls_project",
        "_ls_tags",
        "_checkpointer",
        "_store",
        "_agent",
        "_callbacks",
        "_langfuse_handler",
        "_debug",
        "_mcp_discovery_timeout",
        "_build_kwargs",
        "_agent_lock",
    ))

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        model: str = "",
        max_iterations: int = 90,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        quiet_mode: bool = False,
        skip_memory: bool = False,
        skip_context_files: bool = False,
        session_id: str = None,
        platform: str = None,
        runtime: str = "deepagents",
        langgraph_checkpointer: bool = False,  # Enable checkpointing / tracing
        langgraph_store: bool = False,  # Enable LangGraph store
        # (The gateway / AIAgent facade also passes the following via __setattr__)
        # - debug                               # Enable debug-level runnable traces
        # - langsmith_api_key                   # LangSmith project key (env)
        # - langsmith_project                   # LangSmith project name
        # - langsmith_tags                      # LangSmith tags list
        **kwargs,
    ):
        self.mode = "deepagents"
        self._quiet_mode = quiet_mode
        self._skip_memory = skip_memory
        self._platform = platform
        self._session_id = session_id or ""
        self._max_iterations = max_iterations
        self.provider = provider or ""
        self._base_url = base_url
        self._api_key = api_key
        self._langgraph_checkpointer = langgraph_checkpointer
        self._langgraph_store = langgraph_store
        self._debug: bool = False
        # Upper bound (seconds) for waiting on MCP discovery before baking the
        # tool list. The background thread self-bounds on each server's
        # connect_timeout, so this is just a safety ceiling. Override via env
        # for slow/remote servers (e.g. the NyxStrike sidecar in K8s).
        try:
            self._mcp_discovery_timeout = float(
                os.environ.get("HERMES_DEEPAGENTS_MCP_DISCOVERY_TIMEOUT", "120")
            )
        except (TypeError, ValueError):
            self._mcp_discovery_timeout = 120.0
        self._langsmith_api_key: str | None = None
        self._langsmith_project: str = "hermes"
        self._langsmith_tags: list[str] = ["hermes"]

        # Resolve model string (LangChain format – no provider prefix).
        # Keep the raw name too: OpenAI-compatible endpoints (vLLM) serve
        # models under their full id (e.g. "nvidia/Qwen…"), prefix included.
        self._model_raw = model or ""
        model_str = self._resolve_model(model)
        if not model_str:
            model_str = self._default_model()

        # Set env vars for LangChain model bindings
        _inject_provider_env(provider, base_url, api_key)

        # LangSmith tracing defaults (set before _build_langgraph_agent so
        # they exist even when __init__ is bypassed by test mocks).
        self._ls_api_key = None
        self._ls_project = "hermes"
        self._ls_tags = ["hermes"]

        # Langfuse tracing: memoized handler (None = not yet built,
        # False = checked-and-unavailable, otherwise the CallbackHandler).
        self._langfuse_handler = None
        # Seed credentials from env so the Docker / gateway path works
        # out of the box. The gateway may still override these afterwards
        # via attribute assignment (captured into _callbacks). Unit tests
        # bypass __init__, so the handler logic itself never reads env.
        self._seed_langfuse_from_env()

        # Remember the construction inputs so the agent can be rebuilt in place
        # (see rebuild_agent): the compiled LangGraph graph bakes its tool list,
        # so picking up newly discovered MCP tools means recompiling and
        # atomically swapping. ``system_prompt=None`` keeps it lazily rebuilt.
        self._build_kwargs = dict(
            model=model_str,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            skip_context_files=skip_context_files,
            quiet_mode=quiet_mode,
            skip_memory=skip_memory,
            system_prompt=None,  # will be built lazily
        )
        # Guards the atomic swap of ``self._agent`` between rebuild_agent() and
        # the run paths (which snapshot the graph under this lock at entry).
        self._agent_lock = threading.Lock()

        # Seed gateway-style config that arrives as constructor kwargs. The
        # facade ALSO sets these via attribute, but only AFTER construction —
        # too late, since the graph (and its reasoning-aware model binding) is
        # built here in __init__. Capture them now so extended thinking takes
        # effect on the initial build, not just after a rebuild_agent().
        _rc = kwargs.get("reasoning_config")
        if _rc is not None:
            self.reasoning_config = _rc
        _st = kwargs.get("service_tier")
        if _st is not None:
            self.service_tier = _st

        # Build the LangGraph agent
        self._agent = self._build_langgraph_agent(**self._build_kwargs)

    def _ensure_mcp_discovery(self):
        """Make sure configured MCP servers are discovered *before* the
        LangGraph agent bakes its tool list.

        The native runtime re-snapshots tools every turn, so it tolerates MCP
        discovery finishing late (the background daemon thread started at
        process startup). This runtime compiles tools once at construction via
        ``create_deep_agent``, so any MCP tool that lands afterwards is
        invisible for the life of the agent. A remote / sidecar server (e.g.
        the NyxStrike MCP) routinely needs more than the shared 0.75s join, so:

          * if a background discovery thread is running, wait for it (bounded by
            ``_mcp_discovery_timeout``; the thread self-bounds on each server's
            ``connect_timeout``, so the join returns as soon as it gives up); or
          * if discovery was never started in this process, run it synchronously
            (``discover_mcp_tools`` is idempotent and bounded per server).

        Best-effort: any failure is logged and construction proceeds with
        whatever tools are available — discovery never blocks the agent from
        coming up.
        """
        # Cheap config probe first so non-MCP deepagents users never import the
        # MCP stack just to discover there's nothing to do.
        try:
            from hermes_cli.config import read_raw_config

            mcp_servers = (read_raw_config() or {}).get("mcp_servers")
            if not (isinstance(mcp_servers, dict) and mcp_servers):
                return
        except Exception:
            # Probe failed — fall through; discovery itself no-ops when no
            # servers are configured, so this is safe and conservative.
            pass

        try:
            from hermes_cli import mcp_startup
        except Exception:
            mcp_startup = None

        started = getattr(mcp_startup, "_mcp_discovery_started", False)
        thread = getattr(mcp_startup, "_mcp_discovery_thread", None)
        if mcp_startup is not None and (started or thread is not None):
            # Background discovery owns the connection lifecycle — wait for it
            # to land rather than racing it with a second discovery pass.
            try:
                mcp_startup.wait_for_mcp_discovery(
                    timeout=self._mcp_discovery_timeout
                )
            except Exception:
                logger.debug("wait_for_mcp_discovery failed", exc_info=True)
            return

        # Nobody kicked off background discovery in this process — do it
        # ourselves. Idempotent, and each server is bounded by its own
        # connect_timeout so a dead server can't hang construction forever.
        try:
            from tools.mcp_tool import discover_mcp_tools

            discover_mcp_tools()
        except Exception:
            logger.debug("synchronous MCP discovery failed", exc_info=True)

    def _resolve_model(self, model):
        """Resolve model string for LangChain / deep agents.

        DeepAgents uses ``create_deep_agent`` which internally calls
        LangChain's ``init_chat_model``.  That helper expects model names
        without the Hermes provider prefix – e.g. ``"gpt-4o"`` not
        ``"openai/gpt-4o"``.  The provider itself is injected via the
        ``_inject_provider_env`` side-channel (env vars).

        Also handles provider-specific aliases (e.g. "gemini" ->
        "google/gemini-pro").
        """
        if not model:
            return None
        # Strip provider prefix if present (provider/model -> model)
        if "/" in model and not model.startswith("http"):
            model = model.split("/", 1)[1]
        return model

    def _default_model(self, fallback="gpt-4o"):
        """Return a default model based on the configured provider."""
        p = (self.provider or "").lower()
        if "anthropic" in p:
            return "claude-sonnet-4-0"
        elif "google" in p:
            return "gemini-2.0-flash"
        return fallback

    def _build_hermes_system_prompt(self) -> str:
        """Render the full Hermes system prompt for the deepagents runtime.

        ``build_system_prompt`` reads a broad set of attributes off the native
        ``AIAgent``; this runtime is a separate class without those subsystems.
        We hand it a lightweight read-only view populated from the impl's real
        config (model / provider / tool names) with safe defaults for the
        pieces deepagents handles differently (context files via
        FilesystemMiddleware, memory via _HermesMiddleware) so identity,
        tool-use enforcement and skills guidance are all injected. Falls back
        to the real agent identity — never the bare "helpful AI assistant"
        string — if assembly fails.
        """
        try:
            from types import SimpleNamespace
            from agent.system_prompt import build_system_prompt

            status_cb = self._get_cap("status_callback")

            def _emit_status(*args, **kwargs):
                if status_cb:
                    try:
                        status_cb("status", args[0] if args else "")
                    except Exception:
                        pass

            view = SimpleNamespace(
                # identity: load SOUL.md persona regardless of context files
                load_soul_identity=True,
                # deepagents has no file subsystem — context files (AGENTS.md,
                # .cursorrules) are out of scope here; identity still loads.
                skip_context_files=True,
                # tool-aware guidance keys off the real bound tool names
                valid_tool_names=self.valid_tool_names,
                tools=[],
                get_toolset_for_tool=lambda _name: None,
                _tool_use_enforcement="auto",
                tool_use_enforcement="auto",
                _task_completion_guidance=True,
                task_completion_guidance=True,
                coding_context=False,
                # runtime identity
                model=getattr(self, "_model_raw", "") or "",
                provider=self.provider,
                platform=getattr(self, "_platform", None),
                session_id=getattr(self, "_session_id", "") or "",
                pass_session_id=False,
                # memory / profile blocks handled by middleware, not the prompt
                _memory_enabled=False,
                _user_profile_enabled=False,
                _memory_manager=None,
                _memory_store=None,
                # subsystems the native prompt may touch — unused with
                # skip_context_files / memory disabled, but must exist.
                environment_probe=None,
                file_safety=None,
                prompt_builder=None,
                runtime_cwd=None,
                _cached_system_prompt=None,
                _emit_status=_emit_status,
            )
            prompt = build_system_prompt(view)
            if prompt and prompt.strip():
                return prompt
        except Exception:
            logger.warning(
                "deepagents: full system prompt assembly failed; "
                "falling back to agent identity",
                exc_info=True,
            )

        # Fallback: the real agent identity (NOT a generic assistant string),
        # so tool-use behavior degrades as little as possible.
        try:
            from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

            return DEFAULT_AGENT_IDENTITY
        except Exception:
            return "You are Hermes, an autonomous agent. Use your tools to act."

    def _build_langgraph_agent(
        self,
        model,
        enabled_toolsets,
        disabled_toolsets,
        skip_context_files,
        quiet_mode,
        skip_memory,
        system_prompt,
    ):
        """Construct the DeepAgents LangGraph agent with Hermes tools."""
        hermes_home = get_hermes_home()

        # Build system prompt lazily. This must be the *full* Hermes prompt
        # (identity + tool-use enforcement + skills guidance) — running with a
        # bare "You are a helpful AI assistant." strips the enforcement that
        # makes the model actually call tools instead of describing them or
        # typing tool names as text.
        if system_prompt is None:
            system_prompt = self._build_hermes_system_prompt()

        # Middleware - pass through for create_deep_agent auto-stack
        # (FilesystemMiddleware, TodoListMiddleware, etc. are added by default).
        # User middleware is inserted between base and tail layers.
        middlewares = []

        # Memory middleware (Hermes-provided)
        if not skip_memory:
            from agent.deep_agents_middleware import _HermesMiddleware

            middlewares.append(_HermesMiddleware(self))

        # MCP tools are registered asynchronously by background discovery.
        # Block until that lands before snapshotting tools: this runtime bakes
        # its tool list once and (unlike native) cannot pick up late arrivals.
        self._ensure_mcp_discovery()

        # Build Hermes tools from registry
        tools = build_hermes_tools(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
        )

        # Optional: LangGraph checkpointer for tracing / state persistence
        checkpointer = None
        if self._langgraph_checkpointer:
            checkpointer = MemorySaver()
            logger.info("checkpointer enabled")

        # Optional: LangGraph store for persistent memory
        store = None
        if self._langgraph_store:
            store = InMemoryStore()
            logger.info("store enabled")

        # Enable debug / LangSmith tracing if configured
        debug_val = self._get_cap("debug") or False
        if debug_val is True:
            logger.info("debug traces enabled")

        # --- LangSmith env setup ---------------------------------------------------
        ls_api_key = self._langsmith_api_key or os.environ.get("LANGSMITH_API_KEY")
        ls_project = self._langsmith_project or "hermes"
        ls_tags = self._langsmith_tags or ["hermes"]

        if ls_api_key:
            os.environ["LANGSMITH_API_KEY"] = ls_api_key
            os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
            os.environ.setdefault("LANGSMITH_PROJECT", ls_project)
            logger.info("LangSmith tracing enabled (project='%s')", ls_project)
        else:
            ls_tags = []
            ls_project = ""

        # Extended thinking: translate the captured ``reasoning_config`` into the
        # provider binding's thinking/reasoning params. Without this the model is
        # built from a bare string and never enables thinking, so the effort the
        # gateway/TUI selected is silently dropped. Skipped for custom base_url
        # endpoints (handled below) since those serve arbitrary vLLM models whose
        # reasoning support we can't assume.
        if isinstance(model, str) and not self._base_url:
            _reasoning_model = _build_reasoning_model(
                self.provider, model, self._get_cap("reasoning_config")
            )
            if _reasoning_model is not None:
                model = _reasoning_model
                logger.info(
                    "deepagents: extended thinking enabled (effort=%s)",
                    _reasoning_effort(self._get_cap("reasoning_config")),
                )

        # Custom / OpenAI-compatible endpoints (vLLM, llama.cpp, …) can't be
        # inferred by LangChain's ``init_chat_model`` from the bare model
        # name — build the ChatOpenAI client explicitly against the
        # configured base_url instead of passing a string through.
        if isinstance(model, str) and self._base_url:
            _p = (self.provider or "").lower()
            if _p in ("", "custom", "openai"):
                # Use the tool-call-repair variant: vLLM/quantized endpoints
                # sometimes emit tool calls as ``<model_tool_calls>`` text the
                # OpenAI-compatible layer doesn't parse, so the graph would see
                # no tool and the XML would land on screen (see module notes).
                model = _make_repair_chat_openai(
                    # The endpoint serves the model under its full id
                    # (prefix included) — don't use the stripped name.
                    model=getattr(self, "_model_raw", None) or model,
                    base_url=self._base_url,
                    api_key=self._api_key or "EMPTY",
                )

        # Create the LangGraph agent (recursion_limit set per-call via config)
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middlewares,
            checkpointer=checkpointer,
            store=store,
            debug=bool(debug_val),
            name="hermes-agent",
        )

        # Store refs for per-call tracing hooks
        self._checkpointer = checkpointer
        self._store = store
        self._ls_api_key = ls_api_key
        self._ls_project = ls_project
        self._ls_tags = ls_tags

        return agent

    # ------------------------------------------------------------------
    # Hot rebuild — recompile the graph to pick up tool-set changes
    # ------------------------------------------------------------------

    def _current_agent(self):
        """Return the live LangGraph agent, read under the swap lock.

        Run paths snapshot the graph here at entry so a concurrent
        :meth:`rebuild_agent` swap can't replace it mid-turn.
        """
        lock = getattr(self, "_agent_lock", None)
        if lock is not None:
            with lock:
                return self._agent
        return self._agent

    def rebuild_agent(self) -> bool:
        """Recompile the LangGraph agent and atomically swap it in.

        The compiled graph bakes its tool list at construction (``model.
        bind_tools`` + the ``ToolNode``), so tools discovered or removed
        afterwards — e.g. after an MCP ``reload-mcp`` — only take effect once
        the graph is rebuilt. The swap is a single reference assignment under
        ``_agent_lock``: a turn already in flight finishes on the old graph
        (conversation state lives in the messages / checkpointer, not the
        graph), and the next turn picks up the new one.

        Returns True on success, False if the rebuild failed — in which case the
        existing agent is kept so the session stays usable.
        """
        build_kwargs = getattr(self, "_build_kwargs", None)
        if not build_kwargs:
            logger.warning("rebuild_agent: no stored build params; skipping")
            return False
        try:
            new_agent = self._build_langgraph_agent(**build_kwargs)
        except Exception:
            logger.exception("rebuild_agent: failed to rebuild LangGraph agent")
            return False

        lock = getattr(self, "_agent_lock", None)
        if lock is not None:
            with lock:
                self._agent = new_agent
        else:
            self._agent = new_agent
        logger.info("DeepAgents agent rebuilt (tool list refreshed)")
        return True

    # ------------------------------------------------------------------
    # Callback forwarding (__setattr__ / _get_cap / __getattr__)
    #
    # The gateway sets callback attributes on the agent after __init__,
    # e.g. ``agent.tool_progress_callback = cb``.  We capture those in
    # _callbacks and serve them on read so run_conversation can find them.
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        """Forward known callback/config attributes to a stored dict."""
        if name in self._CAPTURED_NAMES:
            # ``hasattr`` is True even before initialization: ``_callbacks``
            # is in _DEFAULT_ATTRS, so __getattr__ returns None instead of
            # raising — check for None explicitly.
            if getattr(self, "_callbacks", None) is None:
                object.__setattr__(self, "_callbacks", {})
            self._callbacks[name] = value
        else:
            super().__setattr__(name, value)

    def _get_cap(self, name, default=None):
        """Read a forwarded callback / config attribute."""
        caps = getattr(self, "_callbacks", {})
        if caps is None:
            return default
        return caps.get(name, default)

    def __getattr__(self, name):
        if name in self._CAPTURED_NAMES:
            return self._get_cap(name)
        if name in self._DEFAULT_ATTRS:
            return None
        raise AttributeError(f"'{type(self).__name__}' object has no attr '{name}'")

    # ------------------------------------------------------------------
    # Conversation / Chat
    # ------------------------------------------------------------------

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: list = None,
        task_id: str = None,
        stream_callback=None,
    ) -> dict:
        """Run a single-turn conversation via DeepAgents/LangGraph.

        Returns the same dict shape as run_agent.AIAgent.run_conversation.
        Reads gateway-set callbacks (stream_delta_callback, etc.) from
        the forwarding dict so the gateway's per-turn attribute-setting
        pattern works without modification.
        """
        langchain_history = _convert_messages_to_langchain(conversation_history or [])
        if system_message:
            langchain_history = [
                SystemMessage(content=system_message)
            ] + langchain_history

        human = HumanMessage(content=user_message)
        messages = langchain_history + [human]

        config = {
            "configurable": {
                "session_id": self._session_id or "default",
                "task_id": task_id or "",
            },
        }

        # Langfuse tracing: the plugin hooks in conversation_loop.py never
        # fire under this runtime, so tracing rides LangChain's callback
        # system instead. Inert without Langfuse credentials. In v3 the
        # session is grouped via config metadata (``langfuse_session_id``),
        # which the CallbackHandler reads — the handler constructor no longer
        # accepts a session id.
        _lf_handler = self._get_langfuse_handler()
        if _lf_handler is not None:
            config["callbacks"] = [_lf_handler]
            if self._session_id:
                config.setdefault("metadata", {})["langfuse_session_id"] = (
                    self._session_id
                )

        # Add LangSmith tracing tags / metadata when tracing is enabled
        if self._ls_api_key:
            tags = list(self._ls_tags or [])
            if task_id:
                tags.append(f"task:{task_id}")
            if self._platform:
                metadata = {
                    "platform": self._platform,
                    "session_id": config["configurable"].get("session_id", ""),
                }
                config["metadata"] = metadata
            config["tags"] = tags

        # Collect forwarded callbacks so the streaming bridge can invoke them.
        # Visible text fans out to BOTH the gateway-set ``stream_delta_callback``
        # and the per-call ``stream_callback`` (the TUI passes the latter to
        # run_conversation, not as an attribute) — mirroring the native runtime,
        # which drives both sinks.
        delta_sinks = [
            cb
            for cb in (self._get_cap("stream_delta_callback"), stream_callback)
            if cb is not None
        ]

        def _delta(text):
            for cb in delta_sinks:
                try:
                    cb(text)
                except Exception:
                    logger.debug("stream delta sink failed", exc_info=True)

        # Tool-call ids already present in the input history. The real graph
        # re-surfaces these in ``updates`` mode (keeping model context), so the
        # bridge must skip them or every prior turn's tool trail reprints.
        input_tool_ids: set[str] = set()
        for _m in messages:
            if isinstance(_m, AIMessage):
                for _tc in getattr(_m, "tool_calls", None) or []:
                    _id = _tc.get("id") if isinstance(_tc, dict) else None
                    if _id:
                        input_tool_ids.add(_id)
            elif isinstance(_m, ToolMessage):
                _id = getattr(_m, "tool_call_id", "") or ""
                if _id:
                    input_tool_ids.add(_id)

        bridge = _HermesStreamingBridge(
            self,
            stream_delta=_delta if delta_sinks else None,
            tool_progress=self._get_cap("tool_progress_callback"),
            thinking=self._get_cap("thinking_callback"),
            step=self._get_cap("step_callback"),
            tool_start=self._get_cap("tool_start_callback"),
            tool_complete=self._get_cap("tool_complete_callback"),
            tool_gen=self._get_cap("tool_gen_callback"),
            input_tool_ids=input_tool_ids,
        )

        # Tracing for the DeepAgents runtime rides LangChain's callback system
        # (the Langfuse CallbackHandler wired into ``config["callbacks"]`` above),
        # which is the v2-compatible path and captures tool calls + token usage
        # natively. The conversation_loop.py plugin hooks are intentionally NOT
        # emitted here: the bundled langfuse plugin targets the Langfuse v3 SDK
        # API, so it would no-op on the pinned v2 SDK while risking duplicate
        # traces alongside the CallbackHandler.
        if bridge.any_callbacks_set():
            result = self._run_streamed(messages, config, task_id, bridge)
        elif stream_callback:
            result = self._run_streamed(messages, config, task_id, bridge)
        else:
            result = self._run_sync(messages, config, task_id)

        return result

    def _seed_langfuse_from_env(self):
        """Populate Langfuse credential callbacks from ``HERMES_LANGFUSE_*`` env.

        This is the env/Docker sourcing layer: it runs only from ``__init__``
        (bypassed by unit tests). The handler logic in ``_get_langfuse_handler``
        reads exclusively from these captured callbacks, never from env, so it
        stays deterministic and injectable in tests.
        """
        public_key = os.environ.get("HERMES_LANGFUSE_PUBLIC_KEY") or os.environ.get(
            "LANGFUSE_PUBLIC_KEY"
        )
        secret_key = os.environ.get("HERMES_LANGFUSE_SECRET_KEY") or os.environ.get(
            "LANGFUSE_SECRET_KEY"
        )
        host = os.environ.get("HERMES_LANGFUSE_BASE_URL") or os.environ.get(
            "LANGFUSE_BASE_URL"
        )
        if public_key:
            self.langfuse_public_key = public_key
        if secret_key:
            self.langfuse_secret_key = secret_key
        if host:
            self.langfuse_base_url = host

    def _get_langfuse_handler(self):
        """Return a memoized Langfuse ``CallbackHandler``, or ``None``.

        Credentials come from gateway-set callbacks (``langfuse_public_key`` /
        ``langfuse_secret_key`` / ``langfuse_base_url``), seeded from
        ``HERMES_LANGFUSE_*`` env at init. Built once per agent (session-scoped)
        and cached; ``False`` marks "checked and unavailable" so construction
        is never retried.

        Langfuse v3: the credentials live on a ``Langfuse`` client (a process
        singleton); the LangChain ``CallbackHandler`` takes no creds and reads
        from that client. The client is what flushes (the handler has no
        ``flush``), so we keep a reference on ``self._langfuse_client``. session
        grouping is applied per-call via config metadata (``langfuse_session_id``
        in ``run_conversation``), not the constructor.
        """
        cached = getattr(self, "_langfuse_handler", None)
        if cached is not None:
            return cached or None

        handler = False
        self._langfuse_client = None
        public_key = self._get_cap("langfuse_public_key")
        secret_key = self._get_cap("langfuse_secret_key")
        if LANGFUSE_AVAILABLE and public_key and secret_key:
            host = self._get_cap("langfuse_base_url") or "https://cloud.langfuse.com"
            try:
                self._langfuse_client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                handler = CallbackHandler()
                logger.info("Langfuse tracing enabled (host=%s)", host)
            except Exception as e:
                logger.warning("Langfuse callback unavailable: %s", e)
                handler = False
                self._langfuse_client = None

        self._langfuse_handler = handler
        return handler or None

    def _run_sync(self, messages, config, task_id):
        """Sync invocation."""
        result = None
        _flushed = False
        agent = self._current_agent()
        try:
            cfg = {**config, "recursion_limit": self._max_iterations}
            result = agent.invoke({"messages": messages}, config=cfg)
            self._maybe_flush_langfuse()
            _flushed = True
            return _parse_langgraph_result(result, task_id)
        except Exception as e:
            if not _flushed:
                self._maybe_flush_langfuse()
            logger.error("DeepAgents invoke failed: %s", e, exc_info=True)
            return _parse_error_result(e)

    def _run_streamed(self, messages, config, task_id, bridge):
        """Streaming invocation."""
        agent = self._current_agent()
        cfg = {**config, "recursion_limit": self._max_iterations}
        # ``["updates", "messages"]`` yields BOTH completed node outputs (for
        # tool-call / result chrome and the authoritative final messages) AND
        # token-level ``AIMessageChunk``s (for live text/thinking streaming).
        # ``updates`` alone delivers whole messages at node boundaries, which is
        # why text used to land all at once.
        stream_result = agent.stream(
            {"messages": messages},
            config=cfg,
            stream_mode=["updates", "messages"],
            subgraphs=False,
        )

        # Accumulate messages from the streamed updates as we go. The stream is
        # the only reliable result source when no checkpointer is configured —
        # ``get_state()`` requires one. We prefer the streamed messages and fall
        # back to ``get_state()`` only when the stream surfaced no message
        # updates (a checkpointer IS configured, or a custom stream shape).
        streamed_messages: list = []
        for item in stream_result:
            # Never let a display-bridge error abort graph consumption: the
            # graph has already run the node (tool execution included); we must
            # keep draining so the final answer still accumulates and streams.
            try:
                bridge.process_stream_item(item)
            except Exception:
                logger.debug("deepagents stream bridge failed on item", exc_info=True)
            mode, payload = _split_stream_item(item)
            if mode == "updates" and isinstance(payload, dict):
                for node_update in payload.values():
                    if isinstance(node_update, dict):
                        msgs = node_update.get("messages") or []
                        streamed_messages.extend(m for m in msgs if m is not None)

        # Emit any text the reasoning-leak filter held back at stream end.
        bridge.flush_visible()

        # After streaming, reconstruct the FULL conversation for the result.
        #
        # The native runtime (turn_context: ``list(conversation_history) +
        # [user] + new``) and the sync path (``agent.invoke`` returns the whole
        # graph state) both return the full conversation, and consumers rely on
        # that contract (the CLI does ``self.conversation_history =
        # result['messages']``; the gateway does ``session['history'] =
        # result['messages']``).
        #
        # ``streamed_messages`` is graph-dependent: in ``updates`` mode the real
        # deep-agents graph re-surfaces the INPUT messages (full history) ahead
        # of this turn's new output, while a pure-delta stream yields only the
        # new node output. Discriminate with a signal the graph never produces
        # as node output: a ``HumanMessage`` in ``streamed_messages`` can only
        # be the echoed input, so its presence means the stream already carries
        # the full conversation. Otherwise prepend the input to restore it.
        def _reconstruct(streamed):
            if any(isinstance(m, HumanMessage) for m in streamed):
                return list(streamed)
            return list(messages) + list(streamed)

        try:
            raw = streamed_messages
            if not raw:
                # No streamed updates: a checkpointer's get_state already holds
                # the full accumulated state, so it needs no input prepend.
                try:
                    raw = list(agent.get_state(config).values.get("messages", []))
                except Exception:
                    raw = list(messages)
            else:
                raw = _reconstruct(streamed_messages)
            r = _parse_langgraph_result({"messages": raw})
            self._maybe_flush_langfuse()
            return r
        except Exception:
            if streamed_messages:
                try:
                    r = _parse_langgraph_result(
                        {"messages": _reconstruct(streamed_messages)}
                    )
                    self._maybe_flush_langfuse()
                    return r
                except Exception:
                    pass
            r = _parse_error_result(Exception("Failed to get streaming final result"))
            self._maybe_flush_langfuse()
            return r

    def chat(self, message: str) -> str:
        """Simple chat interface — returns final response string."""
        result = self.run_conversation(message)
        return result.get("final_response", "")

    # ------------------------------------------------------------------
    # Observability — Langfuse tracing for deep agents.
    #
    # The deep agents runtime doesn't use run_agent.py or
    # conversation_loop.py, so the Hermes plugin hooks on `pre_api_request` /
    # `post_api_request` never fire here. Tracing instead rides LangChain's
    # native callback system: the Langfuse ``CallbackHandler`` is built from
    # the captured credentials and injected into ``config['callbacks']`` in
    # ``run_conversation``, capturing generations and tool spans directly from
    # LangGraph. This is the Langfuse v2-compatible path (the bundled
    # langfuse plugin targets the v3 SDK API, which is not installed).
    # ------------------------------------------------------------------

    def _maybe_flush_langfuse(self):
        """Flush the Langfuse client if tracing is active.

        v3 flushes on the ``Langfuse`` client, not the ``CallbackHandler``
        (which has no ``flush``). ``_langfuse_client`` is set alongside the
        handler in :meth:`_get_langfuse_handler`.
        """
        try:
            client = getattr(self, "_langfuse_client", None)
            if client is not None and hasattr(client, "flush"):
                client.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Compatibility forwarders (matching run_agent.AIAgent attrs)
    # ------------------------------------------------------------------

    @property
    def model(self):
        return ""  # LangGraph tracks internally

    @property
    def iteration_budget(self):
        return self

    @property
    def valid_tool_names(self):
        # Compute from tools
        try:
            from model_tools import get_tool_definitions

            defs = get_tool_definitions()
            return [d.get("name", "") for d in defs]
        except Exception:
            return []

    @property
    def tools(self):
        return []  # LangGraph manages internally

    @property
    def quiet_mode(self):
        return self._quiet_mode

    @property
    def platform(self):
        return self._platform

    @property
    def skip_memory(self):
        return self._skip_memory

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def has_checkpointer(self):
        """Return True if a LangGraph checkpointer is active."""
        return self._checkpointer is not None

    @property
    def has_store(self):
        """Return True if a LangGraph store is active."""
        return self._store is not None

    @property
    def has_langsmith_tracing(self):
        """Return True if LangSmith tracing is configured."""
        # Check both direct and callback-set values
        direct = self._ls_api_key or ""
        cb_key = self._get_cap("langsmith_api_key") or ""
        return bool(direct) or bool(cb_key) or bool(os.environ.get("LANGSMITH_API_KEY"))

    @property
    def has_langfuse_tracing(self):
        """Return True if Langfuse tracing is active (a handler was built)."""
        return self._get_langfuse_handler() is not None

    def get_tracing_config(self) -> dict:
        """Return the current tracing / observability config.

        Returns a dict of all tracing-related settings so gateway / external
        callers can inspect the active tracing configuration.
        """
        # read any gateway-set tracing values before returning
        ls_project = self._get_cap("langsmith_project") or self._ls_project or "hermes"
        ls_tags = self._get_cap("langsmith_tags") or self._ls_tags or ["hermes"]
        ls_api = self._get_cap("langsmith_api_key") or self._ls_api_key
        if isinstance(ls_tags, str):
            ls_tags = [ls_tags]

        return {
            "checkpointer": self._langgraph_checkpointer,
            "store": self._langgraph_store,
            "debug": self._debug is True,
            "langsmith_project": ls_project,
            "langsmith_tags": list(ls_tags),
            "langsmith_enabled": bool(ls_api),
            "langfuse_enabled": self.has_langfuse_tracing,
        }

    # ------------------------------------------------------------------
    # Memory delegation (pass-through)
    # ------------------------------------------------------------------

    def get_memory_context(self):
        return None

    def save_memory(self, messages):
        pass

    def interrupt(self):
        pass  # LangGraph doesn't have a direct interrupt
