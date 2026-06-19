"""E2E integration tests for the DeepAgents runtime path.

Tests the full call chain from the agent facade through DeepAgents components
with minimal mocking — just enough to make internal function calls succeed.

Test categories:
  - TestRunConversation: DeepAgentsAIAgent.run_conversation end-to-end
  - TestStreamingEndToEnd: Full streaming event routing
  - TestToolDispatcherEndToEnd: Tool execution integration
  - TestMessageRoundTrip: Hermes ↔ LangChain message conversion
  - TestDeepAgentsAIAgentIntegration: Full facade with agent attributes
"""

import json
import os
import sys
import types
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Reusable named tuple for ToolResult used in tests
ToolResult = namedtuple("ToolResult", ["value", "image_url", "is_error"])


def _ensure_deepagents_mocks():
    """Register minimal deepagents SDK mocks in sys.modules so the
    deep_agents_runtime module can import its SDK constants."""
    if "deepagents" not in sys.modules:
        mock_mod = MagicMock()
        mock_mod.ToolResult = ToolResult
        mock_mod.Thread = MagicMock()
        mock_mod.Runner = MagicMock()
        sys.modules["deepagents"] = mock_mod

    if "deepagents.runtime" not in sys.modules:
        rt_mod = types.ModuleType("deepagents.runtime")
        rt_mod.ToolResult = ToolResult
        rt_mod.Thread = MagicMock()
        rt_mod.Runner = MagicMock()
        sys.modules["deepagents.runtime"] = rt_mod


def _mock_langchain_for_tests():
    """Ensure langchain_core messages are importable for tests that
    use them directly (convert tests)."""
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        return {
            "AIMessage": AIMessage,
            "HumanMessage": HumanMessage,
            "SystemMessage": SystemMessage,
            "ToolMessage": ToolMessage,
        }
    except ImportError:
        return None


def _make_langchain_message(msg_type, **kwargs):
    """Create a LangChain message of the given type."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    content = kwargs.get("content", "")
    additional_kwargs = kwargs.get("additional_kwargs", {})
    if msg_type == "system":
        return SystemMessage(content=content)
    elif msg_type == "user":
        return HumanMessage(content=content)
    elif msg_type == "assistant":
        msg = AIMessage(content=content)
        if kwargs.get("tool_calls"):
            msg.tool_calls = kwargs["tool_calls"]
        if kwargs.get("reasoning"):
            msg.reasoning = kwargs["reasoning"]
        if kwargs.get("additional_kwargs"):
            msg.additional_kwargs = kwargs["additional_kwargs"]
        return msg
    elif msg_type == "tool":
        return ToolMessage(
            content=content, tool_call_id=kwargs.get("tool_call_id", "call_1")
        )
    else:
        raise ValueError(f"Unknown msg_type: {msg_type}")


def _make_hermes_message(role, content="", tool_calls=None, tool_call_id=None):
    """Create a Hermes-format message dict."""
    msg = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


# ---------------------------------------------------------------------------
# Autouse fixture: ensure deepagents mocks for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_deepagents_modules():
    """Ensure a clean deepagents mock before each test."""
    _ensure_deepagents_mocks()
    # Remove deep_agents_runtime from cache so functions re-evaluate imports
    for mod_name in list(sys.modules.keys()):
        if mod_name == "agent.deep_agents_runtime":
            del sys.modules[mod_name]
        elif mod_name.startswith("agent.deep_agents_"):
            del sys.modules[mod_name]
    yield
    # Cleanup: remove our mocks so other tests aren't affected
    sys.modules.pop("deepagents", None)
    sys.modules.pop("deepagents.runtime", None)


# ---------------------------------------------------------------------------
# TestRunConversation — DeepAgentsAIAgent facade integration
# ---------------------------------------------------------------------------


class TestRunConversation:
    """E2E: DeepAgentsAIAgent.run_conversation calls through the full stack."""

    def _make_sync_agent(self, **overrides):
        """Build an agent configured to take the sync (non-streaming) path."""
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
            _callbacks={},
            _agent=MagicMock(),
        )
        defaults.update(overrides)
        agent = object.__new__(DeepAgentsAIAgent)
        for k, v in defaults.items():
            object.__setattr__(agent, k, v)
        return agent

    def test_run_conversation_sync_path(self):
        """Sync run_conversation calls self._agent.invoke and returns parsed result."""
        agent = self._make_sync_agent(_max_iterations=30)
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="Hello!")],
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            result = agent.run_conversation(user_message="Hi")

        assert result["final_response"] == "Hello!"
        assert result["completed"] is True
        agent._agent.invoke.assert_called_once()
        call_kwargs = agent._agent.invoke.call_args.kwargs
        assert call_kwargs["config"]["recursion_limit"] == 30

    def test_run_conversation_sync_error_handling(self):
        """Errors during agent.invoke are caught and returned as error dict."""
        agent = self._make_sync_agent()
        agent._agent.invoke.side_effect = RuntimeError("model unavailable")

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            result = agent.run_conversation(user_message="test")

        assert result["failed"] is True
        assert result["completed"] is False
        assert "model unavailable" in result["final_response"]

    def test_run_conversation_with_conversation_history(self):
        """run_conversation appends history messages to the user message."""
        agent = self._make_sync_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="continuing...")],
        }

        history = [
            _make_hermes_message("system", "You are helpful."),
            _make_hermes_message("user", "first question"),
            _make_hermes_message("assistant", "first answer"),
        ]

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(
                user_message="second question",
                conversation_history=history,
            )

        call_args = agent._agent.invoke.call_args
        messages_arg = call_args[0][0]["messages"]  # first positional arg is state dict
        assert len(messages_arg) == 4  # 3 history + 1 new user message

    def test_run_conversation_system_message_prepended(self):
        """System message is prepended to conversation history."""
        agent = self._make_sync_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="ok")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(
                user_message="hello",
                system_message="You are a cat.",
                conversation_history=[],
            )

        call_args = agent._agent.invoke.call_args
        messages_arg = call_args[0][0]["messages"]
        from langchain_core.messages import SystemMessage, HumanMessage

        assert isinstance(messages_arg[0], SystemMessage)
        assert messages_arg[0].content == "You are a cat."
        assert isinstance(messages_arg[1], HumanMessage)
        assert messages_arg[1].content == "hello"

    def test_run_conversation_chat_returns_final_response(self):
        """chat() is a thin wrapper that returns final_response string."""
        agent = self._make_sync_agent()
        agent._agent.invoke.return_value = {
            "messages": [
                _make_langchain_message("assistant", content="the answer is 42")
            ]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            response = agent.chat("What is 6 * 7?")

        assert response == "the answer is 42"

    def test_run_conversation_passes_task_id_in_config(self):
        """task_id ends up in the config's 'configurable' section."""
        agent = self._make_sync_agent(
            _session_id="main-sess",
            provider="",
            _api_key=None,
            _base_url=None,
            _max_iterations=30,
        )
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="ok")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(
                user_message="test",
                task_id="unique-task-id-123",
            )

        cfg = agent._agent.invoke.call_args.kwargs["config"]
        assert cfg["configurable"]["task_id"] == "unique-task-id-123"
        assert cfg["configurable"]["session_id"] == "main-sess"

    def test_run_conversation_session_id_fallback(self):
        """When session_id is empty, it defaults to 'default'."""
        agent = self._make_sync_agent()
        object.__setattr__(agent, "_session_id", "")
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="ok")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(user_message="hi")

        cfg = agent._agent.invoke.call_args.kwargs["config"]
        assert cfg["configurable"]["session_id"] == "default"


# ---------------------------------------------------------------------------
# TestStreamingEndToEnd — streaming event routing
# ---------------------------------------------------------------------------


def _make_bare_deepagents_agent(**kwargs):
    """Create a DeepAgentsAIAgent via __new__ with the minimal attribute state
    the non-LangGraph code paths read (no graph compiled)."""
    from agent.deep_agents_runtime import DeepAgentsAIAgent

    agent = object.__new__(DeepAgentsAIAgent)
    object.__setattr__(agent, "mode", "deepagents")
    object.__setattr__(agent, "_quiet_mode", kwargs.get("quiet_mode", False))
    object.__setattr__(agent, "_skip_memory", kwargs.get("skip_memory", True))
    object.__setattr__(agent, "_platform", kwargs.get("platform"))
    object.__setattr__(agent, "_session_id", kwargs.get("session_id", "test"))
    object.__setattr__(agent, "_max_iterations", kwargs.get("max_iterations", 90))
    object.__setattr__(agent, "provider", kwargs.get("provider", ""))
    object.__setattr__(agent, "_model_raw", kwargs.get("model", "test-model"))
    object.__setattr__(agent, "_api_key", kwargs.get("api_key"))
    object.__setattr__(agent, "_base_url", kwargs.get("base_url"))
    object.__setattr__(agent, "_callbacks", kwargs.get("_callbacks", {}))
    return agent


class TestDeepAgentsSystemPrompt:
    """The deepagents runtime must run with the *real* Hermes system prompt
    (identity + tool-use enforcement + skills guidance), not a trivial
    'You are a helpful AI assistant.' fallback — otherwise the model describes
    tools in prose / types tool names as text instead of calling them."""

    def test_builds_full_hermes_prompt_not_trivial_fallback(self):
        agent = _make_bare_deepagents_agent(model="qwen", provider="custom")
        captured = {}

        def _fake_build(view, system_message=None):
            captured["view"] = view
            return "RICH HERMES PROMPT WITH ENFORCEMENT"

        with patch("agent.system_prompt.build_system_prompt", _fake_build):
            sp = agent._build_hermes_system_prompt()

        assert sp == "RICH HERMES PROMPT WITH ENFORCEMENT"
        assert sp != "You are a helpful AI assistant."

    def test_prompt_view_exposes_real_config(self):
        """The view handed to build_system_prompt must carry the impl's real
        model/provider/tool names so tool-aware guidance is injected."""
        agent = _make_bare_deepagents_agent(model="qwen-x", provider="custom")
        captured = {}

        def _fake_build(view, system_message=None):
            captured["view"] = view
            return "ok"

        with patch("agent.system_prompt.build_system_prompt", _fake_build), patch.object(
            type(agent), "valid_tool_names", new=property(lambda self: ["skills_list", "ls"])
        ):
            agent._build_hermes_system_prompt()

        view = captured["view"]
        assert view.model == "qwen-x"
        assert view.provider == "custom"
        assert "skills_list" in view.valid_tool_names
        # Context-file loading needs a file subsystem the deepagents impl lacks,
        # so the prompt path must skip it (identity/SOUL still loads).
        assert view.skip_context_files is True

    def test_falls_back_to_identity_not_trivial_on_error(self):
        """If prompt assembly raises, fall back to the real agent identity —
        never the bare 'helpful AI assistant' string that strips enforcement."""
        agent = _make_bare_deepagents_agent()
        with patch(
            "agent.system_prompt.build_system_prompt",
            side_effect=RuntimeError("boom"),
        ):
            sp = agent._build_hermes_system_prompt()
        assert sp
        assert sp != "You are a helpful AI assistant."


class TestStreamingEndToEnd:
    """E2E: streaming path — bridge processes events and callbacks fire."""

    def _make_deepagents_agent(self, **kwargs):
        """Create a DeepAgentsAIAgent via __new__ with a mock _agent."""
        return _make_bare_deepagents_agent(**kwargs)

    def test_streaming_path_fires_callbacks(self):
        """When callbacks are set, run_conversation enters _run_streamed and
        token chunks (messages mode) drive stream_delta + step callbacks while
        node updates carry the authoritative final messages."""
        from langchain_core.messages import AIMessageChunk

        deltas = []
        steps = []

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()

        final_ai = _make_langchain_message("assistant", content="final answer")

        def fake_stream(**kwargs):
            # messages mode: token-level chunks → live delta streaming
            yield ("messages", (AIMessageChunk(content="final "), {}))
            yield ("messages", (AIMessageChunk(content="answer"), {}))
            # updates mode: completed node output → authoritative messages
            yield ("updates", {"model": {"messages": [final_ai]}})

        agent._agent.stream.return_value = fake_stream()

        # Set up callbacks via the _CAPTURED_NAMES mechanism
        agent.stream_delta_callback = lambda x: deltas.append(x)
        agent.step_callback = lambda n, s: steps.append((n, s))

        response = agent.run_conversation(user_message="hello")

        assert response["final_response"] == "final answer"
        assert deltas == ["final ", "answer"]
        assert (1, []) in steps  # step callback fires on each token chunk
        # Token streaming uses the multi-mode stream, not a single "updates" mode.
        assert agent._agent.stream.call_args.kwargs["stream_mode"] == [
            "updates",
            "messages",
        ]

    def test_streaming_path_fires_tool_callbacks(self):
        """Tool calls in updates mode drive tool_start/tool_complete like native."""
        from langchain_core.messages import AIMessage, ToolMessage

        starts, completes = [], []

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()

        ai = AIMessage(content="checking")
        ai.tool_calls = [{"name": "read_file", "args": {"path": "x"}, "id": "c1"}]
        tm = ToolMessage(content="contents", tool_call_id="c1", name="read_file")
        final_ai = _make_langchain_message("assistant", content="done")

        def fake_stream(**kwargs):
            yield ("updates", {"model": {"messages": [ai]}})
            yield ("updates", {"tools": {"messages": [tm]}})
            yield ("updates", {"model": {"messages": [final_ai]}})

        agent._agent.stream.return_value = fake_stream()
        agent.tool_start_callback = lambda i, n, a: starts.append((i, n, a))
        agent.tool_complete_callback = lambda i, n, a, r: completes.append((i, n, a, r))

        response = agent.run_conversation(user_message="read it")

        assert response["final_response"] == "done"
        assert starts == [("c1", "read_file", {"path": "x"})]
        assert completes == [("c1", "read_file", {"path": "x"}, "contents")]

    def test_streaming_forwards_stream_callback_param(self):
        """The TUI passes stream_callback to run_conversation (not as an attr);
        it must reach the delta sink."""
        from langchain_core.messages import AIMessageChunk

        deltas = []
        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()
        final_ai = _make_langchain_message("assistant", content="hi there")

        def fake_stream(**kwargs):
            yield ("messages", (AIMessageChunk(content="hi "), {}))
            yield ("messages", (AIMessageChunk(content="there"), {}))
            yield ("updates", {"model": {"messages": [final_ai]}})

        agent._agent.stream.return_value = fake_stream()

        response = agent.run_conversation(
            user_message="hello",
            stream_callback=lambda x: deltas.append(x),
        )
        assert response["final_response"] == "hi there"
        assert deltas == ["hi ", "there"]

    def test_streaming_result_messages_include_history_and_user(self):
        """The streaming path must return the FULL conversation in
        result['messages'] — conversation_history + the user turn + this turn's
        new messages — matching the native runtime (turn_context builds
        ``list(conversation_history) + [user] + new``) and the sync path
        (``agent.invoke`` returns the whole state).

        The gateway does ``session['history'] = result['messages']``
        (tui_gateway/server.py), so a streaming result that carries only the
        node-output deltas silently truncates session history every turn,
        re-feeding malformed history and duplicating tool-call chrome on screen.
        """
        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()

        new_ai = _make_langchain_message("assistant", content="second answer")

        def fake_stream(**kwargs):
            # updates mode yields only the NEW node output, not the input state.
            yield ("updates", {"model": {"messages": [new_ai]}})

        agent._agent.stream.return_value = fake_stream()
        agent.stream_delta_callback = lambda x: None

        history = [
            _make_hermes_message("user", "first question"),
            _make_hermes_message("assistant", "first answer"),
        ]

        result = agent.run_conversation(
            user_message="second question",
            conversation_history=history,
        )

        roles_and_text = [(m["role"], m.get("content")) for m in result["messages"]]
        # Full conversation: prior history + this turn's user + new assistant reply.
        assert ("user", "first question") in roles_and_text
        assert ("assistant", "first answer") in roles_and_text
        assert ("user", "second question") in roles_and_text
        assert ("assistant", "second answer") in roles_and_text
        # No injected system message leaks into the returned history (the gateway
        # supplies system separately; native history carries none here).
        assert all(role != "system" for role, _ in roles_and_text)

    def test_streaming_echoed_input_not_duplicated_in_result(self):
        """When the real graph re-surfaces the input history in ``updates``
        mode (a ``HumanMessage`` appears in the stream), result['messages'] must
        be the streamed list as-is — NOT input + streamed, which would duplicate
        the whole history and, fed back via session/history, snowball each turn.
        """
        from langchain_core.messages import HumanMessage

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()

        # The graph echoes the input (user turn) plus this turn's new reply.
        echoed_user = HumanMessage(content="second question")
        new_ai = _make_langchain_message("assistant", content="second answer")

        def fake_stream(**kwargs):
            yield ("updates", {"model": {"messages": [echoed_user, new_ai]}})

        agent._agent.stream.return_value = fake_stream()
        agent.stream_delta_callback = lambda x: None

        history = [
            _make_hermes_message("user", "first question"),
            _make_hermes_message("assistant", "first answer"),
        ]

        result = agent.run_conversation(
            user_message="second question",
            conversation_history=history,
        )

        users = [m for m in result["messages"] if m["role"] == "user"]
        # Exactly one "second question" — the echo, not echo + prepended input.
        assert sum(1 for m in users if m["content"] == "second question") == 1
        assert ("assistant", "second answer") in [
            (m["role"], m.get("content")) for m in result["messages"]
        ]

    def test_streaming_skips_historical_tool_calls_resurfaced_from_input(self):
        """Tools already present in the input history must NOT re-fire
        tool_start/tool_complete when the graph re-surfaces them in ``updates``
        mode — otherwise every prior turn's tool trail reprints each turn.
        Only this turn's NEW tool should drive the callbacks.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        starts, completes = [], []

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()

        # Prior turn's tool call (id "old1") lives in the conversation history.
        history = [
            _make_hermes_message("user", "earlier"),
            _make_hermes_message(
                "assistant",
                "",
                tool_calls=[{"name": "web_search", "args": {"q": "a"}, "id": "old1"}],
            ),
            _make_hermes_message("tool", "old result", tool_call_id="old1"),
            _make_hermes_message("assistant", "earlier answer"),
        ]

        # The graph re-surfaces the historical tool messages, then runs ONE new
        # tool ("new1") and replies.
        old_ai = AIMessage(content="")
        old_ai.tool_calls = [{"name": "web_search", "args": {"q": "a"}, "id": "old1"}]
        old_tm = ToolMessage(content="old result", tool_call_id="old1", name="web_search")
        new_ai = AIMessage(content="")
        new_ai.tool_calls = [{"name": "web_search", "args": {"q": "b"}, "id": "new1"}]
        new_tm = ToolMessage(content="new result", tool_call_id="new1", name="web_search")
        final = _make_langchain_message("assistant", content="done")

        def fake_stream(**kwargs):
            yield ("updates", {"model": {"messages": [HumanMessage(content="again"), old_ai]}})
            yield ("updates", {"tools": {"messages": [old_tm]}})
            yield ("updates", {"model": {"messages": [new_ai]}})
            yield ("updates", {"tools": {"messages": [new_tm]}})
            yield ("updates", {"model": {"messages": [final]}})

        agent._agent.stream.return_value = fake_stream()
        agent.tool_start_callback = lambda i, n, a: starts.append((i, n, a))
        agent.tool_complete_callback = lambda i, n, a, r: completes.append((i, n, a, r))

        agent.run_conversation(user_message="again", conversation_history=history)

        # Only the new tool fires; the re-surfaced historical "old1" is skipped.
        assert starts == [("new1", "web_search", {"q": "b"})]
        assert completes == [("new1", "web_search", {"q": "b"}, "new result")]

    # --- text-embedded tool-call recovery (vLLM / quantized model drift) -----
    # Recover tool calls a server returned as ``<model_tool_calls>`` text
    # (vLLM without a matching --tool-call-parser / quantized model drift)
    # instead of structured tool_calls.
    _XML = (
        "<model_tool_calls>\n"
        '<tool name="web_search">\n'
        "<query>\nEspaña vs Cabo Verde resultado</query>\n"
        "</tool>\n"
        "</model_tool_calls>"
    )

    def test_parse_observed_format(self):
        from agent.deep_agents_runtime import _parse_text_tool_calls

        calls = _parse_text_tool_calls(self._XML)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"
        assert calls[0]["args"] == {"query": "España vs Cabo Verde resultado"}
        assert calls[0]["id"]

    def test_parse_multiple_tools_and_args(self):
        from agent.deep_agents_runtime import _parse_text_tool_calls

        xml = (
            "<model_tool_calls>"
            '<tool name="web_search"><query>a</query><count>3</count></tool>'
            '<tool name="web_extract"><url>http://x</url></tool>'
            "</model_tool_calls>"
        )
        calls = _parse_text_tool_calls(xml)
        assert [c["name"] for c in calls] == ["web_search", "web_extract"]
        assert calls[0]["args"] == {"query": "a", "count": "3"}
        assert calls[1]["args"] == {"url": "http://x"}

    def test_parse_plain_text_returns_empty(self):
        from agent.deep_agents_runtime import _parse_text_tool_calls

        assert _parse_text_tool_calls("just a normal answer, no tools") == []

    def test_parse_unclosed_wrapper(self):
        from agent.deep_agents_runtime import _parse_text_tool_calls

        xml = '<model_tool_calls><tool name="web_search"><query>x</query></tool>'
        calls = _parse_text_tool_calls(xml)
        assert calls and calls[0]["name"] == "web_search"

    def test_strip_removes_block(self):
        from agent.deep_agents_runtime import _strip_text_tool_calls

        assert _strip_text_tool_calls(f"before {self._XML} after") == "before  after"

    def test_repair_chat_result_populates_tool_calls(self):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        from agent.deep_agents_runtime import _repair_chat_result

        msg = AIMessage(content=f"Voy a buscar.\n{self._XML}")
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        _repair_chat_result(result)
        assert msg.tool_calls and msg.tool_calls[0]["name"] == "web_search"
        assert "<model_tool_calls>" not in msg.content
        assert msg.content == "Voy a buscar."

    def test_repair_chat_result_noop_when_already_structured(self):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        from agent.deep_agents_runtime import _repair_chat_result

        msg = AIMessage(content="hi")
        msg.tool_calls = [{"name": "x", "args": {}, "id": "real1"}]
        result = ChatResult(generations=[ChatGeneration(message=msg)])
        _repair_chat_result(result)
        assert msg.tool_calls == [{"name": "x", "args": {}, "id": "real1"}]
        assert msg.content == "hi"

    def test_repair_stream_hides_xml_and_synthesizes_tool_call(self):
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk
        from agent.deep_agents_runtime import _repair_stream

        # Stream the visible preamble then the XML, token by token.
        tokens = ["Voy ", "a buscar.\n", *list(self._XML)]
        src = (
            ChatGenerationChunk(message=AIMessageChunk(content=t)) for t in tokens
        )
        out = list(_repair_stream(src))

        merged = None
        for ch in out:
            merged = ch.message if merged is None else merged + ch.message

        assert "<model_tool_calls>" not in merged.content
        assert merged.content.strip() == "Voy a buscar."
        assert merged.tool_calls and merged.tool_calls[0]["name"] == "web_search"
        assert merged.tool_calls[0]["args"] == {
            "query": "España vs Cabo Verde resultado"
        }

    def test_repair_stream_passes_through_structured_tool_chunks(self):
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk
        from agent.deep_agents_runtime import _repair_stream

        # Server already returned a real structured tool chunk — don't touch it.
        real = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "web_search", "args": '{"q":"x"}', "id": "r1", "index": 0}
                ],
            )
        )
        out = list(_repair_stream(iter([real])))
        assert len(out) == 1 and out[0] is real

    def test_streaming_path_no_callbacks_runs_sync(self):
        """Without callbacks, run_conversation degrades to sync path."""
        agent = self._make_deepagents_agent(_callbacks={})
        agent._agent = MagicMock()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="answer")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            response = agent.run_conversation(user_message="hello")

        assert response["final_response"] == "answer"
        agent._agent.invoke.assert_called_once()
        agent._agent.stream.assert_not_called()

    def test_streaming_bridge_event_routing(self):
        """Bridge routes messages-mode chunks to stream_delta and updates-mode
        tool calls to tool_start/tool_progress."""
        from langchain_core.messages import AIMessage, AIMessageChunk
        from agent.deep_agents_runtime import _HermesStreamingBridge

        deltas = []
        starts = []
        progress = []

        bridge = _HermesStreamingBridge(
            agent=MagicMock(),
            stream_delta=lambda x: deltas.append(x),
            tool_start=lambda i, n, a: starts.append((i, n, a)),
            tool_progress=lambda a, *rest: progress.append(a),
        )

        bridge.process_stream_item(("messages", (AIMessageChunk(content="stream "), {})))
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "read_file", "args": {"path": "x.txt"}, "id": "c1"}]
        bridge.process_stream_item(("updates", {"model": {"messages": [ai]}}))

        assert "stream " in deltas
        assert starts == [("c1", "read_file", {"path": "x.txt"})]
        assert progress == ["tool.started"]

    def test_streaming_bridge_handles_non_items_gracefully(self):
        """Malformed stream items don't crash the bridge."""
        from agent.deep_agents_runtime import _HermesStreamingBridge

        bridge = _HermesStreamingBridge(agent=MagicMock())

        bridge.process_stream_item("string event")
        bridge.process_stream_item(None)
        bridge.process_stream_item(123)
        bridge.process_stream_item(["list", "of", "items"])
        bridge.process_stream_item(("messages", None))
        bridge.process_stream_item(("updates", "not-a-dict"))

        # Should not raise

    def test_streaming_bridge_empty_stream(self):
        """An empty stream yields no callbacks but _run_streamed doesn't crash."""
        from agent.deep_agents_runtime import _HermesStreamingBridge

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()
        agent._agent.stream.return_value = iter([])
        agent._agent.get_state.return_value = type(
            "State",
            (),
            {
                "values": {
                    "messages": [_make_langchain_message("assistant", content="")]
                }
            },
        )
        object.__setattr__(agent, "_quiet_mode", False)
        agent.stream_delta_callback = lambda x: None

        response = agent.run_conversation(user_message="hello")

        assert response["final_response"] == ""
        assert response["completed"] is False


# ---------------------------------------------------------------------------
# TestToolDispatcherEndToEnd — tool execution integration
# ---------------------------------------------------------------------------


class TestToolDispatcherEndToEnd:
    """E2E: tool dispatcher integration with agent context."""

    def test_hermes_tool_adapter_creates_structured_tool(self):
        """_HermesToolAdapter produces a LangChain StructuredTool."""
        from langchain_core.tools import StructuredTool
        from agent.deep_agents_runtime import _HermesToolAdapter

        entry = types.SimpleNamespace(
            name="wc",
            toolset="search",
            schema={
                "name": "wc",
                "description": "word count",
                "parameters": {"type": "object"},
            },
            description="wc tool",
        )
        adapter = _HermesToolAdapter(entry)
        assert isinstance(adapter.langchain_tool, StructuredTool)
        assert adapter.name == "wc"
        assert adapter.toolset == "search"

    def test_tool_adapter_invokes_handle_function_call(self):
        """Calling the adapter's tool executes handle_function_call."""
        with patch("model_tools.handle_function_call") as mock_hfc:
            mock_hfc.return_value = json.dumps({"count": 42})

            from agent.deep_agents_runtime import _HermesToolAdapter

            entry = types.SimpleNamespace(
                name="test_tool",
                toolset="test",
                schema={
                    "name": "test_tool",
                    "description": "test",
                    "parameters": {"type": "object"},
                },
                description="test",
            )
            adapter = _HermesToolAdapter(entry)

            result = adapter.langchain_tool.func(myarg="value")
            result_obj = json.loads(result)

            assert result_obj == {"count": 42}
            mock_hfc.assert_called_once_with(
                function_name="test_tool",
                function_args={"myarg": "value"},
            )

    def test_tool_adapter_wraps_errors_as_json(self):
        """Exceptions in handle_function_call are returned as JSON error."""
        with patch("model_tools.handle_function_call") as mock_hfc:
            mock_hfc.side_effect = FileNotFoundError("no such file")

            from agent.deep_agents_runtime import _HermesToolAdapter

            entry = types.SimpleNamespace(
                name="read_file",
                toolset="file",
                schema={
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
                description="read file",
            )
            adapter = _HermesToolAdapter(entry)

            result = adapter.langchain_tool.func(path="/nonexistent")
            result_obj = json.loads(result)

            assert "error" in result_obj
            assert "no such file" in result_obj["error"]

    def test_build_hermes_tools_creates_tools_from_definitions(self):
        """build_hermes_tools resolves via get_tool_definitions and returns
        a StructuredTool per definition."""
        from agent.deep_agents_runtime import build_hermes_tools
        import model_tools

        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "list files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ]
        with patch.object(
            model_tools, "get_tool_definitions", return_value=definitions
        ):
            tools = build_hermes_tools(
                enabled_toolsets=["terminal"], disabled_toolsets=[]
            )
        assert len(tools) == 1
        assert tools[0].name == "list_files"
        assert hasattr(tools[0], "func")  # StructuredTool has .func
        assert "path" in tools[0].args  # parameter schema forwarded

    def test_build_hermes_tools_includes_bridge_tools(self):
        """tool_search/tool_describe/tool_call have no registry entry yet must
        still be adapted (tool-search assembly emits them for big catalogs)."""
        from agent.deep_agents_runtime import build_hermes_tools
        import model_tools

        definitions = [
            {"type": "function", "function": {"name": "tool_search", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_call", "parameters": {}}},
        ]
        with patch.object(
            model_tools, "get_tool_definitions", return_value=definitions
        ):
            tools = build_hermes_tools(enabled_toolsets=["x"], disabled_toolsets=[])
        assert {t.name for t in tools} == {"tool_search", "tool_call"}

    def test_build_hermes_tools_handles_resolution_exception(self):
        """If get_tool_definitions raises, an empty list is returned."""
        from agent.deep_agents_runtime import build_hermes_tools
        import model_tools

        with patch.object(
            model_tools,
            "get_tool_definitions",
            side_effect=ImportError("registry gone"),
        ):
            tools = build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[])
        assert tools == []


class TestEnsureMcpDiscovery:
    """The deepagents runtime bakes tools once, so it must block until MCP
    discovery completes before building (unlike native, which re-snapshots)."""

    def _make_agent(self, timeout=42.0):
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = object.__new__(DeepAgentsAIAgent)
        object.__setattr__(agent, "_mcp_discovery_timeout", timeout)
        return agent

    def test_waits_for_background_discovery_when_running(self):
        import hermes_cli.mcp_startup as ms
        import tools.mcp_tool as mt

        agent = self._make_agent(timeout=55.0)
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"mcp_servers": {"nyxstrike": {}}},
        ), patch.object(ms, "_mcp_discovery_started", True), patch.object(
            ms, "_mcp_discovery_thread", MagicMock()
        ), patch.object(
            ms, "wait_for_mcp_discovery"
        ) as wait, patch.object(
            mt, "discover_mcp_tools"
        ) as disc:
            agent._ensure_mcp_discovery()
        # Joins the in-flight thread with our generous timeout; does NOT race
        # it with a second discovery pass.
        wait.assert_called_once_with(timeout=55.0)
        assert not disc.called

    def test_runs_synchronous_discovery_when_not_started(self):
        import hermes_cli.mcp_startup as ms
        import tools.mcp_tool as mt

        agent = self._make_agent()
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"mcp_servers": {"nyxstrike": {}}},
        ), patch.object(ms, "_mcp_discovery_started", False), patch.object(
            ms, "_mcp_discovery_thread", None
        ), patch.object(
            ms, "wait_for_mcp_discovery"
        ) as wait, patch.object(
            mt, "discover_mcp_tools"
        ) as disc:
            agent._ensure_mcp_discovery()
        disc.assert_called_once()
        assert not wait.called

    def test_noop_when_no_mcp_servers_configured(self):
        import hermes_cli.mcp_startup as ms
        import tools.mcp_tool as mt

        agent = self._make_agent()
        with patch(
            "hermes_cli.config.read_raw_config", return_value={"mcp_servers": {}}
        ), patch.object(ms, "wait_for_mcp_discovery") as wait, patch.object(
            mt, "discover_mcp_tools"
        ) as disc:
            agent._ensure_mcp_discovery()
        assert not wait.called and not disc.called

    def test_never_raises_on_discovery_failure(self):
        import tools.mcp_tool as mt

        agent = self._make_agent()
        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"mcp_servers": {"nyxstrike": {}}},
        ), patch("hermes_cli.mcp_startup._mcp_discovery_started", False), patch(
            "hermes_cli.mcp_startup._mcp_discovery_thread", None
        ), patch.object(
            mt, "discover_mcp_tools", side_effect=RuntimeError("boom")
        ):
            # Must not propagate — construction proceeds with available tools.
            agent._ensure_mcp_discovery()


class TestRebuildAgent:
    """The compiled LangGraph graph bakes its tool list, so picking up new MCP
    tools means recompiling and atomically swapping the graph in place."""

    def _make_agent(self):
        import threading
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = object.__new__(DeepAgentsAIAgent)
        object.__setattr__(agent, "_agent", MagicMock(name="old_graph"))
        object.__setattr__(agent, "_agent_lock", threading.Lock())
        object.__setattr__(
            agent, "_build_kwargs", {"model": "m", "enabled_toolsets": ["x"]}
        )
        return agent

    def test_rebuild_swaps_in_new_graph(self):
        agent = self._make_agent()
        old = agent._agent
        new_graph = MagicMock(name="new_graph")
        object.__setattr__(
            agent, "_build_langgraph_agent", MagicMock(return_value=new_graph)
        )

        assert agent.rebuild_agent() is True
        assert agent._agent is new_graph and agent._agent is not old
        agent._build_langgraph_agent.assert_called_once_with(**agent._build_kwargs)

    def test_rebuild_keeps_old_graph_on_failure(self):
        agent = self._make_agent()
        old = agent._agent
        object.__setattr__(
            agent,
            "_build_langgraph_agent",
            MagicMock(side_effect=RuntimeError("compile boom")),
        )

        assert agent.rebuild_agent() is False
        assert agent._agent is old  # unchanged — session stays usable

    def test_rebuild_without_build_kwargs_returns_false(self):
        agent = self._make_agent()
        object.__setattr__(agent, "_build_kwargs", None)
        object.__setattr__(
            agent, "_build_langgraph_agent", MagicMock(return_value=MagicMock())
        )

        assert agent.rebuild_agent() is False
        agent._build_langgraph_agent.assert_not_called()

    def test_current_agent_returns_live_graph(self):
        agent = self._make_agent()
        assert agent._current_agent() is agent._agent
        new_graph = MagicMock()
        object.__setattr__(agent, "_agent", new_graph)
        assert agent._current_agent() is new_graph


# ---------------------------------------------------------------------------
# TestMessageRoundTrip — Hermes ↔ LangChain conversion
# ---------------------------------------------------------------------------


class TestMessageRoundTrip:
    """E2E: Hermes message dict ↔ LangChain message round-trip."""

    def test_hermes_to_langchain_to_hermes_roundtrip_system_user(self):
        """System → User roundtrip preserves content and roles."""
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        original = [
            _make_hermes_message("system", "You are a cat."),
            _make_hermes_message("user", "Meow?"),
        ]
        lc = _convert_messages_to_langchain(original)
        back = _convert_langchain_to_hermes(lc)

        assert back == original

    def test_hermes_to_langchain_to_hermes_roundtrip_tool_call(self):
        """Assistant with tool_calls → LangChain AIMessage → back."""
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        tool_calls = [
            {"id": "call_abc", "function": {"name": "cmd", "arguments": '{"n": 1}'}}
        ]
        original = [
            _make_hermes_message("assistant", "checking", tool_calls=tool_calls),
            _make_hermes_message("tool", "result", tool_call_id="call_abc"),
        ]
        lc = _convert_messages_to_langchain(original)
        back = _convert_langchain_to_hermes(lc)

        assert back == original

    def test_full_sequence_roundtrip(self):
        """Full conversation sequence: system → user → assistant → tool → assistant."""
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        original = [
            _make_hermes_message("system", "Be brief."),
            _make_hermes_message("user", "What is 2+2?"),
            _make_hermes_message(
                "assistant",
                "4",
                tool_calls=[
                    {"id": "call_1", "function": {"name": "verify", "arguments": "{}"}},
                ],
            ),
            _make_hermes_message("tool", "verified", tool_call_id="call_1"),
            _make_hermes_message("assistant", "2+2=4"),
        ]

        lc = _convert_messages_to_langchain(original)
        back = _convert_langchain_to_hermes(lc)

        assert back == original
        roles = [m["role"] for m in back]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_langchain_to_hermes_none_input_raises(self):
        """_convert_langchain_to_hermes raises TypeError on None."""
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        with pytest.raises(TypeError):
            _convert_langchain_to_hermes(None)

    def test_hermes_to_langchain_empty_messages(self):
        """Empty list returns empty list."""
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert _convert_messages_to_langchain([]) == []

    def test_hermes_to_langchain_with_tool_calls_preserves_args(self):
        """Tool calls on assistant messages preserve id and args."""
        from agent.deep_agents_runtime import _convert_messages_to_langchain
        from langchain_core.messages import AIMessage

        hermes = [
            _make_hermes_message(
                "assistant",
                "let me check",
                tool_calls=[
                    {
                        "id": "tc-123",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "test query"}',
                        },
                    }
                ],
            ),
        ]
        lc = _convert_messages_to_langchain(hermes)
        ai = lc[0]
        assert isinstance(ai, AIMessage)
        assert len(ai.tool_calls) == 1
        assert ai.tool_calls[0]["id"] == "tc-123"
        assert ai.tool_calls[0]["function"]["name"] == "web_search"
        assert ai.tool_calls[0]["function"]["arguments"] == '{"q": "test query"}'

    def test_parse_langgraph_result_full_sequence(self):
        """Full conversation: tool_calls → AI message with text → parse."""
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai1 = _make_langchain_message(
            "assistant",
            content="checking tools",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
                {
                    "id": "call_2",
                    "function": {"name": "run_command", "arguments": "{}"},
                },
            ],
        )
        ai2 = _make_langchain_message(
            "assistant", content="All done, here's the answer."
        )
        ai2.reasoning = "reasoned thinking"

        result = _parse_langgraph_result({"messages": [ai1, ai2]})

        assert result["final_response"] == "All done, here's the answer."
        assert result["api_calls"] == 1  # only ai2 counts (it has no tool_calls)
        assert result["completed"] is True
        assert result["last_reasoning"] == "reasoned thinking"
        assert result["failed"] is False
        assert result["turn_exit_reason"] == "completed"

    def test_parse_error_result_full_shape(self):
        """Error result dict has all the keys that run_conversation expects."""
        from agent.deep_agents_runtime import _parse_error_result

        e = Exception("deepagents failed because X")
        result = _parse_error_result(e)

        expected_keys = {
            "final_response",
            "messages",
            "api_calls",
            "completed",
            "failed",
            "interrupted",
            "partial",
            "turn_exit_reason",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["failed"] is True
        assert "X" in result["final_response"]

    def test_parse_langgraph_result_multimodal_content(self):
        """Multimodal content (list of dicts) extracts text parts."""
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content=[
                {"type": "text", "text": "Here's the image analysis."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        )

        result = _parse_langgraph_result({"messages": [ai]})
        assert result["final_response"] == "Here's the image analysis."

    def test_inject_provider_env_isolation(self):
        """_inject_provider_env sets the right env vars, no cross-contamination."""
        from agent.deep_agents_runtime import _inject_provider_env

        old_api = os.environ.pop("OPENAI_API_KEY", None)
        old_base = os.environ.pop("OPENAI_API_BASE", None)
        try:
            _inject_provider_env("openai", "https://my-api.com", "sk-key")
            assert os.environ["OPENAI_API_KEY"] == "sk-key"
            assert os.environ["OPENAI_API_BASE"] == "https://my-api.com"
        finally:
            if old_api is not None:
                os.environ["OPENAI_API_KEY"] = old_api
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if old_base is not None:
                os.environ["OPENAI_API_BASE"] = old_base
            else:
                os.environ.pop("OPENAI_API_BASE", None)


# ---------------------------------------------------------------------------
# TestDeepAgentsAIAgentIntegration — full facade with agent attributes
# ---------------------------------------------------------------------------


class TestDeepAgentsAIAgentIntegration:
    """E2E: test the DeepAgentsAIAgent facade as the agent loop would use it."""

    def _build_agent(self, **overrides):
        """Build a DeepAgentsAIAgent with minimal state via __new__."""
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = object.__new__(DeepAgentsAIAgent)
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
            _callbacks={},
            _agent=MagicMock(),
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            object.__setattr__(agent, k, v)
        return agent

    def test_agent_has_correct_mode(self):
        agent = self._build_agent()
        assert agent.mode == "deepagents"

    def test_agent_compatibility_properties(self):
        """Properties that the agent loop reads are accessible."""
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = self._build_agent()
        assert isinstance(agent.iteration_budget, DeepAgentsAIAgent)  # returns self
        assert agent.model == ""
        assert agent.tools == []
        assert agent.quiet_mode is False
        assert isinstance(agent.skip_memory, bool)

    def test_callback_forwarding_via_setattr(self):
        """Setting callback attrs stores them in _callbacks dict."""
        agent = self._build_agent()

        def my_delta(text):
            pass

        agent.tool_progress_callback = my_delta
        assert agent._get_cap("tool_progress_callback") is my_delta

    def test_callback_get_cap_returns_default_missing(self):
        agent = self._build_agent()
        assert agent._get_cap("nonexistent_callback", "fallback") == "fallback"

    def test_callback_get_cap_returns_none_missing_with_no_default(self):
        agent = self._build_agent()
        assert agent._get_cap("missing") is None
        agent2 = self._build_agent()
        object.__setattr__(agent2, "_callbacks", {})
        assert agent2._get_cap("missing") is None

    def test_interrupt_is_noop(self):
        agent = self._build_agent()
        agent.interrupt()  # should not raise

    def test_get_memory_context_returns_none(self):
        agent = self._build_agent()
        assert agent.get_memory_context() is None

    def test_save_memory_is_noop(self):
        agent = self._build_agent()
        agent.save_memory([])  # should not raise

    def test_chat_calls_run_conversation(self):
        agent = self._build_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="chat response")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            resp = agent.chat("hello")

        assert resp == "chat response"
        assert agent._agent.invoke.called
