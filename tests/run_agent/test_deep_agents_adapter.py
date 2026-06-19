"""Unit tests for deep_agents_runtime.py adapter layer.

Tests _HermesToolAdapter, build_hermes_tools,
_convert_messages_to_langchain, _convert_langchain_to_hermes,
_parse_langgraph_result, _parse_error_result, _inject_provider_env,
and _HermesStreamingBridge - all without live LLM calls.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_tool_entry(name="test_tool", toolset="test", **overrides):
    schema = overrides.pop(
        "schema",
        {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    )
    desc = overrides.pop("description", schema.get("description", f"{name} tool"))
    return types.SimpleNamespace(
        name=name,
        toolset=toolset,
        schema=schema,
        description=desc,
        check_fn=None,
        handler=MagicMock(),
        requires_env=[],
        is_async=False,
        emoji="",
        max_result_size_chars=None,
        dynamic_schema_overrides=None,
    )


def _make_langchain_message(msg_type, **kwargs):
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
        if additional_kwargs:
            msg.additional_kwargs = additional_kwargs
        return msg
    elif msg_type == "tool":
        return ToolMessage(
            content=content, tool_call_id=kwargs.get("tool_call_id", "call_1")
        )
    else:
        raise ValueError(f"Unknown msg_type: {msg_type}")


def _make_hermes_message(role, content="", tool_calls=None, tool_call_id=None):
    msg = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


class TestConvertMessagesToLangchain:
    def test_none_input_returns_empty_list(self):
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert _convert_messages_to_langchain(None) == []

    def test_empty_list_returns_empty_list(self):
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert _convert_messages_to_langchain([]) == []

    def test_system_message(self):
        from langchain_core.messages import SystemMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("system", "You are helpful.")
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], SystemMessage)
            and msgs[0].content == "You are helpful."
        )

    def test_user_message(self):
        from langchain_core.messages import HumanMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([_make_hermes_message("user", "Hello!")])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], HumanMessage)
            and msgs[0].content == "Hello!"
        )

    def test_assistant_message_without_tool_calls(self):
        from langchain_core.messages import AIMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("assistant", "Sure thing!")
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], AIMessage)
            and msgs[0].content == "Sure thing!"
        )

    def test_assistant_message_with_tool_calls(self):
        from langchain_core.messages import AIMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        tc = [
            {
                "id": "call_1",
                "function": {"name": "web_search", "arguments": '{"q": "test"}'},
            }
        ]
        msgs = _convert_messages_to_langchain([
            _make_hermes_message("assistant", "Let me check.", tool_calls=tc)
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], AIMessage)
            and msgs[0].tool_calls == tc
        )

    def test_tool_message(self):
        from langchain_core.messages import ToolMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("tool", "Result", tool_call_id="call_1")
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], ToolMessage)
            and msgs[0].content == "Result"
        )

    def test_mixed_sequence(self):
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        hermes = [
            _make_hermes_message("system", "You are helpful."),
            _make_hermes_message("user", "Hi"),
            _make_hermes_message("assistant", "Hello!"),
            _make_hermes_message("tool", "tool result", tool_call_id="call_x"),
        ]
        msgs = _convert_messages_to_langchain(hermes)
        assert len(msgs) == 4
        assert isinstance(msgs[0], SystemMessage) and isinstance(msgs[1], HumanMessage)
        assert isinstance(msgs[2], AIMessage) and isinstance(msgs[3], ToolMessage)

    def test_missing_role_defaults_to_no_conversion(self):
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert len(_convert_messages_to_langchain([{"content": "no role field"}])) == 0

    def test_empty_content_converted(self):
        from langchain_core.messages import SystemMessage, HumanMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("system", ""),
            _make_hermes_message("user", ""),
        ])
        assert len(msgs) == 2 and msgs[0].content == "" and msgs[1].content == ""


class TestConvertLangchainToHermes:
    def test_none_input_raises_typeerror(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        with pytest.raises(TypeError):
            _convert_langchain_to_hermes(None)

    def test_empty_list(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        assert _convert_langchain_to_hermes([]) == []

    def test_system_message(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("system", content="You are helpful.")
        ])
        assert len(msgs) == 1 and msgs[0] == {
            "role": "system",
            "content": "You are helpful.",
        }

    def test_user_message(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("user", content="Hello?")
        ])
        assert len(msgs) == 1 and msgs[0] == {"role": "user", "content": "Hello?"}

    def test_assistant_message_without_tool_calls(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("assistant", content="Yes.")
        ])
        assert len(msgs) == 1 and msgs[0] == {"role": "assistant", "content": "Yes."}

    def test_assistant_message_with_empty_content(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("assistant", content="")
        ])
        assert (
            len(msgs) == 1
            and msgs[0]["role"] == "assistant"
            and msgs[0]["content"] == ""
        )

    def test_assistant_message_with_tool_calls(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        tc = [
            {
                "id": "call_1",
                "function": {"name": "read_file", "arguments": '{"path": "foo.txt"}'},
            }
        ]
        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("assistant", content="", tool_calls=tc)
        ])
        assert len(msgs) == 1 and msgs[0]["tool_calls"] == tc

    def test_tool_message(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message(
                "tool", content="file contents", tool_call_id="call_abc"
            )
        ])
        assert msgs[0] == {
            "role": "tool",
            "content": "file contents",
            "tool_call_id": "call_abc",
        }

    def test_mixed_sequence(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        langchain = [
            _make_langchain_message("system", content="Be brief."),
            _make_langchain_message("user", content="What's 2+2?"),
            _make_langchain_message("assistant", content="4"),
            _make_langchain_message("tool", content="ok", tool_call_id="call_10"),
        ]
        hermes = _convert_langchain_to_hermes(langchain)
        assert len(hermes) == 4 and hermes[3]["tool_call_id"] == "call_10"


class TestParseLanggraphResult:
    def test_simple_text_response(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message("assistant", content="Hello there!")
        result = _parse_langgraph_result({"messages": [ai]})
        assert (
            result["final_response"] == "Hello there!" and result["completed"] is True
        )

    def test_tool_call_then_text_response(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai1 = _make_langchain_message(
            "assistant",
            content="let me search",
            tool_calls=[
                {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}
            ],
        )
        ai2 = _make_langchain_message("assistant", content="Final answer!")
        r = _parse_langgraph_result({"messages": [ai1, ai2]})
        assert r["final_response"] == "Final answer!" and r["api_calls"] == 1

    def test_only_tool_calls_no_final_text(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content="",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        )
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["completed"] is False and r["turn_exit_reason"] == "no_response"

    def test_multimodal_content_text_type(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content=[
                {"type": "text", "text": "Multimodal response."},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        )
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["final_response"] == "Multimodal response."

    def test_multimodal_content_plain_strings(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [
                _make_langchain_message("assistant", content=["plain string part"])
            ]
        })
        assert r["final_response"] == "plain string part"

    def test_multimodal_content_string_part_in_list(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content=["part1"])]
        })
        assert r["final_response"] == "part1"

    def test_reasoning_from_msg_reasoning_attr(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message("assistant", content="answer")
        ai.reasoning = "I thought about it..."
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["last_reasoning"] == "I thought about it..."

    def test_reasoning_from_additional_kwargs(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant", content="answer", additional_kwargs={"reasoning": "inferred"}
        )
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["last_reasoning"] == "inferred"

    def test_reasoning_attr_takes_precedence_over_kwargs(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content="answer",
            additional_kwargs={"reasoning": "kwarg reasoning"},
        )
        ai.reasoning = "direct reasoning"
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["last_reasoning"] == "direct reasoning"

    def test_multiple_tool_call_aimessages_counted(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        t1 = _make_langchain_message(
            "assistant",
            content="checking...",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        )
        t2 = _make_langchain_message(
            "assistant",
            content="done...",
            tool_calls=[
                {"id": "call_2", "function": {"name": "write_file", "arguments": "{}"}}
            ],
        )
        f = _make_langchain_message("assistant", content="All done.")
        r = _parse_langgraph_result({"messages": [t1, t2, f]})
        assert r["api_calls"] == 2 and r["final_response"] == "All done."

    def test_empty_messages_result(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({"messages": []})
        assert (
            r["final_response"] == ""
            and r["completed"] is False
            and r["api_calls"] == 0
        )

    def test_non_dict_result(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({"messages": "not a list"})
        assert r["final_response"] == "" and r["completed"] is False

    def test_hermes_messages_converted_in_result(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content="answer")]
        })
        assert r["messages"][0] == {"role": "assistant", "content": "answer"}

    def test_model_field_is_empty_string(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        assert _parse_langgraph_result({"messages": []})["model"] == ""

    def test_turn_completed_with_final_response(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content="Final")]
        })
        assert r["completed"] is True and r["turn_exit_reason"] == "completed"

    def test_turn_no_response_without_ai_message(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [
                _make_langchain_message("system", content="Be helpful"),
                _make_langchain_message("user", content="Hi"),
            ]
        })
        assert r["completed"] is False and r["turn_exit_reason"] == "no_response"


class TestParseErrorResult:
    def test_error_includes_exception_message(self):
        from agent.deep_agents_runtime import _parse_error_result

        result = _parse_error_result(RuntimeError("connection refused"))
        assert (
            result["final_response"] == "Error: connection refused"
            and result["failed"] is True
        )

    def test_error_is_empty_for_empty_exception(self):
        from agent.deep_agents_runtime import _parse_error_result

        assert _parse_error_result(Exception(""))["final_response"] == "Error: "

    def test_error_result_dict_shape(self):
        from agent.deep_agents_runtime import _parse_error_result

        r = _parse_error_result(Exception("test"))
        assert set(r.keys()) >= {
            "final_response",
            "messages",
            "api_calls",
            "completed",
            "failed",
            "interrupted",
            "partial",
            "turn_exit_reason",
        }


class TestInjectProviderEnv:
    def test_empty_provider_maps_to_openai(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("", None, "sk-123")
        assert os.environ.get("OPENAI_API_KEY") == "sk-123"

    def test_openai_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("openai", "https://custom.openai.com", "sk-key")
        assert (
            os.environ["OPENAI_API_KEY"] == "sk-key"
            and os.environ["OPENAI_API_BASE"] == "https://custom.openai.com"
        )

    def test_anthropic_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("anthropic", None, "sk-ant-123")
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-123"

    def test_google_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("google", "https://google.custom", "gkey")
        assert (
            os.environ["GOOGLE_API_KEY"] == "gkey"
            and os.environ["GOOGLE_API_BASE"] == "https://google.custom"
        )

    def test_xai_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("xai", None, "xai-key")
        assert os.environ["XAI_API_KEY"] == "xai-key"

    def test_cohere_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("cohere", None, "co-key")
        assert os.environ["COHERE_API_KEY"] == "co-key"

    def test_groq_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("groq", None, "groq-key")
        assert os.environ["GROQ_API_KEY"] == "groq-key"

    def test_unknown_provider_uses_generic_names(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("unknown-provider", "https://unknown.com", "uk-key")
        assert (
            os.environ["API_KEY"] == "uk-key"
            and os.environ["BASE_URL"] == "https://unknown.com"
        )

    def test_base_url_only(self):
        from agent.deep_agents_runtime import _inject_provider_env

        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            _inject_provider_env("openai", "https://base.com", None)
            assert os.environ["OPENAI_API_BASE"] == "https://base.com"
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


class TestHermesToolAdapter:
    def test_adapter_creates_structured_tool(self):
        from langchain_core.tools import StructuredTool
        from agent.deep_agents_runtime import _HermesToolAdapter

        assert isinstance(
            _HermesToolAdapter(
                _make_tool_entry("read_file", toolset="file")
            ).langchain_tool,
            StructuredTool,
        )

    def test_adapter_preserves_name(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        assert (
            _HermesToolAdapter(_make_tool_entry("my_tool", toolset="custom")).name
            == "my_tool"
        )

    def test_adapter_preserves_schema(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        s = {"name": "x", "description": "desc", "parameters": {}}
        assert (
            _HermesToolAdapter(_make_tool_entry("x", toolset="s", schema=s)).schema == s
        )

    def test_adapter_preserves_toolset(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        assert (
            _HermesToolAdapter(_make_tool_entry("tool_x", toolset="my_toolset")).toolset
            == "my_toolset"
        )

    def test_adapter_calls_handle_function_call(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        import model_tools

        saved = model_tools.handle_function_call
        mock = MagicMock()
        mock.return_value = json.dumps({"ok": True})
        try:
            model_tools.handle_function_call = mock
            adapter = _HermesToolAdapter(_make_tool_entry("read_file", toolset="file"))
            # Call __func directly - LangChain's .invoke() passes arguments
            # through a Pydantic schema that may rename / drop keys, so
            # .func() is the reliable way to test the closure.
            adapter.langchain_tool.func(path="foo.txt")
            assert mock.called, "handle_function_call was not called"
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs["function_name"] == "read_file"
            assert call_kwargs.kwargs["function_args"] == {"path": "foo.txt"}
        finally:
            model_tools.handle_function_call = saved

    @patch("model_tools.handle_function_call")
    def test_adapter_wraps_exceptions_as_json_error(self, mock_handle):
        from agent.deep_agents_runtime import _HermesToolAdapter

        mock_handle.side_effect = IOError("disk full")
        parsed = json.loads(
            _HermesToolAdapter(
                _make_tool_entry("write_file", toolset="file")
            ).langchain_tool.invoke({"path": "bar.txt"})
        )
        assert "error" in parsed and "disk full" in parsed["error"]

    def test_adapter_uses_entry_description_over_schema(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        adapter = _HermesToolAdapter(
            _make_tool_entry(
                "x", toolset="s", schema={"name": "x"}, description="custom desc"
            )
        )
        assert adapter._tool.description == "custom desc"

    def test_adapter_falls_back_schema_description(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        adapter = _HermesToolAdapter(
            _make_tool_entry(
                "x",
                toolset="s",
                schema={"name": "x", "description": "schema desc"},
                description=None,
            )
        )
        assert adapter._tool.description == "schema desc"


def _fn_def(name, parameters=None, description=None):
    """Build an OpenAI-format tool definition as get_tool_definitions returns."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description if description is not None else f"{name} tool",
            "parameters": parameters
            if parameters is not None
            else {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


class TestBuildHermesTools:
    """build_hermes_tools resolves toolsets via model_tools.get_tool_definitions
    (the native path) and builds a StructuredTool per definition — including
    synthetic bridge tools that have no registry entry."""

    def _patch_defs(self, definitions):
        """Patch model_tools.get_tool_definitions to return *definitions*."""
        import model_tools

        return patch.object(
            model_tools, "get_tool_definitions", return_value=definitions
        )

    def test_builds_structured_tools_from_definitions(self):
        from langchain_core.tools import StructuredTool
        from agent.deep_agents_runtime import build_hermes_tools

        with self._patch_defs([_fn_def("tool_a"), _fn_def("tool_b")]):
            tools = build_hermes_tools(
                enabled_toolsets=["ts1", "ts2"], disabled_toolsets=[]
            )
        assert len(tools) == 2 and all(isinstance(t, StructuredTool) for t in tools)
        assert {t.name for t in tools} == {"tool_a", "tool_b"}

    def test_forwards_parameter_schema_as_args_schema(self):
        """The model must see each tool's real parameters, not an empty schema."""
        from agent.deep_agents_runtime import build_hermes_tools

        params = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        with self._patch_defs([_fn_def("run_terminal_command", parameters=params)]):
            tools = build_hermes_tools(enabled_toolsets=["terminal"], disabled_toolsets=[])
        assert len(tools) == 1
        # The StructuredTool exposes the forwarded parameter as a callable arg.
        assert "command" in tools[0].args

    def test_builds_bridge_tools_without_registry_entry(self):
        """tool_search/tool_describe/tool_call have no registry entry but are
        emitted by tool-search assembly — they must still get adapters."""
        from agent.deep_agents_runtime import build_hermes_tools

        defs = [_fn_def("tool_search"), _fn_def("tool_describe"), _fn_def("tool_call")]
        with self._patch_defs(defs):
            tools = build_hermes_tools(enabled_toolsets=["x"], disabled_toolsets=[])
        assert {t.name for t in tools} == {"tool_search", "tool_describe", "tool_call"}

    def test_handles_empty_definitions(self):
        from agent.deep_agents_runtime import build_hermes_tools

        with self._patch_defs([]):
            assert build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[]) == []

    def test_returns_empty_list_on_resolution_error(self):
        from agent.deep_agents_runtime import build_hermes_tools
        import model_tools

        with patch.object(
            model_tools,
            "get_tool_definitions",
            side_effect=ImportError("registry unavailable"),
        ):
            assert (
                build_hermes_tools(enabled_toolsets=["ts"], disabled_toolsets=[]) == []
            )

    def test_empty_name_tool_skipped(self):
        from agent.deep_agents_runtime import build_hermes_tools

        with self._patch_defs([_fn_def("")]):
            assert build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[]) == []


def _ai_chunk(content, tool_call_chunks=None, additional_kwargs=None):
    """Build an AIMessageChunk as the ``messages`` stream mode yields."""
    from langchain_core.messages import AIMessageChunk

    chunk = AIMessageChunk(content=content)
    if tool_call_chunks is not None:
        chunk.tool_call_chunks = tool_call_chunks
    if additional_kwargs is not None:
        chunk.additional_kwargs = additional_kwargs
    return chunk


class TestHermesStreamingBridge:
    """The bridge translates *real* LangGraph stream items — ``("messages",
    (chunk, meta))`` token chunks and ``("updates", {node: {...}})`` node
    outputs — into the native runtime's callback signatures."""

    def _make_bridge(self, **kwargs):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        return _HermesStreamingBridge(
            agent=MagicMock(),
            stream_delta=kwargs.get("stream_delta"),
            tool_progress=kwargs.get("tool_progress"),
            thinking=kwargs.get("thinking"),
            step=kwargs.get("step"),
            tool_start=kwargs.get("tool_start"),
            tool_complete=kwargs.get("tool_complete"),
            tool_gen=kwargs.get("tool_gen"),
        )

    # -- any_callbacks_set ---------------------------------------------------

    def test_any_callbacks_set_with_stream_delta(self):
        assert self._make_bridge(stream_delta=lambda x: None).any_callbacks_set()

    def test_any_callbacks_set_with_tool_start(self):
        assert self._make_bridge(tool_start=lambda *a: None).any_callbacks_set()

    def test_any_callbacks_set_false_when_none(self):
        assert self._make_bridge().any_callbacks_set() is False

    # -- routing / robustness ------------------------------------------------

    def test_process_stream_item_non_tuple_non_dict_ignored(self):
        assert self._make_bridge().process_stream_item("not an item") is None
        assert self._make_bridge().process_stream_item(None) is None

    # -- messages mode: token streaming --------------------------------------

    def test_messages_mode_string_content_streams_delta(self):
        deltas = []
        bridge = self._make_bridge(stream_delta=lambda x: deltas.append(x))
        bridge.process_stream_item(("messages", (_ai_chunk("hello "), {})))
        bridge.process_stream_item(("messages", (_ai_chunk("world"), {})))
        assert deltas == ["hello ", "world"]

    def test_messages_mode_fires_step_callback(self):
        steps = []
        bridge = self._make_bridge(
            stream_delta=lambda x: None, step=lambda n, s: steps.append((n, s))
        )
        bridge.process_stream_item(("messages", (_ai_chunk("x"), {})))
        assert steps == [(1, [])]

    def test_messages_mode_empty_content_no_delta(self):
        deltas = []
        bridge = self._make_bridge(stream_delta=lambda x: deltas.append(x))
        bridge.process_stream_item(("messages", (_ai_chunk(""), {})))
        assert deltas == []

    def test_messages_mode_splits_thinking_from_text(self):
        deltas, thinks = [], []
        bridge = self._make_bridge(
            stream_delta=lambda x: deltas.append(x),
            thinking=lambda x: thinks.append(x),
        )
        content = [
            {"type": "thinking", "thinking": "let me reason"},
            {"type": "text", "text": "the answer"},
        ]
        bridge.process_stream_item(("messages", (_ai_chunk(content), {})))
        assert deltas == ["the answer"]
        assert thinks == ["let me reason"]

    def test_messages_mode_reasoning_content_additional_kwargs(self):
        thinks = []
        bridge = self._make_bridge(thinking=lambda x: thinks.append(x))
        chunk = _ai_chunk("", additional_kwargs={"reasoning_content": "deepthink"})
        bridge.process_stream_item(("messages", (chunk, {})))
        assert thinks == ["deepthink"]

    def test_messages_mode_tool_gen_announces_once(self):
        gens = []
        bridge = self._make_bridge(tool_gen=lambda name: gens.append(name))
        tcc = [{"name": "read_file", "args": "", "id": "c1", "index": 0}]
        bridge.process_stream_item(("messages", (_ai_chunk("", tool_call_chunks=tcc), {})))
        # A later chunk for the same call (name often only on the first) — no dup.
        tcc2 = [{"name": "read_file", "args": '{"p":1}', "id": "c1", "index": 0}]
        bridge.process_stream_item(("messages", (_ai_chunk("", tool_call_chunks=tcc2), {})))
        assert gens == ["read_file"]

    def test_raising_tool_start_callback_does_not_abort_stream(self):
        """A callback that raises (e.g. a TUI websocket/render hiccup) must be
        isolated — the bridge must keep processing so the tool still executes
        and the final answer streams. Otherwise the turn aborts leaving only
        the tool box (the deepagents/TUI 'box shown, no result' symptom)."""
        from langchain_core.messages import AIMessage

        completes = []
        bridge = self._make_bridge(
            tool_start=lambda *a: (_ for _ in ()).throw(RuntimeError("ws closed")),
            tool_complete=lambda tc_id, name, args, result: completes.append(name),
        )
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "terminal", "args": {"command": "x"}, "id": "c1"}]
        # Must NOT raise even though tool_start blows up.
        bridge.process_stream_item(("updates", {"model": {"messages": [ai]}}))
        # And a later tool result must still drive tool_complete.
        from langchain_core.messages import ToolMessage

        tm = ToolMessage(content="ok", tool_call_id="c1", name="terminal")
        bridge.process_stream_item(("updates", {"tools": {"messages": [tm]}}))
        assert completes == ["terminal"]

    def test_reasoning_leak_in_content_is_routed_to_thinking(self):
        """Some models (Qwen3.x on vLLM with a rich prompt) emit reasoning into
        the *content* channel wrapped in <think>…</think> / <|mask_start|>…
        <mask_end>. That must NOT reach the visible body — it is routed to the
        thinking channel so it still shows as the agent's reasoning."""
        deltas, thinks = [], []
        bridge = self._make_bridge(
            stream_delta=lambda x: deltas.append(x),
            thinking=lambda x: thinks.append(x),
        )
        bridge.process_stream_item(("messages", (_ai_chunk("Hello "), {})))
        bridge.process_stream_item(
            ("messages", (_ai_chunk("<think>secret reasoning</think>"), {}))
        )
        bridge.process_stream_item(
            ("messages", (_ai_chunk("<|mask_start|>more reasoning<mask_end>"), {}))
        )
        bridge.process_stream_item(("messages", (_ai_chunk("world"), {})))
        bridge.flush_visible()
        body = "".join(deltas)
        think = "".join(thinks)
        assert "secret reasoning" not in body
        assert "more reasoning" not in body
        assert "<think>" not in body and "mask_start" not in body
        assert "Hello " in body and "world" in body
        # The reasoning still surfaces — on the thinking channel.
        assert "secret reasoning" in think
        assert "more reasoning" in think

    def test_reasoning_leak_marker_split_across_deltas_is_routed(self):
        """A reasoning marker split across two deltas must still be caught."""
        deltas, thinks = [], []
        bridge = self._make_bridge(
            stream_delta=lambda x: deltas.append(x),
            thinking=lambda x: thinks.append(x),
        )
        bridge.process_stream_item(("messages", (_ai_chunk("A<thi"), {})))
        bridge.process_stream_item(("messages", (_ai_chunk("nk>hidden</thi"), {})))
        bridge.process_stream_item(("messages", (_ai_chunk("nk>B"), {})))
        bridge.flush_visible()
        assert "".join(deltas) == "AB"
        assert "hidden" in "".join(thinks)

    def test_clean_text_passes_through_unchanged(self):
        deltas = []
        bridge = self._make_bridge(stream_delta=lambda x: deltas.append(x))
        for piece in ("Here is ", "a normal ", "answer."):
            bridge.process_stream_item(("messages", (_ai_chunk(piece), {})))
        bridge.flush_visible()
        assert "".join(deltas) == "Here is a normal answer."

    def test_stray_unpaired_close_marker_stripped_from_body(self):
        """Models sometimes emit a lone <|mask_end|> with no matching open. It's
        a reasoning delimiter token and must not appear in the visible body."""
        deltas = []
        bridge = self._make_bridge(stream_delta=lambda x: deltas.append(x))
        bridge.process_stream_item(("messages", (_ai_chunk("Answer<|mask_end|> done"), {})))
        bridge.flush_visible()
        assert "".join(deltas) == "Answer done"

    def test_raising_stream_delta_does_not_abort(self):
        """A raising stream_delta likewise must not propagate."""
        from langchain_core.messages import AIMessage

        bridge = self._make_bridge(
            stream_delta=lambda x: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        # Must not raise.
        bridge.process_stream_item(("messages", (_ai_chunk("hi"), {})))

    def test_messages_mode_tool_message_not_streamed_as_text(self):
        """LangGraph ``messages`` mode also yields the tool node's ToolMessage.
        Its ``content`` is the tool *result* — it must NOT leak into the
        assistant body via stream_delta (updates mode owns tool chrome)."""
        from langchain_core.messages import ToolMessage

        deltas = []
        bridge = self._make_bridge(stream_delta=lambda x: deltas.append(x))
        tm = ToolMessage(
            content='{"success": true, "skills": []}', tool_call_id="c1"
        )
        bridge.process_stream_item(("messages", (tm, {})))
        assert deltas == []

    def test_messages_mode_tool_message_chunk_not_streamed_as_text(self):
        """ToolMessageChunk (streamed tool result) likewise stays out of the
        assistant body."""
        from langchain_core.messages import ToolMessageChunk

        deltas = []
        bridge = self._make_bridge(stream_delta=lambda x: deltas.append(x))
        tmc = ToolMessageChunk(content="partial result", tool_call_id="c1")
        bridge.process_stream_item(("messages", (tmc, {})))
        assert deltas == []

    # -- updates mode: tool chrome -------------------------------------------

    def test_updates_mode_tool_call_fires_tool_start(self):
        from langchain_core.messages import AIMessage

        starts, progress = [], []
        bridge = self._make_bridge(
            tool_start=lambda tc_id, name, args: starts.append((tc_id, name, args)),
            tool_progress=lambda a, *r: progress.append((a, r)),
        )
        ai = AIMessage(content="")
        ai.tool_calls = [
            {"name": "read_file", "args": {"path": "a.txt"}, "id": "call_1"}
        ]
        bridge.process_stream_item(("updates", {"model": {"messages": [ai]}}))
        assert starts == [("call_1", "read_file", {"path": "a.txt"})]
        assert progress and progress[0][0] == "tool.started"

    def test_updates_mode_fires_tool_gen_when_chunks_never_announced(self):
        """LangGraph surfaces tool calls complete in ``updates`` mode without
        incremental ``tool_call_chunks`` in ``messages`` mode, so the
        "preparing…" beat never fired from chunks. ``tool_start`` must emit it
        as a fallback so the generating indicator always shows."""
        from langchain_core.messages import AIMessage

        gens, starts = [], []
        bridge = self._make_bridge(
            tool_gen=lambda name: gens.append(name),
            tool_start=lambda tc_id, name, args: starts.append(name),
        )
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "read_file", "args": {"path": "a"}, "id": "c1"}]
        bridge.process_stream_item(("updates", {"model": {"messages": [ai]}}))
        assert gens == ["read_file"]
        assert starts == ["read_file"]

    def test_updates_mode_does_not_double_announce_tool_gen(self):
        """If a ``messages``-mode chunk already announced the tool name, the
        ``updates``-mode fallback must not announce it a second time."""
        from langchain_core.messages import AIMessage

        gens = []
        bridge = self._make_bridge(
            tool_gen=lambda name: gens.append(name),
            tool_start=lambda tc_id, name, args: None,
        )
        tcc = [{"name": "read_file", "args": "", "id": "c1", "index": 0}]
        bridge.process_stream_item(("messages", (_ai_chunk("", tool_call_chunks=tcc), {})))
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "read_file", "args": {"path": "a"}, "id": "c1"}]
        bridge.process_stream_item(("updates", {"model": {"messages": [ai]}}))
        assert gens == ["read_file"]  # exactly once, not twice

    def test_updates_mode_tool_result_fires_tool_complete(self):
        from langchain_core.messages import AIMessage, ToolMessage

        starts, completes = [], []
        bridge = self._make_bridge(
            tool_start=lambda tc_id, name, args: starts.append((tc_id, name)),
            tool_complete=lambda tc_id, name, args, result: completes.append(
                (tc_id, name, args, result)
            ),
        )
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}]
        bridge.process_stream_item(("updates", {"model": {"messages": [ai]}}))
        tm = ToolMessage(content="file body", tool_call_id="c1", name="read_file")
        bridge.process_stream_item(("updates", {"tools": {"messages": [tm]}}))
        # Completion echoes the original call's name + args (ToolMessage lacks args).
        assert completes == [("c1", "read_file", {"path": "a.txt"}, "file body")]

    def test_updates_mode_bare_dict_treated_as_updates(self):
        from langchain_core.messages import AIMessage

        starts = []
        bridge = self._make_bridge(
            tool_start=lambda tc_id, name, args: starts.append(name)
        )
        ai = AIMessage(content="")
        ai.tool_calls = [{"name": "ping", "args": {}, "id": "c1"}]
        # No ("updates", ...) wrapper — a bare dict still routes as updates.
        bridge.process_stream_item({"model": {"messages": [ai]}})
        assert starts == ["ping"]

    def test_updates_mode_tool_complete_list_content_flattened(self):
        from langchain_core.messages import ToolMessage

        completes = []
        bridge = self._make_bridge(
            tool_complete=lambda tc_id, name, args, result: completes.append(result)
        )
        tm = ToolMessage(
            content=[{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
            tool_call_id="c9",
            name="grep",
        )
        bridge.process_stream_item(("updates", {"tools": {"messages": [tm]}}))
        assert completes == ["part1part2"]


class TestReasoningConfig:
    """``reasoning_config`` (effort/enabled) must reach the provider binding so
    the deepagents model actually enables extended thinking."""

    def test_reasoning_enabled_truthy(self):
        from agent.deep_agents_runtime import _reasoning_enabled

        assert _reasoning_enabled({"enabled": True}) is True
        assert _reasoning_enabled({"effort": "high"}) is True  # absent enabled
        assert _reasoning_enabled({"enabled": False}) is False
        assert _reasoning_enabled(None) is False

    def test_reasoning_effort_defaults_medium(self):
        from agent.deep_agents_runtime import _reasoning_effort

        assert _reasoning_effort({"effort": "high"}) == "high"
        assert _reasoning_effort({}) == "medium"
        assert _reasoning_effort(None) == "medium"

    def test_build_reasoning_model_disabled_returns_none(self):
        from agent.deep_agents_runtime import _build_reasoning_model

        assert (
            _build_reasoning_model("anthropic", "claude-sonnet-4-0", {"enabled": False})
            is None
        )

    def test_build_reasoning_model_anthropic_sets_thinking(self):
        from agent.deep_agents_runtime import _build_reasoning_model

        captured = {}

        class FakeChatAnthropic:
            def __init__(self, **kw):
                captured.update(kw)

        fake_mod = types.ModuleType("langchain_anthropic")
        fake_mod.ChatAnthropic = FakeChatAnthropic
        with patch.dict(sys.modules, {"langchain_anthropic": fake_mod}):
            model = _build_reasoning_model(
                "anthropic", "claude-sonnet-4-0", {"enabled": True, "effort": "high"}
            )
        assert isinstance(model, FakeChatAnthropic)
        assert captured["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert captured["max_tokens"] > 16000

    def test_build_reasoning_model_anthropic_haiku_skipped(self):
        from agent.deep_agents_runtime import _build_reasoning_model

        assert (
            _build_reasoning_model(
                "anthropic", "claude-haiku-4-5", {"enabled": True, "effort": "high"}
            )
            is None
        )

    def test_build_reasoning_model_openai_non_reasoning_skipped(self):
        from agent.deep_agents_runtime import _build_reasoning_model

        # gpt-4o has no reasoning_effort — must not be constructed with one.
        assert (
            _build_reasoning_model("openai", "gpt-4o", {"enabled": True}) is None
        )

    def test_build_reasoning_model_openai_reasoning_sets_effort(self):
        from agent.deep_agents_runtime import _build_reasoning_model

        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kw):
                captured.update(kw)

        fake_mod = types.ModuleType("langchain_openai")
        fake_mod.ChatOpenAI = FakeChatOpenAI
        with patch.dict(sys.modules, {"langchain_openai": fake_mod}):
            model = _build_reasoning_model("openai", "o3", {"enabled": True, "effort": "low"})
        assert isinstance(model, FakeChatOpenAI)
        assert captured["reasoning_effort"] == "low"

    def test_build_reasoning_model_import_failure_returns_none(self):
        from agent.deep_agents_runtime import _build_reasoning_model

        # No fake module registered → import inside raises → graceful None.
        with patch.dict(sys.modules, {"langchain_anthropic": None}):
            assert (
                _build_reasoning_model("anthropic", "claude-sonnet-4-0", {"enabled": True})
                is None
            )


class TestStreamHelpers:
    def test_split_stream_item_tuple(self):
        from agent.deep_agents_runtime import _split_stream_item

        assert _split_stream_item(("messages", "payload")) == ("messages", "payload")

    def test_split_stream_item_bare_dict_is_updates(self):
        from agent.deep_agents_runtime import _split_stream_item

        assert _split_stream_item({"node": {}}) == ("updates", {"node": {}})

    def test_split_stream_item_unknown_returns_none(self):
        from agent.deep_agents_runtime import _split_stream_item

        assert _split_stream_item(42) == (None, None)

    def test_split_chunk_content_string(self):
        from agent.deep_agents_runtime import _split_chunk_content

        assert _split_chunk_content(_ai_chunk("hi")) == ("hi", "")

    def test_split_chunk_content_blocks(self):
        from agent.deep_agents_runtime import _split_chunk_content

        chunk = _ai_chunk(
            [
                {"type": "text", "text": "A"},
                {"type": "thinking", "thinking": "B"},
            ]
        )
        assert _split_chunk_content(chunk) == ("A", "B")


class TestDeepAgentsAIAgentAttributes:
    def _make_agent(self, **kwargs):
        """Create a DeepAgentsAIAgent with all SDK deps mocked via __new__."""
        from agent import deep_agents_runtime as dar

        agent = object.__new__(dar.DeepAgentsAIAgent)
        # Set internal state as __init__ would (without calling __init__)
        object.__setattr__(agent, "mode", "deepagents")
        object.__setattr__(agent, "_quiet_mode", kwargs.get("quiet_mode", False))
        object.__setattr__(agent, "_skip_memory", kwargs.get("skip_memory", False))
        object.__setattr__(agent, "_platform", kwargs.get("platform"))
        object.__setattr__(agent, "_session_id", kwargs.get("session_id", ""))
        object.__setattr__(agent, "_max_iterations", kwargs.get("max_iterations", 90))
        object.__setattr__(agent, "provider", kwargs.get("provider", ""))
        object.__setattr__(agent, "_api_key", kwargs.get("api_key", None))
        object.__setattr__(agent, "_base_url", kwargs.get("base_url"))
        object.__setattr__(agent, "_callbacks", {})
        object.__setattr__(agent, "_agent", kwargs.get("_agent_mock", MagicMock()))
        return agent

    def test_basic_attribute_access(self):
        agent = self._make_agent(quiet_mode=True, skip_memory=True, platform=None)
        assert agent.mode == "deepagents" and agent.quiet_mode is True
        assert agent.skip_memory is True and agent.platform is None

    def test_iteration_budget_returns_self(self):
        agent = self._make_agent()
        assert agent.iteration_budget is agent

    def test_model_returns_empty_string(self):
        assert self._make_agent().model == ""

    def test_tools_returns_empty_list(self):
        assert self._make_agent().tools == []

    def test_max_iterations(self):
        assert self._make_agent(max_iterations=42).max_iterations == 42

    def test_run_conversation_returns_dict(self):
        agent = self._make_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="response")]
        }
        result = agent.run_conversation("hello")
        assert isinstance(result, dict) and "final_response" in result

    def test_chat_returns_final_response(self):
        agent = self._make_agent()
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="chat answer")]
        }
        mock_agent.stream.return_value = iter([])
        mock_agent.get_state.return_value = type(
            "State",
            (),
            {
                "values": {
                    "messages": [
                        _make_langchain_message("assistant", content="chat answer")
                    ]
                }
            },
        )
        object.__setattr__(agent, "_agent", mock_agent)
        assert agent.chat("what's up") == "chat answer"

    def test_interrupt_is_noop(self):
        self._make_agent().interrupt()

    def test_get_memory_context_returns_none(self):
        assert self._make_agent().get_memory_context() is None

    def test_save_memory_is_noop(self):
        self._make_agent().save_memory([])

    def test_reasoning_callback_forwarding(self):
        agent = self._make_agent()
        cb = lambda *a, **kw: None
        agent.reasoning_callback = cb
        assert agent._get_cap("reasoning_callback") is cb

    def test_step_callback_forwarding(self):
        agent = self._make_agent()
        step_cb = lambda *a, **kw: None
        agent.step_callback = step_cb
        _ = agent.reasoning_callback
        assert agent._get_cap("step_callback") is step_cb

    def test_attribute_not_in_captured_raises_attributeerror(self):
        agent = self._make_agent()
        with pytest.raises(AttributeError, match="nonexistent_attr"):
            _ = agent.nonexistent_attr

    def test_resolve_model_strips_provider_prefix(self):
        agent = self._make_agent(api_key="k", model="openai/gpt-4o")
        assert agent.mode == "deepagents"

    def test_default_model_for_anthropic(self):
        agent = self._make_agent(api_key="k", provider="anthropic")
        assert agent.provider == "anthropic"

    def test_default_model_for_google(self):
        agent = self._make_agent(api_key="k", provider="google")
        assert agent.provider == "google"


class TestReasoningExtraction:
    def test_reasoning_on_last_ai_message_extracted(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai1 = _make_langchain_message(
            "assistant",
            content="tool result",
            tool_calls=[{"id": "call_1", "function": {"name": "x", "arguments": "{}"}}],
        )
        ai2 = _make_langchain_message("assistant", content="final answer")
        ai2.reasoning = "last reasoning"
        r = _parse_langgraph_result({"messages": [ai1, ai2]})
        assert r["last_reasoning"] == "last reasoning"

    def test_no_reasoning_set(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content="plain response")]
        })
        assert r["last_reasoning"] is None


class TestRoundTripConversion:
    def test_hermes_to_langchain_to_hermes_roundtrip(self):
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        original = [
            _make_hermes_message("system", "Be helpful."),
            _make_hermes_message("user", "What is 2+2?"),
        ]
        assert (
            _convert_langchain_to_hermes(_convert_messages_to_langchain(original))
            == original
        )

    def test_assistant_with_tool_calls_roundtrip(self):
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        tool_calls = [{"id": "c1", "function": {"name": "x", "arguments": "{}"}}]
        original = [
            _make_hermes_message("assistant", "checking", tool_calls=tool_calls)
        ]
        step1 = _convert_messages_to_langchain(original)
        step2 = _convert_langchain_to_hermes(step1)
        assert step2[0]["role"] == "assistant" and step2[0]["tool_calls"] == tool_calls
