"""Tests for _HadesStreamingBridge and _shorten.

Covers: AIMessageChunk → message_delta, ToolCall → tool_start,
ToolResult → tool_complete, unknown events, raw events, _process_output,
final_result defaults, _shorten boundary conditions.
"""

import json

from types import SimpleNamespace

import pytest

from agent.deep_agents_streaming import _HadesStreamingBridge, _shorten


# ── _shorten ───────────────────────────────────────────────────────────────


class TestShorten:

    def test_returns_unchanged_when_short(self):
        """Strings at or under max_len are returned as-is."""
        assert _shorten("hello", 10) == "hello"
        assert _shorten("hello", 5) == "hello"

    def test_truncates_with_ellipsis(self):
        """Strings over max_len are truncated and appended with ..."""
        result = _shorten("a" * 300, 50)
        assert len(result) == 50
        assert result.endswith("...")
        assert result == "a" * 47 + "..."

    def test_shortens_to_exact_boundary(self):
        """First max_len - 3 text characters plus '...' equals max_len."""
        result = _shorten("x" * 100, 10)
        assert len(result) == 10
        assert result == "xxxxxxx..."

    def test_empty_string(self):
        """Empty string is not affected."""
        assert _shorten("", 50) == ""

    def test_max_len_zero(self):
        """max_len=0 produces just '...' (if max_len < 3)."""
        result = _shorten("abc", 0)
        assert result == "..."

    def test_max_len_three(self):
        """max_len=3 on an oversized string gives exactly '...'."""
        result = _shorten("abcde", 3)
        assert result == "..."

    def test_single_byte_text(self):
        """ASCII-only long text truncates correctly."""
        result = _shorten("hello world foo bar baz" * 10, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_unicode_text_truncation(self):
        """Unicode is byte-length safe (chars count in Python)."""
        result = _shorten("\u2603" * 300, 50)  # snowman repeated
        assert len(result) == 50
        assert result.endswith("...")


# ── _HadesStreamingBridge – construction ──────────────────────────────────


class TestBridgeConstruction:

    def test_defaults(self):
        """Bridge stores callback, agent, task_id; _result starts None."""
        mock_cb = lambda _e, **_k: None
        bridge = _HadesStreamingBridge(mock_cb, "agent", "task-1")
        assert bridge._result is None
        assert bridge._events == []

    def test_attribute_isolation(self):
        """Different bridges get independent state."""
        bridge_a = _HadesStreamingBridge(lambda _e, **_k: None, None, "task-a")
        bridge_b = _HadesStreamingBridge(lambda _e, **_k: None, None, "task-b")
        bridge_a._result = {"foo": 1}
        assert bridge_b._result is None


# ── _process_sub_event – AIMessageChunk ────────────────────────────────────


class TestProcessSubEventAIMessageChunk:

    def test_yields_message_delta_for_content(self):
        """AIMessageChunk with content yields ('message_delta', ...)."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "AIMessageChunk",
            "data": {"content": "Hello"},
        }))
        assert ("message_delta", {"text": "Hello"}) in events

    def test_skips_empty_content(self):
        """AIMessageChunk with empty content yields nothing."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "AIMessageChunk",
            "data": {"content": ""},
        }))
        assert events == []

    def test_skips_unset_content_key(self):
        """AIMessageChunk without 'content' key yields nothing."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "AIMessageChunk",
            "data": {},
        }))
        assert events == []

    def test_empty_string_content_skipped(self):
        """AIMessageChunk with empty string content is skipped."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "AIMessageChunk",
            "data": {"content": ""},
        }))
        assert events == []


# ── _process_sub_event – ToolCall ──────────────────────────────────────────


class TestProcessSubEventToolCall:

    def test_yields_tool_start(self):
        """ToolCall event yields ('tool_start', ...) with name + str(args)."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "ToolCall",
            "data": {"name": "read_file", "args": {"path": "/foo.txt"}},
        }))
        assert events == [("tool_start", {"tool_name": "read_file", "args": "{'path': '/foo.txt'}"})]

    def test_defaults_for_missing_fields(self):
        """ToolCall with missing name/args uses empty strings."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "ToolCall",
            "data": {},
        }))
        assert events == [("tool_start", {"tool_name": "", "args": "{}"})]


# ── _process_sub_event – ToolResult ────────────────────────────────────────


class TestProcessSubEventToolResult:

    def test_yields_tool_complete_with_short_result(self):
        """Non-truncated results are stored as-is."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "ToolResult",
            "data": {"name": "read_file", "content": "short"},
        }))
        assert events == [("tool_complete", {"tool_name": "read_file", "result_preview": "short"})]

    def test_yields_tool_complete_truncates_long_result(self):
        """Long results are truncated to 200 chars + '...'."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        long_content = "x" * 500
        events = list(bridge._process_sub_event({
            "type": "ToolResult",
            "data": {"name": "read_file", "content": long_content},
        }))
        assert events == [("tool_complete", {
            "tool_name": "read_file",
            "result_preview": "x" * 197 + "...",
        })]

    def test_truncated_text_is_not_triple_ellipsis(self):
        """If max_len=200, the preview ends with exactly one '...'."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "ToolResult",
            "data": {"name": "t", "content": "y" * 300},
        }))
        assert events[0][1]["result_preview"][-3:] == "..."

    def test_missing_content_key(self):
        """ToolResult without content uses empty string."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "ToolResult",
            "data": {"name": "read_file"},
        }))
        assert events == [("tool_complete", {"tool_name": "read_file", "result_preview": ""})]


# ── _process_sub_event – unknown event type ────────────────────────────────


class TestProcessSubEventUnknownType:

    def test_yields_raw_event_for_unknown_type(self):
        """Unknown event_type yields ('raw_event', ...) with type + data."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "CustomNode",
            "data": {"key": "value", "count": 42},
        }))
        assert ("raw_event", {"type": "CustomNode", "data": str({"key": "value", "count": 42})[:200]}) in events

    def test_empty_type_yields_raw_event(self):
        """Empty type string still yields a raw_event."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event({
            "type": "",
            "data": {},
        }))
        assert ("raw_event", {"type": "", "data": "{}"}) in events

    def test_non_dict_sub_event_rejected(self):
        """Passing a non-dict sub_event returns without yielding."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event("not a dict"))
        assert events == []

    def test_none_sub_event_rejected(self):
        """Passing None sub_event returns without yielding."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_sub_event(None))
        assert events == []


# ── _process_output ────────────────────────────────────────────────────────


class TestProcessOutput:

    def test_dict_output_stored_and_yields_complete(self):
        dict_output = {"final_response": "done", "api_calls": 3}
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_output(dict_output))
        assert bridge._result == dict_output
        assert ("complete", {"result": dict_output}) in events

    def test_string_json_output_parsed(self):
        """JSON string is parsed, stored, and yielded."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        json_str = json.dumps({"done": True, "value": 42})
        events = list(bridge._process_output(json_str))
        assert bridge._result == {"done": True, "value": 42}
        assert ("complete", {"result": {"done": True, "value": 42}}) in events

    def test_string_non_json_output_stored_as_error(self):
        """Non-JSON string produces error dict with final_response."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_output("failed with error"))
        expected = {"final_response": "failed with error", "errors": ["failed with error"]}
        assert bridge._result == expected
        assert ("complete", {"result": expected}) in events

    def test_empty_string_becomes_error_dict(self):
        """Empty string fails JSON.parse → error dict."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_output(""))
        assert bridge._result == {"final_response": "", "errors": [""]}

    def test_non_dict_non_str_output_ignored(self):
        """Numbers, lists etc. produce no yields and no side effects."""
        bridge = _HadesStreamingBridge(lambda _e, **_k: None, None, None)
        events = list(bridge._process_output(42))
        assert events == []
        assert bridge._result is None


# ── iter_events – full stream ──────────────────────────────────────────────


class TestIterEvents:

    def test_processes_Events_key(self):
        """Streaming dict with 'events' keys is dispatched to _process_sub_event."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        stream = [
            {"events": [
                {"type": "AIMessageChunk", "data": {"content": "hi"}},
                {"type": "ToolCall", "data": {"name": "cmd", "args": {}}},
            ]},
        ]
        yields = list(bridge.iter_events(stream))
        yield_types = [y[0] for y in yields]
        assert "message_delta" in yield_types
        assert "tool_start" in yield_types

    def test_processes_output_key(self):
        """Streaming dict with 'output' key delegates to _process_output."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        stream = [
            {"output": {"final_response": "done", "api_calls": 1}},
        ]
        yields = list(bridge.iter_events(stream))
        yield_types = [y[0] for y in yields]
        assert "complete" in yield_types

    def test_processes_output_as_json_string(self):
        """Output delivered as JSON string is parsed."""
        captured = []
        def cb(**k):
            captured.append(k)
        bridge = _HadesStreamingBridge(cb, None, None)
        stream = [
            {"output": json.dumps({"answer": "ok"})},
        ]
        yields = list(bridge.iter_events(stream))
        assert bridge._result == {"answer": "ok"}

    def test_unknown_dict_yields_unknown(self):
        """Dict without 'events' or 'output' yields ('unknown', dict)."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        stream = [
            {"some_other_key": 123},
        ]
        yields = list(bridge.iter_events(stream))
        assert ("unknown", {"some_other_key": 123}) in yields

    def test_non_dict_event_yields_raw_event(self):
        """Non-dict items in the stream yield ('raw_event', {'data': ...})."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        stream = [
            SimpleNamespace(foo=1),
        ]
        yields = list(bridge.iter_events(stream))
        assert ("raw_event", {"data": "namespace(foo=1)"}) in yields

    def test_mixed_stream(self):
        """Mix of dict events, output events, and raw events all processed."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        stream = [
            {"events": [{"type": "AIMessageChunk", "data": {"content": "start"}}]},
            SimpleNamespace(heartbeat=True),
            {"output": {"final": True}},
            {"unknown_field": 42},
        ]
        yields = list(bridge.iter_events(stream))
        yield_types = [y[0] for y in yields]
        assert "message_delta" in yield_types
        assert "raw_event" in yield_types
        assert "complete" in yield_types
        assert "unknown" in yield_types

    def test_callback_is_not_invoked_directly(self):
        """iter_events yields events but does not call the callback."""
        callback_called = []
        def cb(event, **kw):
            callback_called.append((event, kw))
        bridge = _HadesStreamingBridge(cb, None, None)
        stream = [
            {"events": [{"type": "AIMessageChunk", "data": {"content": "test"}}]},
        ]
        yields = list(bridge.iter_events(stream))
        assert len(callback_called) == 0




# ── _get_latest_tool_event ─────────────────────────────────────────────────


class TestGetLatestToolEvent:

    def test_returns_last_tool_event(self):
        event_a = ("tool_start", {"tool_name": "read_file", "args": "..."})
        event_b = ("message_delta", {"text": "hello"})
        event_c = ("tool_complete", {"tool_name": "read_file", "result_preview": "ok"})
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        bridge._events = [event_a, event_b, event_c]
        kind, data = bridge._get_latest_tool_event()
        assert kind == "tool_complete"
        assert data == {"tool_name": "read_file", "result_preview": "ok"}

    def test_returns_none_when_no_tool_events(self):
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        bridge._events = [
            ("message_delta", {"text": "hi"}),
            ("raw_event", {"type": "Custom", "data": ""}),
        ]
        kind, data = bridge._get_latest_tool_event()
        assert kind is None
        assert data is None


# ── final_result ───────────────────────────────────────────────────────────


class TestFinalResult:

    def test_returns_result_when_set_as_dict(self):
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        bridge._result = {"final_response": "done", "api_calls": 3}
        result = bridge.final_result()
        assert result == {"final_response": "done", "api_calls": 3}

    def test_returns_default_when_result_is_none(self):
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        result = bridge.final_result()
        assert result == {
            "final_response": "",
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "failed": False,
        }

    def test_returns_default_when_result_is_string(self):
        """Non-dict _result (e.g. parsed JSON string that was a list) → default."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        bridge._result = [1, 2, 3]  # not a dict
        result = bridge.final_result()
        assert result["completed"] is False

    def test_returns_default_when_result_is_empty_list(self):
        """Empty list is falsy → triggers default."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        bridge._result = []
        result = bridge.final_result()
        assert result == {
            "final_response": "",
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "failed": False,
        }

    def test_returns_default_when_result_is_empty_dict(self):
        """Empty dict is falsy → triggers default."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        bridge._result = {}
        result = bridge.final_result()
        assert result == {
            "final_response": "",
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "failed": False,
        }

    def test_non_dict_string_result(self):
        """Result that was set via _process_output with JSON string → returns parsed dict."""
        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        json_data = json.dumps({"final_response": "ok", "api_calls": 1})
        bridge._result = json.loads(json_data)
        assert isinstance(bridge._result, dict)
        result = bridge.final_result()
        assert result["final_response"] == "ok"


# ── Integration: Full stream round-trip ────────────────────────────────────


class TestFullStreamIntegration:

    def test_full_event_sequence_with_callback(self):
        """Capture all events from a realistic sequence."""
        captured = []
        def capture(event, **kw):
            captured.append((event, kw))

        stream = [
            {"events": [{"type": "AIMessageChunk", "data": {"content": "I"}}]},
            {"events": [{"type": "AIMessageChunk", "data": {"content": "'ll"}}]}
            ,
            {"events": [{"type": "ToolCall", "data": {"name": "terminal", "args": {"command": "ls"}}}]},
            {"events": [{"type": "ToolResult", "data": {"name": "terminal", "content": "file1\nfile2\nfile3"}}]},
        ]

        bridge = _HadesStreamingBridge(capture, "agent", "task-42")
        yields = list(bridge.iter_events(stream))

        # Check yields contain expected events
        yield_types = {y[0] for y in yields}
        assert "message_delta" in yield_types
        assert "tool_start" in yield_types
        assert "tool_complete" in yield_types

        # Check final_result is still None (no output node)
        result = bridge.final_result()
        assert result["completed"] is False
        assert result["final_response"] == ""

    def test_full_sequence_with_output_node(self):
        """Stream includes output node → final_result reflected."""
        captured = []
        def capture(event, **kw):
            captured.append((event, kw))

        stream = [
            {"events": [{"type": "AIMessageChunk", "data": {"content": "result"}}]},
            {"output": json.dumps({"final_response": "done", "api_calls": 2, "completed": True})},
        ]

        bridge = _HadesStreamingBridge(capture, "agent", "task-99")
        yields = list(bridge.iter_events(stream))

        result = bridge.final_result()
        assert result["final_response"] == "done"
        assert result["api_calls"] == 2
        assert result["completed"] is True
        # _process_output yields "complete"
        assert any(y[0] == "complete" for y in yields)

    def test_callback_receives_all_events(self):
        """Callback is called for every event type (message_delta, tool_start, tool_complete, complete)."""
        captured = []
        def capture(event, **kw):
            captured.append((event, {"task_id": "task-5", **kw}))

        stream = [
            {"events": [
                {"type": "AIMessageChunk", "data": {"content": "hi"}},
                {"type": "ToolCall", "data": {"name": "echo", "args": {"msg": "hello"}}},
                {"type": "ToolResult", "data": {"name": "echo", "content": "hello back"}},
            ]},
            {"output": json.dumps({"final_response": "ok"})},
        ]

        bridge = _HadesStreamingBridge(
            callback=capture,
            agent="agent",
            task_id="task-5",
        )
        yields = list(bridge.iter_events(stream))

        # Now manually invoke the callback for each event (since bridge doesn't auto-call)
        for event_type, data in yields:
            bridge._callback(event_type, **data)

        event_types = [c[0] for c in captured]
        assert "message_delta" in event_types
        assert "tool_start" in event_types
        assert "tool_complete" in event_types
        assert "complete" in event_types

    def test_multiple_messages_and_tools(self):
        """Stream with multiple AIMessageChunk and ToolCall/ToolResult pairs."""
        stream = []
        for i in range(3):
            stream.append({"events": [{"type": "AIMessageChunk", "data": {"content": f"step{i}"}}]})
            stream.append({"events": [{"type": "ToolCall", "data": {"name": f"tool{i}", "args": {"n": i}}}]})
            stream.append({"events": [{"type": "ToolResult", "data": {"name": f"tool{i}", "content": f"result{i}"}}]})

        bridge = _HadesStreamingBridge(lambda *a, **k: None, None, None)
        yields = list(bridge.iter_events(stream))

        deltas = [y for y in yields if y[0] == "message_delta"]
        tool_starts = [y for y in yields if y[0] == "tool_start"]
        tool_completes = [y for y in yields if y[0] == "tool_complete"]
        assert len(deltas) == 3
        assert len(tool_starts) == 3
        assert len(tool_completes) == 3
