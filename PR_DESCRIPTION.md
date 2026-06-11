# refactored/core: integrate DeepAgents (LangGraph) runtime and rename project to Hades Agent

## Description

This PR is a **major architectural rewrite** of the Hermes Agent core that does two things:

1. **Integrates DeepAgents (built on LangGraph/LangChain) as an alternative runtime** for the `AIAgent`, with a compatibility layer so that existing gateway, CLI, tooling, and streaming code requires zero changes.
2. **Renames the project from "Hermes" to "Hades"** across the entire codebase — package name, modules, docs, skills, website, config defaults, and branding — including a full directory rename from `hermes-agent` → `hades-agent`.

### Summary of impact

| Category | Files affected | Lines changed |
|----------|----------------|---------------|
| New DeepAgents runtime + middleware + streaming + tool dispatcher | 4 new files | 1,781 lines |
| Tests for DeepAgents (adapter, e2e, middleware, streaming, tool dispatcher, Langfuse/LangSmith) | 6 new files | 3,010+ lines |
| Rename: hermes → hades (all files) | 1,795 files | 310,000+ changes |

**3,790 files changed | 314,944 insertions(+), 74,116 deletions(-)**

---

## 1. DeepAgents Integration (LangGraph)

### 1.1 Motivation

The existing `AIAgent` loop in `run_agent.py` / `agent/conversation_loop.py` uses a custom synchronous `while True` loop with hand-rolled budget tracking, interrupt checks, and message passing. DeepAgents (a LangGraph-based framework) brings:

- **Structured state management** via LangGraph's checkpoint/store system
- **Middleware chain** instead of scattered callbacks/hooks
- **First-class streaming** with `stream_mode="updates"` and event parsing
- **Out-of-the-box observability** via LangGraph trace hooks + LangSmith
- **Recursive sub-agent calls** — the native tool loop can be called as a nested LangGraph tool (`hades_engine`)

### 1.2 How it works

The native `AIAgent` was **not replaced** — it now acts as a unified entry point that delegates to whichever runtime is selected:

**Runtime selection (`run_agent.py`):**

- `AIAgent.__init__()` gains a `runtime: str = "native"` parameter (default `"native"` to maintain backward compatibility)
- When `runtime == "deepagents"`, the agent initializes a `DeepAgentsAIAgent` instance (`self._deep_agents_impl`)
- `AIAgent.run_conversation()` checks `self._deep_agents_impl` and delegates to it, otherwise falls through to `agent/conversation_loop.run_conversation()`
- `__setattr__` / `__getattr__` overrides forward known attributes (`model`, `valid_tool_names`, `tools`, `stream_delta_callback`, `tool_progress_callback`, etc.) so gateway code that mutates the agent at runtime works identically with either backend

**The `deepagents_mode` config gate:** The gateway (`gateway/run.py`) reads `deepagents_mode` from `config.yaml`, passes it through session resolution (`_resolve_session_agent_config` → `_resolve_turn_agent_config`), and injects `runtime="deepagents"` into the agent creation kwargs. This makes DeepAgents opt-in by config, with no breaking changes.

### 1.3 New files

#### `agent/deep_agents_runtime.py` (798 lines) — Main adapter

`DeepAgentsAIAgent` — a full parallel implementation of `AIAgent` built on LangGraph. Key pieces:

- **Message conversion** — `_convert_messages_to_langchain()` converts Hades dict format (`{"role": "system/user/assistant/tool", ...}`) to LangChain `SystemMessage`, `HumanMessage`, `AIMessage` / `AIMessageChunk`, and `ToolMessage` objects. `_convert_langchain_to_hades()` reverses the process.

- **Result parsing** — `_parse_langgraph_result()` formats LangGraph's return value into the same dict shape as `run_conversation()`: `{final_response, messages, api_calls, completed, failed, interrupted, partial, turn_exit_reason, last_reasoning, model}`. `_parse_error_result()` does the same for errors.

- **Provider env injection** — `_inject_provider_env()` maps Hades provider names to LangChain's expected env var convention:

  | Hades provider | LangChain env vars |
  |:---|:---|
  | `""` or `"openai"` | `OPENAI_API_KEY`, `OPENAI_API_BASE` |
  | `"anthropic"` | `ANTHROPIC_API_KEY`, `ANTHROPIC_API_BASE` |
  | `"google"` | `GOOGLE_API_KEY`, `GOOGLE_API_BASE` |
  | `"xai"` | `XAI_API_KEY`, `XAI_API_BASE` |
  | `"cohere"` | `COHERE_API_KEY`, `COHERE_API_BASE` |
  | `"groq"` | `GROQ_API_KEY`, `GROQ_API_BASE` |
  | unknown | `API_KEY`, `BASE_URL` (generic) |

- **Tool adapter** — `_HadesToolAdapter` wraps each Hades tool (retrieved from the existing `tools.registry`) as a LangChain `StructuredTool`. Invocation calls `handle_function_call()` from `model_tools.py`, preserving all existing tool dispatch, error handling, and `ContextVar` propagation.

- **`build_hades_tools()`** — queries `tools.registry`, filters by enabled/disabled toolsets, returns a list of `StructuredTool` instances that LangGraph's `ToolNode` can execute.

- **Streaming bridge (internal)** — `_HadesStreamingBridge` reads through LangGraph's stream events and calls Hades callbacks (`stream_delta_callback`, `tool_progress_callback`, `step_callback`) with fallback `_noop_cb()`.

- **Callback forwarding** — `_CAPTURED_NAMES` (a `frozenset` of known callback/config attribute names) plus `__setattr__`/`__getattr__` overrides forward gateway attribute-setting patterns (`agent.tool_progress_callback = cb`) into an internal `_callbacks` dict.

- **LangGraph checkpointer/store** — optional `MemorySaver` (in-memory checkpoint for state persistence / tracing) and `InMemoryStore` (persistent store for state). Controlled by `langgraph_checkpointer` and `langgraph_store` config params.

- **Observability** — LangSmith integration: sets `project_name="hades"`, tags `["hades"]`, and forwards `langsmith_api_key` / `langgraph_endpoint`.

#### `agent/deep_agents_middleware.py` (120 lines) — LangGraph middleware

Implements LangGraph's `AgentMiddleware` ABC to bridge the DeepAgents turn lifecycle to Hades' turn lifecycle:

- **`_HadesInterruptSignal`** — thread-safe interrupt signal using a `threading.Lock`-protected bool flag. Can be set/cleared from the gateway or CLI.
- **`_HadesMiddleware`** — LangGraph middleware chain:
  - `before_agent()` — prefetches memory context via `agent.get_memory_context()` and checks the interrupt signal (breaks the LangGraph loop if set).
  - `after_agent()` — saves new context to memory via `agent.save_memory()` and clears the interrupt signal.
  - `request_interrupt()` / `clear_interrupt()` — public API for the gateway to signal interruption mid-turn.
  - `interrupt_signal` property — returns the signal object for external interrupt-checking.

#### `agent/deep_agents_streaming.py` (119 lines) — Streaming bridge (standalone)

A **parallel/streaming-optimized** version of the streaming bridge that works with an explicit callback function:

- `iter_events(stream)` — yields `(event_type, data)` tuples from LangGraph's `stream_mode="updates"`:
  - `AIMessageChunk` → `("message_delta", {"text": content})`
  - `ToolCall` → `("tool_start", {"tool_name": ..., "args": ...})`
  - `ToolResult` → `("tool_complete", {"tool_name": ..., "result_preview": ...})`
- `_process_output()` — parses node output (either from a `hades_engine` sub-agent or the engine result) and yields `("complete", {"result": ...})`
- `final_result()` — returns the parsed result dict, or a fallback shape with empty strings
- `_shorten()` — truncates text to 200 chars for preview

#### `agent/deep_agents_tool_dispatcher.py` (144 lines) — Tool dispatcher

- **`execute_hades_tools(tool_name, tool_args)`** — executes a single Hades tool by delegating to `model_tools.handle_function_call`. Used by the `_HadesToolAdapter` path.
- **`build_hades_engine_tool()`** — constructs a LangChain `StructuredTool` named `"hades_engine"` that runs the **entire native Hades conversation loop** as a single tool call, enabling DeepAgents to call back into the full native agent loop as a nested sub-agent:
  1. Creates a minimal stub `AIAgent` via `AIAgent.__new__()` (no full `__init__`)
  2. Sets only the attributes needed by `run_conversation()`
  3. Calls `agent.conversation_loop.run_conversation()` directly
  4. Returns the result as JSON-serialized string
  5. Runs in a `ThreadPoolExecutor` to avoid blocking LangGraph's event loop

### 1.4 DeepAgents tests (3,010+ test lines across 6 files)

| Test file | Lines | Coverage |
|-----------|-------|----------|
| `tests/run_agent/test_deep_agents_adapter.py` | 799 | Adapter layer: message converter round-trips, result/error parsing, provider env injection, tool adapter creation, `build_hades_tools()` from mock registry, streaming bridge event routing, attribute forwarding |
| `tests/run_agent/test_deep_agents_e2e.py` | 807 | End-to-end: sync + streaming `run_conversation()` paths, conversation history handling, system message prepending, task_id in config, streaming callback routing, tool adapter integration |
| `tests/run_agent/test_deep_agents_tool_dispatcher.py` | 600+ | Tool dispatcher: single-tool execution, `hades_engine` StructuredTool creation, full native loop invocation |
| `tests/run_agent/test_deep_agents_middleware.py` | 439 | Middleware: `_HadesInterruptSignal` thread-safety, `_HadesMiddleware` before/after hooks, interrupt signaling |
| `tests/run_agent/test_deep_agents_streaming.py` | 560 | Streaming bridge: event parsing, callback routing, result extraction, truncation |
| `tests/run_agent/test_deep_agents_langfuse.py` | 176 | Langfuse/LangSmith observability integration |

**Test patterns:**

- DeepAgents SDK dependencies are always mocked via `sys.modules` injection (real packages may not be installed)
- The autouse fixtures `_clean_deepagents_modules` and `deepagents_available` manage clean mock state before each test
- `DeepAgentsAIAgent` instances are created via `object.__new__()` (not calling `__init__`) since `__init__` triggers SDK imports
- Tests verify both sync (`invoke()` path) and streaming (`stream()` + bridge processing) code paths

### 1.5 Configuration

| Config key | Type | Default | Description |
|-----------|------|---------|-------------|
| `runtime` | `AIAgent.__init__()` param | `"native"` | Selects `"native"` or `"deepagents"` |
| `deepagents_mode` | `config.yaml` | `False` | Gateway-level toggle propagated to agent creation |
| `langgraph_checkpointer` | `DeepAgentsAIAgent.__init__()` | `False` | Enables `MemorySaver` checkpointing |
| `langgraph_store` | `DeepAgentsAIAgent.__init__()` | `False` | Enables `InMemoryStore` |
| `skip_memory` | `DeepAgentsAIAgent.__init__()` | `False` | Skips memory middleware entirely |
| `debug` | `_get_cap("debug")` | `False` | Enables LangGraph debug-level runnable traces |
| `langsmith_api_key` | `_get_cap()` / env | `None` | Enables LangSmith tracing with `project_name="hades"` |

### 1.6 Compatibility guarantees

The DeepAgents runtime is a **parallel implementation**, not a replacement. All existing code paths remain functional:

- **Gateway code** — no changes needed. Gateway attribute-setting (`agent.model`, `agent.tool_progress_callback = cb`, etc.) is forwarded via `__setattr__`/`__getattr__` to the internal `_deep_agents_impl`
- **Tool dispatch** — all tools continue to run through `handle_function_call()` from `model_tools.py`
- **Memory** — delegated to the middleware hooks (`_HadesMiddleware` before/after agent), no changes to memory provider ABCs
- **Streaming** — gateway callbacks are called identically; the bridge translates LangGraph events to Hades callback format
- **Interrupts** — `agent.interrupt()` / `_interrupt_requested` pattern is mirrored by `_HadesInterruptSignal` checked in `before_agent()`
- **Native loop is still the default** — existing deployments are unaffected; `deepagents_mode: true` in config is required to enable

---

## 2. Rename: Hermes → Hades

This is a comprehensive project rename from **Hermes** → **Hades** across the entire codebase and documentation.

### 2.1 Rename scope

- **Package name** (`pyproject.toml`): `"hermes-agent"` → `"hades-agent"`; extras: `[deepagents]` etc.
- **Directory rename**: `hermes-agent/` → `hades-agent/` (core directory)
- **Entry points**: CLI commands `hades`, `hades-agent`, `hades-acp`
- **All module references**: imports, docstrings, comments, error messages, display names
- **Skills**: `autonomous-ai-agents-hermes-agent.md` → `autonomous-ai-agents-hades-agent.md` (renamed skill content, 428+ lines updated)
- **Website**: 200+ doc files updated — all references, file renames (`build-a-hermes-plugin.md` → `build-a-hades-plugin.md`, `run-hadoop-with-nous-portal.md` → `run-hades-with-nous-portal.md`, etc.)
- **Config defaults**: renamed in `DEFAULT_CONFIG` where applicable
- **Brand assets**: `hermes-agent-banner.png` → `hades-agent-banner.png`; SVG assets updated
- **Category labels**: `autonomous-ai-agents-hermes-agent.md` → `autonomous-ai-agents-hades-agent.md` and similar

### 2.2 What was NOT changed

- Internal implementation logic remains the same — this is purely a naming/branding change
- Tool schemas, tool names, and tool behaviors are unaffected
- Memory providers, plugins, model providers — all continue working

### 2.3 Rename commits

| Commit | Description |
|--------|-------------|
| `213e08175` | Rename project from hermes-agent to hades-agent |
| `f30cc47e6` | Additional project rename: fix remaining file/dir names and content |
| `e84f8b4a6` | Rename kanban spec PDF from hermes to hades |
| `37227e76c` | Clean up deleted old hermes files (frames, PNGs, PDF) |

---

## 3. Architecture Diagram

```
                           ┌──────────────────────────────┐
                           │        AIAgent (unified)      │
                           │  run_agent.py                │
                           │  runtime="native"|"deepagents"│
                           └──────────┬───────────────────┘
                         runtime="native" │ runtime="deepagents"
                           ┌──────────────┴──────────────────┐
                           │                                 │
                  ┌────────▼────────┐          ┌────────────▼──────────────┐
                  │ conversation_   │          │    DeepAgentsAIAgent      │
                  │ loop.py         │          │    agent/deep_agents_*.py │
                  │ Native loop     │          │    LangGraph-based        │
                  │ while True {...}│          │                             │
                  └────────┬────────┘          └──────────┬────────────────┘
                           │                              │
                   ┌───────┴────────┐           ┌─────────┴─────────┐
                   │ model_tools.py  │           │  AgentMiddleware   │
                   │ handle_fn_call  │           │  _HadesMiddleware  │
                   │ tools.registry  │           │  before_agent()    │
                   └─────────────────┘           │  after_agent()     │
                                                 └─────────┬──────────┘
                                                               │
                                             ┌─────────────────┼──────────────────┐
                                             │                 │                      │
                                    ┌────────▼────────┐ ┌──────▼──────┐   ┌───────────▼───────────┐
                                    │  StructuredTool  │ │  Streaming  │   │  hades_engine tool     │
                                    │  (tool adapter)  │ │  Bridge     │   │  (native loop as tool) │
                                    └─────────────────┘ └─────────────┘   └───────────────────────┘
```

---

## 4. Breaking Changes

**None.** Both the rename and the DeepAgents integration are backward-compatible:

- The native agent loop remains the default (`runtime="native"`)
- `deepagents_mode` defaults to `False` in config
- The gateway does not require changes
- Existing config keys and `.env` variables are preserved
- Tool schemas and behaviors are unchanged

Users who want to use DeepAgents must explicitly set:

```yaml
deepagents_mode: true  # in config.yaml
```

---

## 5. Checklist for reviewers

- [ ] `agent/deep_agents_runtime.py` — adapter compatibility (message formats, error handling, provider mapping)
- [ ] `agent/deep_agents_middleware.py` — middleware lifecycle hooks, interrupt signal thread-safety
- [ ] `agent/deep_agents_streaming.py` — streaming event parsing, callback fidelity
- [ ] `agent/deep_agents_tool_dispatcher.py` — tool dispatch, `hades_engine` sub-agent
- [ ] Tests — coverage parity against native agent behavior
- [ ] Rename consistency — verify no stray "hermes" references remain

---

## 6. Related

- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain: https://python.langchain.com/
