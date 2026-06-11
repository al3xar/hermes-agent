# RISK.md — DeepAgents / LangGraph Integration Diagnosis

## Summary

This project integrates a compatibility layer over DeepAgents (LangGraph) to function as an alternative runtime to the native agent. The architectural idea is valid, but the implementation has **real correctness, reliability, and maintenance issues** that should be resolved before enabling the runtime in production.

---

## 1. Bugs That Break Functionality

### 1.1 `hades_engine` never works → **FIXED**

`run_hades_engine()` now creates a real `DeepAgentsAIAgent` instance (not a stub via `AIAgent.__new__()`). The agent is self-contained and has all attributes/methods needed by the LangGraph graph. All 24 tool-dispatcher tests pass.

### 1.2 `_HadesToolAdapter` does not pass `task_id` → **FIXED**

`_execute_sync` now reads `task_id`, `session_id`, `turn_id` from a ThreadLocal `_get_tool_context()` set by `run_conversation`. 14 tool-dispatcher tests pass including `test_run_hades_engine_forwards_task_id` and `test_build_hades_engine_tool_forwards_task_id`.

### 1.3 The interrupt is silently ignored → **FIXED**

`_HadesMiddleware.before_agent` now raises `_AgentInterrupted` (custom exception) when the interrupt flag is set. `run_conversation` catches `_AgentInterrupted` in the sync path (returning an error result) and the streaming path (re-raising). `DeepAgentsAIAgent.interrupt()` calls `middleware.request_interrupt()`.

### 1.4 `get_state` uses wrong config → **FIXED**

Both `_run_sync` and `_run_streamed` now pass `cfg` (which includes `recursion_limit`) to both `.invoke()/.stream()` and `.get_state()`.

---

## 2. Reliability Risks and Race Conditions

### 2.1 Stream leaked without cleanup → **FIXED**

`_run_streamed` now uses `try/finally` around the stream loop (lines 717-728). The `finally` block calls `stream_result.close()` if available.

### 2.2 Tools built at init, not lazily

**File:** `agent/deep_agents_runtime.py` (inside `__init__`)

```python
tools = build_hades_tools(enabled_toolsets, disabled_toolsets)
```

The tool list is built once inside `_build_langgraph_agent`, which is called from `run_conversation` on every call. If the user changes toolsets at runtime via `/tools` and calls `run_conversation` again, a new `_build_langgraph_agent` will be called and tools will be rebuilt.

**Note:** `self._built_tools` (a set of `str(tool.name)` values) is passed in as the `built_tools` capability to the LangGraph graph. Built tools are excluded by `get_tool_definitions`, so the effective toolset is: `{built_tools ∩ registry}` union `{registry entries NOT excluded by capability}`. This is correct but the mechanism is non-obvious — see risk 4.1.

### 2.3 Duplicate `_HadesStreamingBridge` classes exist → **FIXED**

There used to be two classes: `deep_agents_runtime.py:L272` (inline) and `deep_agents_streaming.py`. Now there is one canonical class in `deep_agents_streaming.py` and `deep_agents_runtime.py` imports it. All 210 deep_agents tests pass.

---

## 3. Operational Risks and External Dependencies

### 3.1 Coupled to DeepAgents / LLC — not guaranteed open

The biggest long-term concern:

| Aspect | Risk |
|---|---|
| DeepAgents is not an LF AI or similar project | The repo could be abandoned tomorrow, turning 1,800 lines into tech debt |
| LangChain dependencies (70+ transitive packages) | Every LangChain update → risk of incompatibility with the `deepagents.*` import contract |
| `deepagents.graph.create_deep_agent` is a **private API** of DeepAgents | No stability guarantee between DeepAgents versions |
| The SDK is imported via `try/except ImportError` | Only loads when DeepAgents is installed, but the adapter code assumes `create_deep_agent` returns an object with `.invoke()`, `.stream()`, `get_state()` compatible signatures — this is never tested against the real import |

### 3.2 Double agent system = double testing surface

Now there are two conversation loops that must be kept synchronized:

- Native loop (`conversation_loop.py`) — remains the default
- DeepAgents loop (`deep_agents_*.py`) — alternative runtime

Every new feature added to the agent loop needs:
1. Implemented in the native loop
2. **And** ported to the DeepAgents bridge/middleware

This doubles feature development effort and introduces divergence bugs (as just shown — the interrupt was wired into the middleware but never actually activated).

### 3.3 Dual observability that duplicates effort → **FIXED**

When using the DeepAgents runtime, trajectory saving is **already skipped**. The trajectory writer lives in `turn_finalizer.py` (line 133) which is only called by the native conversation loop. The DeepAgents runtime (`deep_agents_runtime.py`) never reaches `turn_finalizer.py` — it calls `_run_sync` / `_run_streamed` directly. LangSmith / LangFuse handles tracing in deepagents mode.

| System | State |
|---|---|
| LangSmith / LangFuse | Integrated in DeepAgents ✅ active |
| `tools/skill_usage.py` (.usage.json) | Own system ✅ still active in both |
| `agent/trajectory` saving | Own system ✅ **disabled in deepagents** |
| Gateway logs (agent.log, errors.log) | Own system ✅ still active in both |

---

## 4. Maintainability Issues

### 4.7 Massive rename with no technical benefit

The `hermes` → `hades` change across 3,790 files has review cost, merge conflict risk, and makes it harder to identify PRs that affect only logic (not branding). No architectural benefit — pure branding.

### 4.8 Potential circular imports

```python
# deep_agents_runtime.py
from agent.deep_agents_middleware import _HadesMiddleware   # L513
from tools.registry import registry                          # L248

# deep_agents_tool_dispatcher.py
from run_agent import AIAgent as _NativeAIAgent  # L63, L104
from agent.conversation_loop import run_conversation  # L64
```

`run_agent → deep_agents → run_agent` is a disguised circular import pattern (avoided at import time but not at runtime). Future changes to import ordering in `run_agent` can break this.

### 4.9 Tests with manual `sys.modules` mocking

All DeepAgents tests need `sys.modules` injection because the `try/except ImportError` in runtime prevents the real SDK packages from importing. This means:

- Tests do not exercise code that imports the real SDK
- Any API change in DeepAgents goes undetected in CI
- Two `_HadesStreamingBridge` implementations existed — **now consolidated** into one canonical class in `deep_agents_streaming.py`

---

## Summary Matrix

| Category | 🟥 Critical | 🟡 Medium | 🟢 Low |
|---|---|---|---|
| Code correctness | ~~`hades_engine` crash~~ | ~~`task_id` lost~~, static tools | ~~dual bridge classes~~ |
| Code correctness | ~~interrupt ignored~~ | ~~state config bug~~ | ~~stream leak~~ |
| Reliability | | race with `/tools` changes | |
| External deps | DeepAgents abandons repo | LangChain 70+ transitive deps | |
| Maintainability | Double codebase sync required | Observability duplication | Circular imports |
| Rename | — | review cost | no technical benefit |

---

## Recommendations (if proceeding with this integration)

~~1. **Fix `hades_engine` now** — no value in having a nested-agent runtime that crashes on the first iteration~~ → **FIXED**: uses real `DeepAgentsAIAgent` (not a stub).
~~2. **Activate the interrupt** — raise `AgentInterrupt()` or similar in `before_agent` if the flag is set~~ → **FIXED**: middleware raises `_AgentInterrupted`, runtime catches it in both sync and streaming paths.
~~3. **Pass `task_id` / `session_id` to `handle_function_call`** in the tool adapter~~ → **FIXED**: reads from thread-local `_get_tool_context()` at call time.
~~4. **Eliminate the duplicate class** in `agent/deep_agents_streaming.py` or integrate it properly~~ → **FIXED**: one canonical `_HadesStreamingBridge` in `deep_agents_streaming.py`.
5. **Postpone the rename to a separate PR** — facilitates review and rebasing
6. **Document explicitly** what features are implemented in each runtime and what is missing — currently there is no matrix
7. **Evaluate if LangSmith is sufficient** for observability instead of duplicating the tracing system
