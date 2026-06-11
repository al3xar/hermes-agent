# TODO.md — DeepAgents remaining work

## Critical (already done)
- [x] 1.1 `hades_engine` crashes → uses real DeepAgentsAIAgent
- [x] 1.2 `task_id` not passed → ThreadLocal `_get_tool_context()`
- [x] 1.3 interrupt ignored → middleware raises `_AgentInterrupted`
- [x] 1.4 `get_state` wrong config → uses `cfg` with `recursion_limit`
- [x] 2.1 stream leak → `try/finally` + `close()`
- [x] 2.3 duplicate `\HadesStreamingBridge` → one canonical class

## Still Open

### 3.x delegate_task → deep agents children (new)
**File:** `tools/delegate_tool.py:1143,1258`
- `delegate_task` now spawns `DeepAgentsAIAgent` children when parent is a deep agents agent
- Keeps nested agents on same execution path (LangGraph middleware, tool runtime, memory model)
- `_build_deep_child_agent()` at line 1258 builds child with same params as native but deep agents constructor
- Deep agents children share parent `_session_id`, skip memory, use provider env vars for auth
- Interrupt propagation via `_interrupt_requested` still works (gate: `hasattr(child, "interrupt")` or `hasattr(child, "_interrupt_requested")`)
- **Status:** Implemented, 210 tests pass

### 2.2 Tools built at init, not lazily
**File:** `agent/deep_agents_runtime.py:424,509`
- `_build_langgraph_agent` called from `__init__` (line 424)
- `build_hades_tools` called inside this method (line 509)
- Tools built once per agent instance, not per-conversation
- When gateway changes toolsets via `/tools`, it creates a NEW agent anyway, so the practical impact is low
- **Status:** Low risk but documented for transparency

### 3.3 Dual observability — trajectory disabled in deepagents ✅
- `turn_finalizer.py:133` (`agent._save_trajectory`) es exclusivo del agent loop nativo
- `deep_agents_runtime.py` nunca llega a `turn_finalizer.py` — usa `_run_sync` / `_run_streamed` directo
- Deepagents: solo LangSmith/LangFuse做tracing. No duplicación de trajectory.
- **Status:** Ya está correcto, no necesita cambios

### 3.x Operational risks (architectural, not fixable without rewriting)
- 3.1 Coupled to DeepAgents / LangChain — third-party risk
- 3.2 Double agent system — maintenance burden
- 3.3 Dual observability — trajectory disabled in deepagents ✅

### 4.x Maintainability (structural decisions)
- 4.7 Massive rename — branding only
- 4.8 Circular imports — import ordering risk
- 4.9 Test mocking — sys.modules hacks for imports

### Recommendations
5. Postpone rename → separate PR
6. Document feature matrix per runtime
7. Evaluate LangSmith for observability ✅ default in deepagents

## Priority
1. Verify all 210+ tests still pass after consolidation
2. Add feature parity check between native and deepagents runtimes
3. Consider if deepagents integration should remain experimental
