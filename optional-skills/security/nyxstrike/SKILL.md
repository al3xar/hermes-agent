---
name: nyxstrike
description: |
  Operate the NyxStrike offensive-security MCP server (185+ tools) end-to-end
  under Hades' PLAN-AND-APPROVE model: recon → enumeration → web exploitation →
  foothold → post-exploitation → evidence-based report. Covers how to invoke the
  mcp_nyxstrike_* tools, which tool to pick per attack phase (PTES / OWASP WSTG),
  how to chain findings through memory, scope guardrails, and async long-scan
  management. Use for authorized cyber-range / lab targets only.
version: 1.0.0
author: Hades Agent
license: MIT
platforms: [linux]
category: security
triggers:
  - "audit [target] with nyxstrike"
  - "run the attack chain against [target]"
  - "pentest the blue cluster"
  - "recon [target]"
  - "scan [target] for vulns"
  - "exploit [service/CVE] on [target]"
  - "use nyxstrike to ..."
toolsets:
  - nyxstrike
  - terminal
  - file
  - skills
  - todo
  - delegation
metadata:
  hermes:
    tags: [security, red-team, pentesting, mcp, nyxstrike, recon, exploitation, cyber-range]
    related_skills: [web-pentest, oss-forensics]
---

# NyxStrike — Offensive MCP Toolbox (Hades operator guide)

## Overview

NyxStrike is the offensive-security MCP server that is *your* toolbox. It exposes
185+ real tools (recon, web exploitation, network, cloud/API, password, OSINT,
binary) plus an **Intelligent Decision Engine** that proposes the next
`AttackStep`. The `nyxstrike_mcp.py` bridge connects Hades to it over HTTP; all
its tools land in your context **prefixed `mcp_nyxstrike_`** (e.g.
`mcp_nyxstrike_nmap_scan`, `mcp_nyxstrike_run_tool`).

You run it under a **PLAN-AND-APPROVE** model (Hades personality): the engine
proposes a step (tool + params + rationale); **you evaluate it** (does it advance
the goal? better pivot? new branch?), approve / adjust / reject, interpret the
result, and **persist findings to memory** so you can chain them. Do not improvise
raw shell when an MCP tool exists for the task.

Connection (from `.hermes-config/config.yaml`): bridge `nyxstrike_mcp.py
--server $NYXSTRIKE_URL --profile full --auth-token $NYXSTRIKE_API_TOKEN`,
300s timeout. In K8s the bridge runs as a sidecar of the Hades pod; the NyxStrike
server lives in the DMZ (`nyxstrike.nyx.svc:8888`).

## When to Use

- Auditing a target inside the **declared lab scope** (e.g. the cyber-range Blue
  cluster workload) — full chain recon→exploit→report.
- Any single offensive step where a NyxStrike tool fits: port scan, subdomain
  enum, web vuln scan, directory brute, parameter discovery, exploit search,
  payload generation, cloud/API audit, password cracking.

**Don't use for:**
- Targets you cannot prove are in scope. Refuse off-scope hosts — no exception.
- Tasks with no NyxStrike tool — only then drop to `terminal`.
- Defensive/blue-team analysis of the captured evidence (that's the White-side
  Wazuh workflow, not this skill).

## ⚠️ Hard Guardrails

1. **Scope gate.** Maintain the in-scope target list. Every tool invocation must
   target an in-scope host/URL/CIDR. If a result points off-scope (3xx to another
   host, a discovered external dependency, cloud metadata `169.254.169.254`),
   STOP and confirm before pivoting. Pivoting off-scope is what makes it illegal.
2. **No exploit, no report.** Every finding needs reproducible runtime evidence —
   request/response, command + output, screenshot. L1 "pattern matched" is a
   candidate, not a finding.
3. **Destructive payloads need approval.** SQLi `DROP/DELETE`, command injection
   with `rm`/`shutdown`, anything mutating beyond a single test row → ask first.
4. **Credential / token hygiene.** Captured secrets go to memory + evidence
   files. In chat history redact to the last 6 chars (Hermes' compression path
   can replay history through the aux client).
5. **Stop condition.** Stop and summarize when the objective is met (foothold /
   host compromised / range owned) or scope is exhausted.

## Two ways to call tools

NyxStrike exposes both **named tools** and a generic **orchestrator**:

```python
# Named tool — preferred, schema is explicit
mcp_nyxstrike_nmap_scan(target="10.20.0.5", ports="1-1000", flags="-sV -sC")

# Generic orchestrator — for any tool in the registry, incl. ones without a
# dedicated wrapper (sqlmap, hydra, metasploit, msfvenom, exploit_db, ...)
mcp_nyxstrike_run_tool(tool="sqlmap", target="http://10.20.0.5/item?id=1",
                       flags="--batch --dbs")

# Let the decision engine pick + chain tools for a target
mcp_nyxstrike_intelligent_smart_scan(target="10.20.0.5")
```

Prefer the **named tool** when one exists (clearer params, validated). Fall back
to `run_tool` for registry tools without a wrapper. Use `execute_command` only
when no registered tool covers the action.

## Attack chain → tool map (PTES / OWASP WSTG)

### Phase 1 — Recon (read-only, scope-bounded)

| Goal | NyxStrike tool |
|---|---|
| Fast port discovery | `masscan_high_speed`, `run_tool(tool="rustscan")` |
| Service/version/NSE | `nmap_scan`, `nmap_advanced_scan` |
| Local subnet hosts | `arp_scan_discovery` |
| Subdomains / assets | `subfinder_scan`, `amass_scan`, `assetfinder_scan`, `shuffledns_scan` |
| DNS | `dnsenum_scan`, `dig_dns`, `fierce_scan` |
| OSINT | `theharvester_scan`, `spiderfoot`, `sherlock` |
| HTTP probe / fingerprint | `httpx_probe`, `whatweb_analyze`, `check_http_headers` |
| Auto / smart sweep | `autorecon_scan`, `bbot_scan`, `intelligent_smart_scan` |

Chain results: feed open ports from masscan/rustscan straight into `nmap_scan`
to avoid redundant full scans.

### Phase 2 — Web enumeration & vuln analysis

| Goal | NyxStrike tool |
|---|---|
| WAF detection | `wafw00f_scan` |
| Content/dir discovery | `ffuf_scan`, `feroxbuster_scan`, `gobuster_scan`, `dirsearch_scan` |
| URL harvesting | `gau_discovery`, `waybackurls_discovery`, `gospider_crawl`, `hakrawler_crawl` |
| Parameter discovery | `arjun_scan`, `paramspider_mining`, `x8_parameter_discovery` |
| Vuln scanning | `nuclei_scan`, `nikto_scan`, `zap_scan`, `jaeles_vulnerability_scan` |
| TLS | `testssl_analyze` |
| API audit | `run_tool(tool="...")` from the api_scan / api_fuzz arsenal |

Pick the best wordlist first: `wordlist_find_best(...)` then pass its path.

### Phase 3 — Exploitation (proof-based, conditional)

| Goal | NyxStrike tool |
|---|---|
| Search public exploits | `run_tool(tool="exploit_db", query="<svc> <ver>")` |
| Run exploit/aux/post module | `run_tool(tool="metasploit", module=..., options={...})` |
| Generate standalone payload | `run_tool(tool="msfvenom", payload=..., format=..., lhost=..., lport=...)` |
| SQLi | `run_tool(tool="sqlmap", ...)` |
| Network service / creds | `netexec_scan`, `smb_enum`, `run_tool(tool="hydra", ...)` |
| Cloud exploitation | `pacu_exploitation` |

Always start a listener (`exploit/multi/handler`) before delivering a payload.
Fire the minimal **witness** payload first, confirm behavior change, then escalate.

### Phase 4 — Post-exploitation & cloud/API

`pacu_exploitation`, cloud_audit tooling (Prowler via `run_tool`), credential
harvest / runtime monitor tools. Privilege escalation, lateral movement,
credential reuse, pivoting — all within scope.

### Phase 5 — Reporting

| Goal | NyxStrike tool |
|---|---|
| Per-scan summary | `create_scan_summary` |
| Vulnerability report | `create_vulnerability_report` |
| Vuln intelligence view | `vulnerability_intelligence_dashboard` |
| Live progress | `get_live_dashboard` |

The NyxStrike attack report is **pillar 5** of the TFM — it is later combined
with White-side Wazuh alerts into the final audit report. Capture enough evidence
(tool, params, output, timestamps) for that correlation.

## Long scans are async — manage processes

Big scans (full-port nmap, masscan on a CIDR, nuclei) run as background
processes. Don't block; poll.

```python
mcp_nyxstrike_list_active_processes()
mcp_nyxstrike_get_process_status(pid_or_id)
mcp_nyxstrike_get_process_dashboard()
mcp_nyxstrike_pause_process(id) / resume_process(id) / terminate_process(id)
```

Kick off a scan, persist its id to memory, work another branch, come back.

## Chaining findings through memory

After each step, persist actionable state (in-scope hosts, open ports, creds,
secrets, tokens, confirmed vulns) to memory so subsequent steps can use it. The
`memory` plugin is enabled exactly so the attack chain survives across turns.
Treat memory as the shared state of the engagement — the decision engine and you
both read from it to choose the next `AttackStep`.

## Common Pitfalls

1. **Calling tools without the prefix.** They are `mcp_nyxstrike_<tool>`, not
   `<tool>`. A bare `nmap_scan(...)` won't resolve.
2. **Reaching for `terminal` first.** Hades' rule: don't improvise raw shell when
   an MCP tool exists. Check the arsenal (named tool or `run_tool`) before
   `execute_command`.
3. **Blocking on a long scan.** Full-port / CIDR / nuclei scans are async —
   poll with the process tools instead of waiting on one call (300s bridge
   timeout will bite you otherwise).
4. **Leaking secrets into chat.** Redact captured creds/tokens to last 6 chars in
   messages; full values to memory/evidence files only.
5. **Treating a clean scan as "secure."** Report "no exploitable issues FOUND in
   scope X within time T using methods Y" — not a PASS.
6. **Pivoting off-scope on an interesting result.** Document it, stop, confirm.
   Never follow a redirect/dependency outside the declared lab scope.
7. **Skipping evidence.** No exploit, no report — an L1 pattern match is a
   candidate; promote only with reproducible runtime proof.

## Verification Checklist

- [ ] Target confirmed in the declared lab scope before the first active request
- [ ] Used a named `mcp_nyxstrike_*` tool where one exists; `run_tool` only for
      unwrapped registry tools; `execute_command` only as last resort
- [ ] Long scans launched async; process id tracked; polled via process tools
- [ ] Findings (hosts, ports, creds, vulns) persisted to memory for chaining
- [ ] Secrets redacted to last 6 chars in chat; full values in evidence/memory
- [ ] Every reported finding has reproducible runtime evidence (no exploit, no
      report)
- [ ] Report generated via `create_vulnerability_report` / `create_scan_summary`
      with enough detail to correlate against Wazuh evidence (TFM pillar 5)
