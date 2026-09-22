---
name: cyber-range-audit
description: "Drive a phased cyber-range audit via plan-and-approve."
version: 0.1.0
author: Al3xar (al3xar), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
category: security
triggers:
  - "cyber-range audit [segment]"
  - "audit segment [CIDR]"
  - "run a phased pentest on [segment]"
  - "red-team the range [CIDR]"
  - "plan-and-approve audit [segment]"
toolsets:
  - nyxstrike
  - skills
  - todo
metadata:
  hermes:
    tags: [Security, Cyber-Range, Red-Team, Pentest, Plan-And-Approve]
    related_skills: [web-pentest]
---

# Cyber-Range Audit (phased, plan-and-approve)

Supervised, evidence-backed audit of one authorized cyber-range network segment,
driven through the NyxStrike **plan-and-approve** contract. You (Hades, the LLM
supervisor) do not improvise commands — NyxStrike builds the deterministic
`AttackChain`; you approve / reject / reorient one step at a time and interpret
each result. The skill's job is to keep the audit moving in **phase order** and
to keep the guardrails enforced, so the run is traceable and reproducible.

It does **not** replace NyxStrike, and it does **not** invent the plan: the phase
order below is fixed, and the per-step choice of tool is NyxStrike's (grounded in
its own tool catalog). You supervise; the deterministic engine plans.

The four attack phases plus reporting:

1. **recon** — discover live hosts, ports, services and web technology in scope.
2. **web-exploit** — OWASP Top 10 against every web app found, proof-based.
3. **foothold** — initial access / credentials / a shell.
4. **post-exploitation** — privilege escalation and lateral movement.
5. **report** — `chain_report` with ATT&CK tagging (derived, not invented).

---

## Hard Guardrails (non-negotiable — every phase obeys them)

1. **Scope.** Attack **only** hosts inside the mission's CIDR segment. Do not
   assume a host or an application — discover what is live in the segment first,
   then audit whatever you find. Ignore any host outside the segment and anything
   you recall from memory, other skills, or past runs. **Never invent hosts.**
2. **Tooling.** Use **only** the NyxStrike MCP tools for discovery, scanning and
   exploitation — this run is **invalid** if you bypass NyxStrike (no local
   `terminal`, `nmap`, `sqlmap`, `curl`). The local terminal is only for reading
   and writing your own notes and report.
3. **No exploit, no report.** Every confirmed finding needs reproducible
   evidence: the tool used, the exact request, the response that proves it, the
   severity and the remediation.
4. **Plan-and-approve, not free-run.** `propose_next_step` only *proposes*;
   nothing runs until you call `execute_step`. Evaluate every proposed step
   (does it add value? is there a better pivot? does it open a new branch?)
   before approving — approving everything degenerates you into a deterministic
   orchestrator.
5. **Redact secrets** (captured tokens/credentials) to the last 6 chars in any
   message; full values stay in report files, never in chat history.

---

## When to Use

- A mission gives you a **network segment** (a CIDR) to audit end-to-end in an
  authorized cyber-range lab.
- You must drive the NyxStrike plan-and-approve contract across ordered phases.
- You need a traceable, phase-disciplined audit (not ad-hoc command runs).

**Don't use for:** a single target / single app — that is the `web-pentest`
skill. Don't use for anything outside the authorized segment, or against
production / third-party systems.

## Prerequisites

- The NyxStrike MCP server is reachable and its **plan-and-approve** tools are
  registered: `profile_target`, `propose_next_step`, `execute_step`,
  `update_chain`, `chain_report` (the 5-tool contract from cap. 8).
- You have the mission: the target **CIDR segment**, the **objective**
  (`comprehensive`, `quick`, or `stealth`), and the **difficulty level**
  (`Level-0` by default; `Level-1` / `Level-2` only if the mission asks for it).
- The generic NyxStrike skills for each phase are available as the tooling the
  Decision Engine picks (`web-recon`, `nmap-recon`, `web-vuln`,
  `subdomain-enum`, `exploitation`, `cloud-audit`, `password-cracking`,
  `binary-analysis`, `smb-enum`) — you do not call their binaries, you let
  NyxStrike route to them.

## How to Run

Start the session once, then loop `propose → evaluate → execute → record` until
the chain is exhausted and the report is written.

```
profile_target(target=<segment-cidr>, objective=<obj>, difficulty=<level>)   # -> session_id + AttackChain
loop:
    step = propose_next_step(session_id)
    if step.completed: break
    evaluate phase + guardrails (below); if wrong, update_chain(action=...)
    result = execute_step(session_id, step_index=<i>, params=<json>)   # the "approve"
    record evidence to memory + notes
report = chain_report(session_id, include_attack=True)
write report -> reports/audit.md   # then stop
```

## Difficulty Levels (a parameter of this skill — not separate profiles)

The **difficulty level is a parameter of the mission passed to
`profile_target`** (`difficulty=<level>`); it changes how much information the
audit starts with, **not** a separate profile or a separate skill. Start at
**Level-0** and move to Level-1 / Level-2 only if the mission asks for more
information. This is grounded in cap. 2 (L394-398) and cap. 3
("Niveles de dificultad"): the level is *increasing information given to the
agent*, the same audit and the same guardrails apply at every level, and
Level-0 is the starting point.

| Level | Information the agent receives | When |
|---|---|---|
| **Level-0** | Only the target's **IP / URL** of the segment. | **Default / starting point.** |
| **Level-1** | Level-0 **plus** vulnerable **URLs** and the segment's **topology**. | If the mission provides more. |
| **Level-2** | Level-1 **plus** the **vulnerability type / CVE** to target. | If the mission provides more. |

The level only narrows or widens the *starting* information; it never
relaxes the hard guardrails. **Scope stays the CIDR segment** at every level
(Level-1's extra URLs/topology must still resolve **inside** the segment),
**tooling stays NyxStrike-only**, and **no exploit, no report** still holds.
Do **not** invent a level or a level's contents: only `Level-0`, `Level-1`,
`Level-2` exist, with the information above — a mission may give any of them,
but never a fourth level or extra facts the mission did not provide.


## Quick Reference

| Call | Purpose | When |
|---|---|---|
| `profile_target` | Build `TargetProfile` + `AttackChain` (session) | Once, at the start; re-call to refresh |
| `propose_next_step` | Return the next `AttackStep` (tool + params + score + justification) | Every iteration — **never executes** |
| `execute_step` | Run one approved step, record result | Only after you vet the proposed step |
| `update_chain` | `add` / `remove` / `skip` / `reorder` / `new_objective` | To reorient mid-audit (preserves executed results) |
| `chain_report` | Counts + per-step status + ATT&CK tactic per step | End of audit, to write the report |

## Procedure

### Phase order (fixed — do not skip a phase to reach the next)

Map each proposed step to its phase by the capability NyxStrike tagged it with
(grounded in the tool catalog). A step belongs to a later phase only after the
earlier phase's objective is met.

| Phase | Goal | Grounded capabilities (tactic) |
|---|---|---|
| **recon** | Live hosts, ports, services, web tech in the segment | `network_scan`, `service_enumeration`, `web_fingerprint`, `api_discovery`, `subdomain`, `url_discovery` (Discovery / Reconnaissance) |
| **web-exploit** | Exploit OWASP Top 10 in every app found, proof-based | `vulnerability_scan`, `web_exploit`, `exploitation`, `payload` (Exploitation) |
| **foothold** | Initial access / a shell / usable credentials | `password`, `credential`, `hash` (Credential Access) |
| **post-exploitation** | Privilege escalation, lateral movement, persistence | `escalation`, `post_exploitation`, `lateral` (Privilege Escalation / Lateral Movement) |
| **report** | Assemble findings + ATT&CK tagging | `chain_report` |

If a capability does not map to a known phase, treat it as `None` (do not
invent a phase) and let `propose_next_step` / `update_chain` handle it.

### Step 1 — Profile the target
Call `profile_target(target=<segment-cidr>, objective=<obj>)`. Record the
`session_id` and the planned `AttackChain`. Confirm the profile only references
hosts **inside** the segment; if it proposes an out-of-scope host, re-profile or
`update_chain(action="remove")` before executing.
*Completion:* a `session_id` exists and the plan is in-scope.

### Step 2 — Recon
Drive recon steps until you have live hosts, open ports, services and web
technology for the whole segment. After each `execute_step`, persist what you
found (host, service, tech) to memory so the next phase can chain on it.
*Completion:* every live host in the segment is discovered and its services
enumerated; a web app is fingerprinted (OWASP target).

### Step 3 — Web exploitation
For each web app found, `propose_next_step` and approve OWASP Top 10 steps
(injection, XSS, auth, authz, SSRF, misconfig, …). Each finding must be
**proof-based**: record request + response + severity + remediation. Do **not**
fire destructive payloads without approving them.
*Completion:* every web app has been audited across OWASP Top 10 with evidence
for each confirmed (L3/L4) finding.

### Step 4 — Foothold
Convert a confirmed exploit into initial access: obtain a shell or usable
credentials. Reuse credentials found in recon/web-exploit rather than
re-attacking from scratch.
*Completion:* you have a foothold (shell or credentials) on at least one in-scope
host, evidenced.

### Step 5 — Post-exploitation
From the foothold, escalate privileges and move laterally **within the segment**
only. Re-check scope at every lateral step.
*Completion:* escalation / lateral movement is attempted and evidenced for each
reachable host in the segment.

### Step 6 — Reorient (any time)
When a proposed step does not fit the current phase, is blocked, or a better
pivot exists, call `update_chain` (`skip` / `remove` / `reorder` /
`new_objective`) — this **preserves** already-executed results. Re-plan with
`new_objective` only when the objective itself changes.
*Completion:* the chain matches the current phase's goal.

### Step 7 — Report
When `propose_next_step` returns `completed` (chain exhausted) or the segment is
audited end-to-end, call `chain_report(include_attack=True)` and write it to
`reports/audit.md`. Every finding carries its evidence and an ATT&CK tactic
(derived from the tool catalog, never invented).
*Completion:* `reports/audit.md` is written with per-finding evidence and ATT&CK
tags; the run stops.

## Pitfalls

- **Skipping a phase.** Do not jump from recon to post-exploitation because the
  plan suggested it; force the phase order with `update_chain`. A skipped phase
  is an invalid audit.
- **Approving everything.** If you approve without evaluating, you become a
  deterministic orchestrator — the whole point of plan-and-approve is that you
  vet each step.
- **Off-scope host.** Any host outside the CIDR is refused, not "later in scope".
  Re-profile or remove it; never execute against it.
- **Inventing a phase / tactic.** If a capability has no phase mapping, it is
  `None`. Do not fabricate a phase name or an ATT&CK tactic.
- **Bypassing NyxStrike.** Running a local binary invalidates the run. All
  discovery/scan/exploit goes through the MCP tools.
- **`propose` is not `execute`.** `propose_next_step` never runs anything; you
  must call `execute_step` to run an approved step.
- **Secrets in chat.** Redact captured tokens/credentials to the last 6 chars in
  any message; keep full values in report files.
- **Leaving the loop dangling.** Always end with `chain_report` →
  `reports/audit.md`; an audit with no report produces no evidence.

## Verification

- A `session_id` was created by `profile_target` and every step used it.
- The audit followed phase order: recon → web-exploit → foothold →
  post-exploitation → report (no skipped phase).
- Every confirmed finding has evidence (tool + exact request + response +
  severity + remediation); no "PASS" claims, only "no exploitable issue found in
  scope X within time T using method Y".
- Every executed host was inside the mission CIDR.
- `chain_report` produced ATT&CK tactic tags, and `reports/audit.md` exists with
  per-finding evidence before the run stopped.
