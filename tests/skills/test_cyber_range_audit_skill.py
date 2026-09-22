"""Per-skill test for the cyber-range-audit skill (TDD, no live network).

Verifies the skill that B3 created is well-formed and that it encodes the
phased plan-and-approve contract it must impose, without inventing the phase
flow. All checks read the committed SKILL.md — no network, no live NyxStrike.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills" / "security" / "cyber-range-audit" / "SKILL.md"

REQUIRED_SECTIONS = [
    "When to Use",
    "Prerequisites",
    "Procedure",
    "Pitfalls",
    "Verification",
]
# The four attack phases plus reporting, in the fixed order the skill must
# impose (cap. 8 line 50 + SOUL rule 3).
PHASE_ORDER = [
    "recon",
    "web-exploit",
    "foothold",
    "post-exploitation",
    "report",
]
# The 5-tool plan-and-approve contract implemented by B1 (cap. 8, sec:plan-and-approve).
CONTRACT_TOOLS = [
    "profile_target",
    "propose_next_step",
    "execute_step",
    "update_chain",
    "chain_report",
]
MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
    re.I,
)


def _load():
    assert SKILL.exists(), f"missing skill file: {SKILL}"
    return SKILL.read_text(encoding="utf-8")


def _frontmatter(content):
    assert content.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "SKILL.md frontmatter is not closed"
    import yaml

    return yaml.safe_load(content[3 : m.start() + 3])


def test_file_exists_and_frontmatter_fields():
    fm = _frontmatter(_load())
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "cyber-range-audit"
    assert fm["platforms"], "platforms must be a non-empty list"


def test_name_matches_directory():
    fm = _frontmatter(_load())
    assert fm["name"] == SKILL.parent.name


def test_description_hardline():
    fm = _frontmatter(_load())
    desc = str(fm["description"])
    assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
    assert desc.rstrip().endswith("."), "description must end with a period"
    assert not MARKETING.search(desc), f"marketing word in description: {desc!r}"


def test_required_sections_present():
    content = _load()
    for section in REQUIRED_SECTIONS:
        assert section in content, f"missing section: {section!r}"


def test_imposes_all_phases_in_order():
    # Verify phase order inside the "## Procedure" section, where the phases are
    # encoded in the fixed order. A global first-occurrence check is wrong here
    # because "report" also appears in the "No exploit, no report" guardrail
    # earlier in the doc — that guardrail mention must not be treated as a phase.
    content = _load()
    proc = content.find("## Procedure")
    assert proc != -1, "missing ## Procedure section"
    body = content[proc:]
    last = -1
    for phase in PHASE_ORDER:
        idx = body.find(phase)
        assert idx != -1, f"phase not encoded in Procedure: {phase!r}"
        assert idx > last, (
            f"phase order violated in Procedure: {phase!r} appears before the "
            f"previous phase"
        )
        last = idx


def test_drives_the_plan_and_approve_contract():
    content = _load()
    for tool in CONTRACT_TOOLS:
        assert tool in content, f"contract tool not referenced: {tool!r}"
    # plan-and-approve core invariant: propose never executes.
    assert "never executes" in content.lower() or "never execute" in content.lower(), (
        "skill must state propose_next_step never executes"
    )


def test_scope_guardrail_present():
    content = _load().lower()
    assert "scope" in content
    assert "segment" in content
    # No exploit, no report.
    assert "no exploit, no report" in content


def test_no_machine_local_paths():
    content = _load()
    m = re.search(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\+Users\\+(?<!)", content)
    assert not m, f"machine-local path found: {content[m.start():m.start()+40]!r}"


def test_related_skills_resolves_in_repo():
    fm = _frontmatter(_load())
    names = {p.parent.name for p in REPO.glob("skills/**/SKILL.md")}
    names |= {p.parent.name for p in REPO.glob("optional-skills/**/SKILL.md")}
    for rel in (fm.get("metadata", {}).get("hermes", {}).get("related_skills") or []):
        assert rel in names, f"dangling related_skill: {rel!r}"


# ---------------------------------------------------------------------------
# C3 — Difficulty levels (Level-0/1/2) as a PARAMETER of the skill
# Grounded in cap. 2 (L394-398), cap. 3 (L126-127), and the wiki decision
# (agentcyberrange-y-evaluacion.md L71): the difficulty level is a PARAMETER
# of the audit skill, NOT a separate profile. Level-0 (only IP/URL) is the
# starting point; Level-1/2 add more information. NOT invented here.
# ---------------------------------------------------------------------------
DIFFICULTY_LEVELS = ["Level-0", "Level-1", "Level-2"]


def test_difficulty_is_a_parameter():
    # cap. 3: "El nivel es un parámetro de la skill de auditoría, no un
    # perfil distinto." The skill must present difficulty as a parameter.
    content = _load()
    assert "difficulty" in content.lower()


def test_difficulty_levels_are_documented():
    # cap. 2 L394-398: "niveles de dificultad 0/1/2 (información creciente)".
    content = _load()
    for level in DIFFICULTY_LEVELS:
        assert level in content, f"difficulty level not documented: {level!r}"


def test_level0_is_only_ip_url():
    # cap. 2 L394-398 / cap. 3 L126: Level-0 = the agent only receives the
    # target's IP/URL (the starting point).
    content = _load()
    lower = content.lower()
    assert "level-0" in lower or "level 0" in lower
    assert "ip/url" in lower or "ip / url" in lower or ("ip" in lower and "url" in lower)


def test_difficulty_levels_are_increasing_information():
    # cap. 2 L394-398: information growing per level
    # (URL/IP -> +URLs/topology -> +vuln type/CVE).
    content = _load().lower()
    # Level-1 adds URLs / vulnerable topology; Level-2 adds vuln type / CVE.
    assert "cve" in content
    assert "topology" in content or "topología" in content
