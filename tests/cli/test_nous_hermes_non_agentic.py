"""Tests for the Nous-Hades-3/4 non-agentic warning detector.

Prior to this check, the warning fired on any model whose name contained
``"hades"`` anywhere (case-insensitive). That false-positived on unrelated
local Modelfiles such as ``hades-brain:qwen3-14b-ctx16k`` — a tool-capable
Qwen3 wrapper that happens to live under the "hades" tag namespace.

``is_nous_hades_non_agentic`` should only match the actual Nous Research
Hades-3 / Hades-4 chat family.
"""

from __future__ import annotations

import pytest

from hades_cli.model_switch import (
    _HADES_MODEL_WARNING,
    _check_hades_model_warning,
    is_nous_hades_non_agentic,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "NousResearch/Hades-3-Llama-3.1-70B",
        "NousResearch/Hades-3-Llama-3.1-405B",
        "hades-3",
        "Hades-3",
        "hades-4",
        "hades-4-405b",
        "hades_4_70b",
        "openrouter/hades3:70b",
        "openrouter/nousresearch/hades-4-405b",
        "NousResearch/Hades3",
        "hades-3.1",
    ],
)
def test_matches_real_nous_hades_chat_models(model_name: str) -> None:
    assert is_nous_hades_non_agentic(model_name), (
        f"expected {model_name!r} to be flagged as Nous Hades 3/4"
    )
    assert _check_hades_model_warning(model_name) == _HADES_MODEL_WARNING


@pytest.mark.parametrize(
    "model_name",
    [
        # Kyle's local Modelfile — qwen3:14b under a custom tag
        "hades-brain:qwen3-14b-ctx16k",
        "hades-brain:qwen3-14b-ctx32k",
        "hades-honcho:qwen3-8b-ctx8k",
        # Plain unrelated models
        "qwen3:14b",
        "qwen3-coder:30b",
        "qwen2.5:14b",
        "claude-opus-4-6",
        "anthropic/claude-sonnet-4.5",
        "gpt-5",
        "openai/gpt-4o",
        "google/gemini-2.5-flash",
        "deepseek-chat",
        # Non-chat Hades models we don't warn about
        "hades-llm-2",
        "hades2-pro",
        "nous-hades-2-mistral",
        # Edge cases
        "",
        "hades",  # bare "hades" isn't the 3/4 family
        "hades-brain",
        "brain-hades-3-impostor",  # "3" not preceded by /: boundary
    ],
)
def test_does_not_match_unrelated_models(model_name: str) -> None:
    assert not is_nous_hades_non_agentic(model_name), (
        f"expected {model_name!r} NOT to be flagged as Nous Hades 3/4"
    )
    assert _check_hades_model_warning(model_name) == ""


def test_none_like_inputs_are_safe() -> None:
    assert is_nous_hades_non_agentic("") is False
    # Defensive: the helper shouldn't crash on None-ish falsy input either.
    assert _check_hades_model_warning("") == ""
