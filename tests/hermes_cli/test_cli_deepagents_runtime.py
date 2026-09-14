"""The classic CLI must honor the same deepagents_mode config flag as the
gateway / TUI, otherwise `hermes chat` silently runs the native runtime even
when config.yaml sets deepagents_mode: true (e.g. inside the container)."""

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

_resolve = CLIAgentSetupMixin._resolve_deepagents_runtime


@pytest.mark.parametrize(
    "config,expected",
    [
        # gateway.deepagents_mode (where the container config puts it)
        ({"gateway": {"deepagents_mode": True}}, "deepagents"),
        ({"gateway": {"deepagents_mode": "true"}}, "deepagents"),
        ({"gateway": {"deepagents_mode": False}}, "native"),
        # top-level fallback (TUI precedence parity)
        ({"deepagents_mode": True}, "deepagents"),
        ({"deepagents_mode": "yes"}, "deepagents"),
        # gateway section wins over absent/contradictory top-level
        ({"gateway": {"deepagents_mode": True}, "deepagents_mode": False}, "deepagents"),
        # defaults / robustness
        ({}, "native"),
        (None, "native"),
        ({"gateway": "not-a-dict"}, "native"),
        ({"gateway": {}}, "native"),
    ],
)
def test_resolve_deepagents_runtime(config, expected):
    assert _resolve(config) == expected
