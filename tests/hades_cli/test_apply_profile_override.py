"""Regression tests for _apply_profile_override HADES_HOME guard (issue #22502).

When HADES_HOME is set to the hades root (e.g. systemd hardcodes
HADES_HOME=/root/.hades), _apply_profile_override must still read
active_profile and update HADES_HOME to the profile directory.

When HADES_HOME is already a profile directory (.../profiles/<name>),
_apply_profile_override must trust it and return without re-reading
active_profile (child-process inheritance contract).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path



def _run_apply_profile_override(
    tmp_path, monkeypatch, *, hades_home: str | None, active_profile: str | None,
    argv: list[str] | None = None,
):
    """Run _apply_profile_override in isolation.

    Returns the value of os.environ["HADES_HOME"] after the call,
    or None if unset.
    """
    hades_root = tmp_path / ".hades"
    hades_root.mkdir(parents=True, exist_ok=True)

    if active_profile is not None:
        (hades_root / "active_profile").write_text(active_profile)

    if active_profile and active_profile != "default":
        (hades_root / "profiles" / active_profile).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if hades_home is not None:
        monkeypatch.setenv("HADES_HOME", hades_home)
    else:
        monkeypatch.delenv("HADES_HOME", raising=False)

    monkeypatch.setattr(sys, "argv", argv or ["hades", "gateway", "start"])

    from hades_cli.main import _apply_profile_override
    _apply_profile_override()

    return os.environ.get("HADES_HOME")


class TestApplyProfileOverrideHadesHomeGuard:
    """Regression guard for issue #22502.

    Verifies that HADES_HOME pointing to the hades root does NOT suppress
    the active_profile check, while HADES_HOME already pointing to a
    profile directory IS trusted as-is.
    """

    def test_hades_home_at_root_with_active_profile_is_redirected(
        self, tmp_path, monkeypatch
    ):
        """HADES_HOME=/root/.hades + active_profile=coder must redirect
        HADES_HOME to .../profiles/coder.

        Bug scenario from #22502: systemd sets HADES_HOME to the hades root
        and the user switches to a profile via `hades profile use`.
        Before the fix, the guard returned early and active_profile was ignored.
        """
        hades_root = tmp_path / ".hades"
        hades_root.mkdir(parents=True, exist_ok=True)

        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hades_home=str(hades_root),
            active_profile="coder",
        )

        assert result is not None, "HADES_HOME must be set after profile redirect"
        assert "profiles" in result, (
            f"Expected HADES_HOME to point into profiles/ dir, got: {result!r}"
        )
        assert result.endswith("coder"), (
            f"Expected HADES_HOME to end with 'coder', got: {result!r}"
        )

    def test_hades_home_already_profile_dir_is_trusted(self, tmp_path, monkeypatch):
        """HADES_HOME=.../profiles/coder must not be overridden even when
        active_profile says something different.

        Preserves the child-process inheritance contract: a subprocess spawned
        with HADES_HOME already set to a specific profile must stay in that
        profile.
        """
        hades_root = tmp_path / ".hades"
        profile_dir = hades_root / "profiles" / "coder"
        profile_dir.mkdir(parents=True, exist_ok=True)

        (hades_root / "active_profile").write_text("other")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HADES_HOME", str(profile_dir))
        monkeypatch.setattr(sys, "argv", ["hades", "gateway", "start"])

        from hades_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HADES_HOME") == str(profile_dir), (
            "HADES_HOME must remain unchanged when already pointing to a profile dir"
        )

    def test_hades_home_unset_reads_active_profile(self, tmp_path, monkeypatch):
        """Classic case: HADES_HOME unset + active_profile=coder must set
        HADES_HOME to the profile directory (existing behaviour must not regress).
        """
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hades_home=None,
            active_profile="coder",
        )

        assert result is not None
        assert "coder" in result

    def test_hades_home_unset_default_profile_no_redirect(self, tmp_path, monkeypatch):
        """active_profile=default must not redirect HADES_HOME."""
        hades_root = tmp_path / ".hades"
        hades_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HADES_HOME", raising=False)
        monkeypatch.setattr(sys, "argv", ["hades", "gateway", "start"])
        (hades_root / "active_profile").write_text("default")

        from hades_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HADES_HOME") is None
