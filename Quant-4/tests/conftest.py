"""Project test configuration.

Hardens the suite for restricted filesystems (DSH sandbox / Windows
ACL-restricted environments) where pytest's tmp machinery and Python's
``tempfile`` cleanup break because POSIX ``0o700`` modes are mapped to
over-restrictive DACLs:

1. ``TempPathFactory.getbasetemp`` is replaced: it never ``rm_rf``'s a
   pre-existing basetemp and creates all tmp dirs with permissive modes
   (POSIX modes are meaningless on Windows ACLs).
2. pytest's ``make_numbered_dir`` uses a permissive mode on Windows.
3. Python's ``tempfile._resetperms`` chmod is skipped on Windows (the sandbox
   denies the ``0o700`` chmod during cleanup).
4. The final dead-symlink scan is non-fatal when a directory is unlistable.

Because ``_pytest.tmpdir`` imports the helpers by name at import time, both
``_pytest.pathlib`` and ``_pytest.tmpdir`` module attributes are patched.

No test semantics are changed: fixtures, isolation and ordering are untouched.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import warnings

import _pytest.pathlib as _pp
import _pytest.tmpdir as _pt


def _resilient_cleanup_dead_symlinks(root) -> None:
    """Same cleanup as pytest, but non-fatal when a directory is unlistable."""
    try:
        _ORIGINAL_CLEANUP(root)
    except PermissionError as exc:  # pragma: no cover - environment dependent
        warnings.warn(
            f"pytest tmp cleanup skipped (PermissionError on {root}): {exc}",
            stacklevel=2,
        )


_ORIGINAL_CLEANUP = _pp.cleanup_dead_symlinks
_pp.cleanup_dead_symlinks = _resilient_cleanup_dead_symlinks
if hasattr(_pt, "cleanup_dead_symlinks"):
    _pt.cleanup_dead_symlinks = _resilient_cleanup_dead_symlinks

if os.name == "nt":  # pragma: no cover - platform dependent

    # --- 1) numbered tmp dirs: permissive mode on Windows --------------------
    _orig_make_numbered_dir = _pp.make_numbered_dir

    def _make_numbered_dir(root, prefix, mode=0o700):
        return _orig_make_numbered_dir(root, prefix, mode=0o777 if mode == 0o700 else mode)

    _pp.make_numbered_dir = _make_numbered_dir
    if hasattr(_pt, "make_numbered_dir"):
        _pt.make_numbered_dir = _make_numbered_dir

    # --- 2) tempfile cleanup: `TemporaryDirectory` applies a 0o700 chmod     ---
    # ---    (meaningless on Windows, denied by the sandbox) during teardown  ---
    # ---    and its internal _rmtree closure bypasses module-level patches;  ---
    # ---    make removal best-effort so restricted dirs never fail a test    ---
    import shutil  # noqa: PLC0415

    def _tolerant_rmtree(cls, name, ignore_errors=False):
        shutil.rmtree(name, ignore_errors=True)

    tempfile.TemporaryDirectory._rmtree = classmethod(_tolerant_rmtree)

    # --- 3) getbasetemp: never rm_rf a pre-existing basetemp; use permissive ---
    # ---    modes and a workspace-local root (OS temp may be restricted)      ---
    _WORKSPACE_TMP_ROOT = Path(__file__).resolve().parent / ".pytest_tmp"

    def _getbasetemp(self):
        """Patched: reuses an existing basetemp and avoids 0o700-only dirs."""
        if self._basetemp is not None:
            return self._basetemp
        if self._given_basetemp is not None:
            basetemp = Path(os.path.abspath(str(self._given_basetemp)))
            basetemp.mkdir(parents=True, exist_ok=True)
            basetemp = basetemp.resolve()
        else:
            _WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
            keep = self._retention_count
            if self._retention_policy == "none":
                keep = 0
            basetemp = _pp.make_numbered_dir_with_cleanup(
                prefix="pytest-",
                root=_WORKSPACE_TMP_ROOT,
                keep=keep,
                lock_timeout=_pp.LOCK_TIMEOUT,
                mode=0o777,
                register=self._exit_stack.callback,
            )
        self._basetemp = basetemp
        self._trace("new basetemp", basetemp)
        return basetemp

    _pt.TempPathFactory.getbasetemp = _getbasetemp
