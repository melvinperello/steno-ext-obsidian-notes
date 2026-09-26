"""Pytest configuration and environment isolation fixtures for obsidian_notes extension."""

import os
import sys
from pathlib import Path
import pytest

# Ensure obsidian_notes package root and parent directory are on sys.path
EXT_DIR = Path(__file__).resolve().parent.parent
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))
if str(EXT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EXT_DIR.parent))

# Ensure steno-svc root is on sys.path for BaseExtension imports if available locally
steno_root_env = os.getenv("STENO_SVC_ROOT")
if steno_root_env and Path(steno_root_env).exists():
    REPO_DIR = Path(steno_root_env)
else:
    REPO_DIR = Path.home() / "Documents" / "source" / "steno-svc" / "src"

if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# If steno_svc is not installed or available, alias base to steno_svc.extensions.base
try:
    import steno_svc.extensions.base  # noqa: F401
except ImportError:
    import types
    import base
    steno_svc_mod = types.ModuleType("steno_svc")
    ext_mod = types.ModuleType("steno_svc.extensions")
    base_mod = types.ModuleType("steno_svc.extensions.base")
    base_mod.BaseExtension = base.BaseExtension
    base_mod.ExtensionContext = base.ExtensionContext
    ext_mod.base = base_mod
    steno_svc_mod.extensions = ext_mod
    sys.modules["steno_svc"] = steno_svc_mod
    sys.modules["steno_svc.extensions"] = ext_mod
    sys.modules["steno_svc.extensions.base"] = base_mod


@pytest.fixture(autouse=True)
def isolate_vault(tmp_path, monkeypatch):
    """Isolates the output vault directory for tests."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OBSIDIAN_OUTPUT_DIR", str(notes_dir))
