"""Версия программы + сборка из git (по ней видно, сделан ли последний git pull)."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def build() -> str:
    """«1a2b3c4 от 16.11 18:46» или пусто, если git недоступен."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--date=format:%d.%m %H:%M"],
                             cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    if not out:
        return ""
    commit, _, when = out.partition(" ")
    return f"{commit} от {when}" + (" + локальные правки" if dirty else "")


def full_version() -> str:
    b = build()
    return f"{__version__} (сборка {b})" if b else __version__
