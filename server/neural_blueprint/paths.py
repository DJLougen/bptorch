"""Repo-local path sandboxing for exports and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SandboxKind = Literal["exports", "checkpoints"]


class PathValidationError(ValueError):
    """Raised when a user-supplied path is absolute, traverses, or escapes its sandbox."""


def get_repo_root() -> Path:
    """Return the visualModeler repository root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_sandbox_path(relative_path: str, sandbox: SandboxKind) -> Path:
    """Resolve a project-relative path under ``exports/`` or ``checkpoints/``.

    Rejects absolute paths, empty paths, ``..`` traversal, and resolved paths that
    escape the sandbox root.
    """
    if relative_path is None or not str(relative_path).strip():
        raise PathValidationError("Path must not be empty")

    raw = Path(relative_path)
    if raw.is_absolute():
        raise PathValidationError("Absolute paths are not allowed")

    if ".." in raw.parts:
        raise PathValidationError("Path traversal is not allowed")

    base = (get_repo_root() / sandbox).resolve()
    base.mkdir(parents=True, exist_ok=True)

    resolved = (base / raw).resolve()
    if not resolved.is_relative_to(base):
        raise PathValidationError(f"Path escapes the {sandbox}/ sandbox")

    return resolved
