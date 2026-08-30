"""Project serialization, schema validation, migrations, and atomic persistence."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Union

from neural_blueprint.ir.models import Project

CURRENT_SCHEMA_VERSION = 1

MigrationFn = Callable[[Dict[str, Any]], Dict[str, Any]]
MIGRATIONS: Dict[int, MigrationFn] = {}


def register_migration(from_version: int, fn: MigrationFn):
    MIGRATIONS[from_version] = fn


def serialize_project(project: Project) -> Dict[str, Any]:
    """Serializes a Project instance into a JSON-compatible dictionary."""
    return project.model_dump(mode="json")


def deserialize_project(data: Dict[str, Any]) -> Project:
    """Validates and deserializes a dictionary into a Project instance."""
    if not isinstance(data, dict):
        raise ValueError("Project data must be a dictionary")

    version = data.get("schema_version", 1)
    if not isinstance(version, int):
        raise ValueError(f"Invalid schema_version: {version}")

    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Project schema version {version} is newer than "
            f"maximum supported version {CURRENT_SCHEMA_VERSION}"
        )

    # Run sequential migrations if needed
    current_data = dict(data)
    while version < CURRENT_SCHEMA_VERSION:
        if version not in MIGRATIONS:
            raise ValueError(
                f"No migration path from schema version {version} to {CURRENT_SCHEMA_VERSION}"
            )
        current_data = MIGRATIONS[version](current_data)
        version = current_data.get("schema_version", version + 1)

    return Project.model_validate(current_data)


def save_project_file(project: Project, filepath: Union[str, Path]) -> None:
    """Atomically writes a project to a JSON file on disk."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = serialize_project(project)
    content = json.dumps(data, indent=2)

    # Write to temp file in same directory and atomically rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix="project_tmp_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_project_file(filepath: Union[str, Path]) -> Project:
    """Loads and deserializes a project from a JSON file."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return deserialize_project(data)
