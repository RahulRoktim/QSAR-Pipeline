"""Run provenance: record exactly what produced a set of results.

A QSAR model is only reproducible if the target, the data snapshot, the split
seed and the library versions that generated it are all recoverable. This module
writes that record next to the outputs so a reviewer - or the author six months
later - can reconstruct the run without guessing.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "rdkit",
    "scikit-learn",
    "pandas",
    "numpy",
    "xgboost",
    "lightgbm",
    "catboost",
    "chembl_webresource_client",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def file_digest(path: str | Path) -> str | None:
    """SHA-256 of a file, or None when it does not exist yet."""
    path = Path(path)
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    settings: dict[str, Any],
    resolved_target: dict[str, Any] | None = None,
    datasets: dict[str, str | Path] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Assemble the run manifest. Pure apart from reading dataset digests."""
    root = root or Path(__file__).resolve().parents[1]

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "resolved_target": resolved_target,
        "datasets": {
            name: {"path": str(path), "sha256": file_digest(path)}
            for name, path in (datasets or {}).items()
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "git_commit": _git_commit(root),
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
