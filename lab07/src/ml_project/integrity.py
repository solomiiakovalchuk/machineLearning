"""Hashing and manifest verification helpers (no model loading here)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import ArtifactIntegrityError


def _no_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def strict_json_loads(text: str):
    """json.loads that rejects duplicate object keys (the stdlib silently keeps the last one)."""
    return json.loads(text, object_pairs_hook=_no_duplicates)


def sha256_file(path) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read file for hashing: {Path(path).name}") from exc
    return h.hexdigest()


def verify_manifest(project_root) -> dict:
    """Compare every file listed in project_manifest.json with its SHA-256.

    Returns {"checked": n}; raises ArtifactIntegrityError listing all mismatches.
    The manifest cannot contain its own hash, so it is not self-covering.
    """
    root = Path(project_root)
    manifest_path = root / "project_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactIntegrityError("project_manifest.json is missing or malformed") from exc
    problems = []
    for rel, expected in sorted(files.items()):
        target = root / rel
        if not target.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(target) != expected:
            problems.append(f"hash mismatch: {rel}")
    if problems:
        raise ArtifactIntegrityError("manifest verification failed: " + "; ".join(problems))
    return {"checked": len(files)}
