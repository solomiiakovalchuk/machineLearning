"""Project configuration: loading and validation into an immutable dataclass."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Mapping

from .errors import ConfigError

REQUIRED_PATHS = ("model", "model_metadata", "schema", "frozen_decision",
                  "inference_example", "split_ids", "manifest")
REQUIRED_KEYS = ("config_version", "project_name", "model_version", "schema_version",
                 "positive_class", "negative_class", "threshold", "max_batch_size",
                 "random_state", "numeric_tolerance", "paths")
MAX_BATCH_LIMIT = 100_000


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    config_version: str
    project_name: str
    model_version: str
    schema_version: str
    positive_class: str
    negative_class: str
    threshold: float
    max_batch_size: int
    random_state: int
    probability_tolerance: float
    metric_tolerance: float
    paths: Mapping[str, Path]  # absolute, already checked to lie inside project_root


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_relative_path(root: Path, name: str, value) -> Path:
    """Reject absolute paths, drive letters and any '..' component, then re-check after resolve()."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"paths.{name} must be a non-empty string")
    for flavour in (PurePosixPath(value), PureWindowsPath(value)):
        if flavour.is_absolute() or flavour.drive or flavour.root:
            raise ConfigError(f"paths.{name} must be relative to the project root")
        if ".." in flavour.parts:
            raise ConfigError(f"paths.{name} must not contain '..'")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError(f"paths.{name} resolves outside the project root")
    return resolved


def load_project_config(path) -> ProjectConfig:
    """Load and validate configs/project_config.json.

    Convention: the file lives in <project_root>/configs/, so the root is its grandparent.
    """
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {config_path.name}") from exc
    except ValueError as exc:
        raise ConfigError(f"config is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"missing config fields: {', '.join(missing)}")
    unknown = sorted(set(raw) - set(REQUIRED_KEYS))
    if unknown:
        raise ConfigError(f"unknown config fields: {', '.join(unknown)}")

    for key in ("config_version", "project_name", "model_version", "schema_version",
                "positive_class", "negative_class"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    if raw["positive_class"] == raw["negative_class"]:
        raise ConfigError("positive_class and negative_class must differ")

    threshold = raw["threshold"]
    if not _is_number(threshold) or not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ConfigError("threshold must be a number strictly between 0 and 1")

    batch = raw["max_batch_size"]
    if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= MAX_BATCH_LIMIT:
        raise ConfigError(f"max_batch_size must be an integer in [1, {MAX_BATCH_LIMIT}]")

    seed = raw["random_state"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("random_state must be a non-negative integer")

    tol = raw["numeric_tolerance"]
    if not isinstance(tol, dict) or set(tol) != {"probability_abs", "metric_abs"}:
        raise ConfigError("numeric_tolerance must have exactly probability_abs and metric_abs")
    for key, value in tol.items():
        if not _is_number(value) or not math.isfinite(value) or not 0.0 < value <= 1e-3:
            raise ConfigError(f"numeric_tolerance.{key} must be a number in (0, 1e-3]")

    paths_raw = raw["paths"]
    if not isinstance(paths_raw, dict):
        raise ConfigError("paths must be an object")
    missing_paths = [k for k in REQUIRED_PATHS if k not in paths_raw]
    if missing_paths:
        raise ConfigError(f"missing paths: {', '.join(missing_paths)}")
    unknown_paths = sorted(set(paths_raw) - set(REQUIRED_PATHS))
    if unknown_paths:
        raise ConfigError(f"unknown paths: {', '.join(unknown_paths)}")

    root = config_path.resolve().parent.parent
    paths = {k: _safe_relative_path(root, k, paths_raw[k]) for k in REQUIRED_PATHS}

    return ProjectConfig(
        project_root=root, config_version=raw["config_version"], project_name=raw["project_name"],
        model_version=raw["model_version"], schema_version=raw["schema_version"],
        positive_class=raw["positive_class"], negative_class=raw["negative_class"],
        threshold=float(threshold), max_batch_size=batch, random_state=seed,
        probability_tolerance=float(tol["probability_abs"]), metric_tolerance=float(tol["metric_abs"]),
        paths=MappingProxyType(paths))
