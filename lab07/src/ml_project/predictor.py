"""Stable inference interface over the frozen, hash-verified pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .errors import ArtifactIntegrityError, ConfigError, InputValidationError
from .integrity import sha256_file, strict_json_loads  # noqa: F401  (re-exported: part of the required interface)
from .validation import SchemaValidator


def _read_json(path: Path, what: str) -> dict:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactIntegrityError(f"cannot read {what}") from exc


class ProjectPredictor:
    """Loads config, schema, metadata and frozen decision; verifies hashes; then loads the model.

    The model file is deserialised with joblib (pickle) ONLY after its SHA-256 equals the value in
    model_metadata.json. Hash equality proves unchanged bytes, not model quality or safety.
    """

    def __init__(self, config: ProjectConfig):
        self._config = config
        paths = config.paths
        self._metadata = _read_json(paths["model_metadata"], "model metadata")
        self._decision = _read_json(paths["frozen_decision"], "frozen decision")
        self._check_hashes()
        self._validator = SchemaValidator.from_file(paths["schema"])
        self._check_consistency()
        self._model = self._load_model()

    # -- integrity -----------------------------------------------------------------------------
    def _check_hashes(self) -> None:
        paths, meta = self._config.paths, self._metadata
        for key, field in (("schema", "schema_sha256"), ("frozen_decision", "frozen_decision_sha256")):
            if sha256_file(paths[key]) != meta.get(field):
                raise ArtifactIntegrityError(f"{key} hash differs from model_metadata.json")
        if sha256_file(paths["model"]) != meta.get("model_sha256"):
            raise ArtifactIntegrityError("model hash differs from model_metadata.json; model not loaded")

    def _check_consistency(self) -> None:
        cfg, meta, val = self._config, self._metadata, self._validator
        pairs = [("schema_version", cfg.schema_version, val.schema_version),
                 ("schema_version(meta)", cfg.schema_version, meta.get("schema_version")),
                 ("model_version", cfg.model_version, meta.get("model_version")),
                 ("positive_class", cfg.positive_class, val.positive_class),
                 ("negative_class", cfg.negative_class, val.negative_class),
                 ("positive_class(meta)", cfg.positive_class, meta.get("positive_class")),
                 ("threshold", cfg.threshold, meta.get("threshold"))]
        bad = [name for name, a, b in pairs if a != b]
        if bad:
            raise ConfigError("config disagrees with frozen artifacts: " + ", ".join(bad)
                              + " (a changed threshold or version needs a new model version)")

    def _load_model(self):
        import joblib  # imported late: nothing deserialises before the checks above

        model = joblib.load(self._config.paths["model"])
        if getattr(model, "n_features_in_", None) != len(self._validator.feature_names):
            raise ArtifactIntegrityError("model feature count differs from schema")
        if list(getattr(model, "classes_", [])) != [0, 1]:
            raise ArtifactIntegrityError("model classes are not [0, 1] (1 = positive class)")
        return model

    # -- public interface ----------------------------------------------------------------------
    def describe(self) -> dict:
        """Metadata for users; contains no sklearn objects and no file paths."""
        cfg, meta = self._config, self._metadata
        import sklearn
        runtime = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                   "numpy": np.__version__}
        trained = meta.get("environment", {})
        runtime_warnings = [f"{k}: runtime {runtime[k]} != training {trained[k]}"
                            for k in runtime if k in trained and trained[k] != runtime[k] and k != "python"]
        return {
            "model_version": cfg.model_version,
            "model_sha256": meta["model_sha256"],
            "schema_version": cfg.schema_version,
            "schema_sha256": meta["schema_sha256"],
            "n_features": len(self._validator.feature_names),
            "feature_names": list(self._validator.feature_names),
            "positive_class": cfg.positive_class,
            "negative_class": cfg.negative_class,
            "threshold": cfg.threshold,
            "max_batch_size": cfg.max_batch_size,
            "family": meta["family"],
            "candidate_id": meta["candidate_id"],
            "intended_use": meta["intended_use"],
            "limitations": meta["limitations"],
            "runtime_warnings": runtime_warnings,
        }

    def _respond(self, validated: list) -> list:
        matrix = np.array([v.values for v in validated], dtype=float)
        probabilities = self._model.predict_proba(matrix)[:, 1]
        cfg = self._config
        return [{
            "probability": float(p),
            "predicted_class": cfg.positive_class if p >= cfg.threshold else cfg.negative_class,
            "threshold": cfg.threshold,
            "positive_class": cfg.positive_class,
            "model_version": cfg.model_version,
            "schema_version": cfg.schema_version,
            "warnings": list(v.warnings),
        } for p, v in zip(probabilities, validated)]

    def predict_record(self, record) -> dict:
        return self._respond([self._validator.validate_record(record)])[0]

    def predict_batch(self, records) -> list:
        return self._respond(self._validator.validate_batch(records, self._config.max_batch_size))

    def model_probabilities(self, matrix) -> np.ndarray:
        """P(positive class) for an already validated n x n_features matrix (used by rebuild checks)."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self._validator.feature_names):
            raise InputValidationError("matrix must be n x n_features")
        return self._model.predict_proba(matrix)[:, 1]

    def model_steps(self) -> list:
        """Fitted final estimator wrapped in a list (for numeric comparison only; not part of the public response)."""
        return [self._model.steps[-1][1]]

    @property
    def validator(self) -> SchemaValidator:
        return self._validator
