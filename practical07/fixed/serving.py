"""Hardened serving layer for the frozen lab06 pipeline.

The model, its parameters, the target, the threshold and the test set are unchanged.
What is added: checksum verification BEFORE unpickling, a named-feature record contract
(order independent, unknown/missing fields rejected), finite/non-negative/type checks,
explicit out-of-range and geometry warnings, and a stable response format.

Run from this directory:  python serving.py
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from pathlib import Path

ACCEPTED, REJECTED, WARNING = "accepted", "rejected", "accepted_with_warning"
HERE = Path(__file__).resolve().parent


class IntegrityError(ValueError):
    """Raised when a model file does not match its recorded SHA-256."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_expected_sha256(sha_path) -> str:
    """Parse 'sha256sum' format: '<64 hex>  <file name>'."""
    token = Path(sha_path).read_text(encoding="utf-8").split()[0].lower()
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise IntegrityError(f"malformed checksum file: {sha_path}")
    return token


def load_verified_model(model_path, sha_path):
    """Verify the SHA-256 of the raw bytes first; unpickle only on a match."""
    raw = Path(model_path).read_bytes()
    expected = read_expected_sha256(sha_path)
    actual = sha256_bytes(raw)
    if actual != expected:
        raise IntegrityError(f"model checksum mismatch: expected {expected}, got {actual}")
    import joblib  # imported late: nothing is unpickled before the check above
    import io
    return joblib.load(io.BytesIO(raw))


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool,))


def validate_record(record, schema) -> tuple:
    """Return (status, messages). status is accepted / rejected / accepted_with_warning.

    Reject: not a mapping, missing or unknown field, non-numeric type, NaN/inf, negative value.
    Warn:   value outside the train range of the schema, radius/area inconsistent beyond tolerance.
    Field order never matters.
    """
    if not isinstance(record, Mapping):
        return REJECTED, ["record must be a mapping feature_name -> number"]
    feats = schema["features"]
    names = [f["name"] for f in feats]
    errors = []
    missing = [n for n in names if n not in record]
    extra = [k for k in record if k not in names]
    if missing:
        errors.append("missing features: " + ", ".join(missing))
    if extra:
        errors.append("unknown features: " + ", ".join(map(str, extra)))
    if errors:
        return REJECTED, errors
    for f in feats:
        v = record[f["name"]]
        if not _is_number(v):
            errors.append(f"{f['name']}: expected a number, got {type(v).__name__}")
        elif not math.isfinite(float(v)):
            errors.append(f"{f['name']}: value must be finite")
        elif float(v) < 0:
            errors.append(f"{f['name']}: value must be non-negative")
    if errors:
        return REJECTED, errors
    warns = []
    for f in feats:
        v = float(record[f["name"]])
        if v < f["min"] or v > f["max"]:
            warns.append(f"{f['name']}={v:g} outside train range [{f['min']:g}, {f['max']:g}]")
    tol = schema.get("domain_checks", {}).get("area_radius_rel_tol")
    if tol is not None:
        r, a = float(record["radius_mean"]), float(record["area_mean"])
        if r > 0:
            ratio = a / (math.pi * r * r)
            if abs(ratio - 1.0) > tol:
                warns.append(f"area_mean/(pi*radius_mean^2)={ratio:.3f} deviates from 1 by more than {tol}")
    return (WARNING, warns) if warns else (ACCEPTED, [])


def record_to_vector(record, schema) -> list:
    """Order the values by the schema, independent of the record's key order."""
    return [float(record[f["name"]]) for f in schema["features"]]


def predict_records(model, records, schema, threshold=0.5, max_batch=None) -> list:
    """Per-record response: {status, messages, probability, label}; rejected records get None."""
    max_batch = max_batch or schema.get("max_batch", 10000)
    records = list(records)
    if len(records) == 0:
        raise ValueError("empty batch")
    if len(records) > max_batch:
        raise ValueError(f"batch of {len(records)} exceeds max_batch={max_batch}")
    out, rows, idx = [], [], []
    for i, rec in enumerate(records):
        status, msgs = validate_record(rec, schema)
        out.append({"status": status, "messages": msgs, "probability": None, "label": None})
        if status != REJECTED:
            rows.append(record_to_vector(rec, schema))
            idx.append(i)
    if rows:
        import numpy as np
        p = model.predict_proba(np.asarray(rows, dtype=float))[:, 1]
        for i, pi in zip(idx, p):
            out[i]["probability"] = float(pi)
            out[i]["label"] = int(pi >= threshold)
    return out


def main():
    schema = json.loads((HERE / "data_schema.json").read_text(encoding="utf-8"))
    model = load_verified_model(HERE / schema["model_file"], HERE / "pipeline.sha256")
    example = json.loads((HERE / "inference_example.json").read_text(encoding="utf-8"))
    names = [f["name"] for f in schema["features"]]
    records = [dict(zip(reversed(names), reversed(row))) for row in example["input"]]  # reversed key order on purpose
    res = predict_records(model, records, schema)
    for r, e in zip(res, example["expected_probability"]):
        assert r["status"] == ACCEPTED, r
        assert abs(r["probability"] - e) < 1e-12, (r["probability"], e)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
