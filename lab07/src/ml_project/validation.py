"""Input contract: named features in schema order; blocking errors vs diagnostic warnings."""
from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import BatchSizeError, ConfigError, InputValidationError

_MAX_LISTED = 5


@dataclass(frozen=True)
class ValidatedRecord:
    values: tuple          # floats, in schema order
    warnings: tuple        # diagnostic strings (do not block prediction)


class SchemaValidator:
    """Validates a record against the machine-readable schema (configs/schema.json)."""

    def __init__(self, schema: Mapping):
        try:
            features = schema["features"]
            self.schema_version = schema["schema_version"]
            self.positive_class = schema["positive_class"]
            self.negative_class = schema["negative_class"]
            self.non_negative = bool(schema["non_negative"])
            names = [f["name"] for f in features]
            self._min = {f["name"]: float(f["min"]) for f in features}
            self._max = {f["name"]: float(f["max"]) for f in features}
            dtypes = {f["dtype"] for f in features}
            required = all(f["required"] is True for f in features)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"malformed schema: {exc!r}") from exc
        if not names or len(set(names)) != len(names):
            raise ConfigError("schema feature names must be non-empty and unique")
        if dtypes != {"float64"} or not required:
            raise ConfigError("schema supports only required float64 features")
        if any(self._min[n] > self._max[n] for n in names):
            raise ConfigError("schema has min > max")
        self.feature_names = tuple(names)

    @classmethod
    def from_file(cls, path) -> "SchemaValidator":
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read schema file: {Path(path).name}") from exc

    @staticmethod
    def _short(items) -> str:
        items = list(items)
        shown = ", ".join(map(str, items[:_MAX_LISTED]))
        return shown + (f" (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else "")

    def validate_record(self, record) -> ValidatedRecord:
        """Return values in schema order plus warnings; raise InputValidationError on a contract violation."""
        if not isinstance(record, Mapping):
            raise InputValidationError("record must be an object with named features")
        bad_keys = [k for k in record if not isinstance(k, str)]
        if bad_keys:
            raise InputValidationError("feature names must be strings")
        expected, actual = set(self.feature_names), set(record)
        if expected - actual:
            raise InputValidationError("missing features: " + self._short(n for n in self.feature_names if n not in actual))
        if actual - expected:
            raise InputValidationError("unexpected features: " + self._short(sorted(actual - expected)))

        values, warnings = [], []
        for name in self.feature_names:
            raw = record[name]
            # bool is an int subclass and numpy.bool_ is not Real; strings are never coerced.
            if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
                raise InputValidationError(f"feature {name} must be a number, got {type(raw).__name__}")
            try:
                value = float(raw)
            except (OverflowError, ValueError) as exc:
                raise InputValidationError(f"feature {name} is not representable as float64") from exc
            if not math.isfinite(value):
                raise InputValidationError(f"feature {name} must be finite")
            if self.non_negative and value < 0:
                raise InputValidationError(f"feature {name} must be non-negative")
            if value < self._min[name] or value > self._max[name]:
                warnings.append(f"{name}={value:g} outside train range [{self._min[name]:g}, {self._max[name]:g}]")
            values.append(value)
        return ValidatedRecord(tuple(values), tuple(warnings))

    def validate_batch(self, records, max_batch_size: int) -> list:
        """Validate a non-empty list of records not exceeding max_batch_size."""
        if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
            raise InputValidationError("batch must be a list of records")
        if len(records) == 0:
            raise InputValidationError("batch must not be empty")
        if len(records) > max_batch_size:
            raise BatchSizeError(f"batch of {len(records)} exceeds max_batch_size={max_batch_size}")
        result = []
        for i, record in enumerate(records):
            try:
                result.append(self.validate_record(record))
            except InputValidationError as exc:
                raise InputValidationError(f"record {i}: {exc}") from exc
        return result
