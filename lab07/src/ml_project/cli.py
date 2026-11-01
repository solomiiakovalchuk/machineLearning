"""Command line interface: metadata, validate, predict, predict-batch, verify."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import load_project_config
from .errors import InputFileError, InputValidationError, ProjectError
from .integrity import verify_manifest
from .predictor import ProjectPredictor


def _read_json_file(path: str):
    try:
        # json accepts NaN/Infinity literals; the validator rejects them as non-finite values.
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise InputFileError(f"cannot read input file: {Path(path).name}") from exc
    except ValueError as exc:
        raise InputFileError(f"input is not valid JSON: {exc}") from exc


def _read_csv_records(path: str) -> list:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            if len(set(header)) != len(header):
                raise InputFileError("CSV header has duplicate column names")
            records = []
            for i, row in enumerate(reader):
                if None in row or None in row.values():
                    raise InputValidationError(f"CSV row {i}: number of cells differs from header")
                record = {}
                for name, cell in row.items():
                    cell = cell.strip()
                    try:
                        record[name] = float(cell) if cell else None
                    except ValueError:
                        raise InputValidationError(f"CSV row {i}: column {name} is not a number") from None
                records.append(record)
            return records
    except OSError as exc:
        raise InputFileError(f"cannot read input file: {Path(path).name}") from exc
    except (UnicodeDecodeError, csv.Error) as exc:
        raise InputFileError(f"input is not a valid CSV: {exc}") from exc


def _load_records(path: str, single: bool = False):
    """Return (payload, is_batch). CSV and JSON lists are batches; a JSON object is one record.

    A wrapper object may carry "record" (used by predict) and/or "records" (batch commands).
    """
    if path.lower().endswith(".csv"):
        return _read_csv_records(path), True
    payload = _read_json_file(path)
    if isinstance(payload, dict):
        keys = ("record", "records") if single else ("records", "record")
        for key in keys:
            if key in payload:
                return payload[key], key == "records"
    return payload, isinstance(payload, list)


def _emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _run(args) -> None:
    config = load_project_config(args.config)
    if args.command == "verify":
        _emit({"manifest_ok": True, **verify_manifest(config.project_root)})
        return
    predictor = ProjectPredictor(config)
    if args.command == "metadata":
        _emit(predictor.describe())
        return
    payload, is_batch = _load_records(args.input, single=args.command == "predict")
    if args.command == "validate":
        if is_batch:
            validated = predictor.validator.validate_batch(payload, config.max_batch_size)
        else:
            validated = [predictor.validator.validate_record(payload)]
        _emit({"valid": True, "records": len(validated),
               "warnings": [list(v.warnings) for v in validated]})
    elif args.command == "predict":
        if is_batch:
            raise InputValidationError("predict expects one record object; use predict-batch for lists")
        _emit(predictor.predict_record(payload))
    else:  # predict-batch
        if not is_batch:
            raise InputValidationError("predict-batch expects a JSON list of records or a CSV file")
        results = predictor.predict_batch(payload)
        _emit({"count": len(results), "results": results})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml_project.cli", description="Frozen breast-cancer classifier component")
    parser.add_argument("--config", default="configs/project_config.json", help="path to project_config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("metadata", help="print model/schema description")
    sub.add_parser("verify", help="check project_manifest.json hashes")
    for name, text in (("validate", "validate a JSON record/list or CSV without predicting"),
                       ("predict", "predict one JSON record"),
                       ("predict-batch", "predict a JSON list or CSV file")):
        p = sub.add_parser(name, help=text)
        p.add_argument("input", help="JSON or CSV file with named features")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except ProjectError as exc:
        print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # last resort: no traceback for users, but never exit 0
        print(f"unexpected error: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
