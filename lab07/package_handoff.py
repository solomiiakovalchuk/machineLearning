"""Build dist/lab07_handoff.zip for lab08 and verify the archive by extracting it elsewhere.

Usage (from lab07/):  python package_handoff.py
Steps: zip -> extract into a fresh temp dir -> verify manifest, run unit tests and CLI smoke from the
extracted copy (never from this working tree) -> write reports/handoff_verification.json.
Run build_artifacts.py first: the archive contains project_manifest.json as it is on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "dist" / "lab07_handoff.zip"
INCLUDE_DIRS = ["configs", "data", "models", "reports", "src", "tests"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "requirements-dev.txt", "build_artifacts.py",
                 "package_handoff.py", "project_manifest.json", "README.md", ".gitattributes"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}
# Written after packaging, so they can never describe the archive that contains them.
SKIP_NAMES = {"handoff_verification.json"}


def collect() -> list[Path]:
    files = [ROOT / f for f in INCLUDE_FILES if (ROOT / f).is_file()]
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not (SKIP_PARTS & set(p.parts)) and not any(x.endswith(".egg-info") for x in p.parts) and p.suffix != ".pyc" and p.name not in SKIP_NAMES:
                files.append(p)
    return sorted(files)


def build_zip() -> list[str]:
    ARCHIVE.parent.mkdir(exist_ok=True)
    names = []
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in collect():
            arc = "lab07/" + p.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())
            names.append(arc)
    return names


def run(cmd, cwd, env):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout, r.stderr


def verify_extracted() -> dict:
    result = {"archive": ARCHIVE.name, "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              "archive_bytes": ARCHIVE.stat().st_size, "python": platform.python_version()}
    with tempfile.TemporaryDirectory(prefix="lab07_extract_") as tmp:
        with zipfile.ZipFile(ARCHIVE) as zf:
            zf.extractall(tmp)
        proj = Path(tmp) / "lab07"
        result["files_in_archive"] = sum(1 for p in proj.rglob("*") if p.is_file())
        result["junk_entries"] = sorted(p.relative_to(proj).as_posix() for p in proj.rglob("*")
                                        if "__pycache__" in p.parts or p.suffix == ".pyc"
                                        or any(x.endswith(".egg-info") for x in p.parts))
        env = dict(os.environ, PYTHONPATH=str(proj / "src"), PYTHONIOENCODING="utf-8")
        cfg = "configs/project_config.json"
        cli = [sys.executable, "-m", "ml_project.cli", "--config", cfg]
        code, out, err = run([sys.executable, "-c", "import ml_project,sys;print(ml_project.__file__)"], proj, env)
        result["package_imported_from_extracted_copy"] = out.strip().startswith(str(proj))
        code, out, err = run(cli + ["verify"], proj, env)
        result["verify_exit"] = code
        result["verify_output"] = json.loads(out) if code == 0 else err.strip()
        code, out, err = run(cli + ["predict", "configs/inference_example.json"], proj, env)
        result["predict_exit"] = code
        if code == 0:
            example = json.loads((proj / cfg).parent.joinpath("inference_example.json").read_text(encoding="utf-8"))
            prob = json.loads(out)["probability"]
            result["predict_probability"] = prob
            result["predict_abs_diff_vs_expected"] = abs(prob - example["expected_probability"][0])
        else:
            result["predict_stderr"] = err.strip()
        code, out, err = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], proj, env)
        result["unittest_exit"] = code
        result["unittest_summary"] = [ln for ln in err.strip().splitlines() if ln.startswith(("Ran ", "OK", "FAILED"))]
        result["passed"] = bool(not result["junk_entries"] and result["package_imported_from_extracted_copy"] and result["verify_exit"] == 0
                                and result["predict_exit"] == 0 and code == 0
                                and result.get("predict_abs_diff_vs_expected", 1) <= 1e-12)
    return result


def main() -> int:
    names = build_zip()
    print(f"archive written: {ARCHIVE.name}, {len(names)} files")
    result = verify_extracted()
    (ROOT / "reports" / "handoff_verification.json").write_bytes(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
