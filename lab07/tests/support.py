"""Shared test helpers: make src importable, load real artifacts, build throw-away project copies."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = PROJECT / "configs" / "project_config.json"
COPIED_DIRS = ("configs", "models", "data")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def control_example():
    return read_json(PROJECT / "configs" / "inference_example.json")


def schema_feature_names():
    return [f["name"] for f in read_json(PROJECT / "configs" / "schema.json")["features"]]


def make_project_copy():
    """Copy configs/, models/, data/ (and the manifest if present) to a temp dir. Caller must clean up."""
    tmp = Path(tempfile.mkdtemp(prefix="lab07_test_"))
    for d in COPIED_DIRS:
        shutil.copytree(PROJECT / d, tmp / d)
    manifest = PROJECT / "project_manifest.json"
    if manifest.exists():
        shutil.copyfile(manifest, tmp / "project_manifest.json")
    return tmp
