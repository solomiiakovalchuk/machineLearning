"""Sensitivity check of the test suite: apply one deliberate defect at a time to a TEMP COPY and expect the matching test to fail.

Usage (from lab07/): python mutation_check.py   (the repository is never modified)
"""
import shutil, subprocess, sys, os, tempfile
from pathlib import Path
SRC = Path(__file__).resolve().parent
muts = {
 "M1 load model BEFORE hash check": ("src/ml_project/predictor.py",
   '        self._check_hashes()\n        self._validator', '        import joblib; joblib.load(self._config.paths["model"])\n        self._check_hashes()\n        self._validator',
   "test_predictor.py", "tampered_model"),
 "M2 values taken in record order, not schema order": ("src/ml_project/validation.py",
   'for name in self.feature_names:\n            raw = record[name]', 'for name in record:\n            raw = record[name]',
   "test_predictor.py", "json_field_order"),
 "M3 out-of-range raises instead of warns": ("src/ml_project/validation.py",
   'warnings.append(f"{name}={value:g} outside', 'raise InputValidationError(f"{name}={value:g} outside',
   "test_validation.py", "out_of_range"),
 "M4 bool accepted as number": ("src/ml_project/validation.py",
   'if isinstance(raw, bool) or not isinstance(raw, numbers.Real):', 'if not isinstance(raw, numbers.Real):',
   "test_validation.py", "wrong_types"),
 "M5 target mapping flipped (target==1 as malignant)": ("src/ml_project/train.py",
   'y = (np.asarray(raw.target) == 0).astype(int)', 'y = (np.asarray(raw.target) == 1).astype(int)',
   "test_rebuild.py", "RebuildTests"),
 "M6 extra features silently ignored": ("src/ml_project/validation.py",
   '        if actual - expected:\n            raise', '        if False:\n            raise',
   "test_validation.py", "extra_feature"),
}
for label, (rel, old, new, tfile, key) in muts.items():
    tmp = Path(tempfile.mkdtemp(prefix="mut_"))
    dst = tmp / "lab07"
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "dist", "*.egg-info"))
    p = dst / rel
    s = p.read_text(encoding="utf-8")
    assert old in s, label
    p.write_text(s.replace(old, new, 1), encoding="utf-8", newline="\n")
    env = dict(os.environ, PYTHONPATH=str(dst / "src"))
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", tfile, "-k", key],
                       cwd=dst, env=env, capture_output=True, text=True)
    tail = [l for l in r.stderr.splitlines() if l.startswith(("Ran", "FAILED", "OK"))]
    print(f"{label}: exit={r.returncode} {tail}")
    shutil.rmtree(tmp, ignore_errors=True)
