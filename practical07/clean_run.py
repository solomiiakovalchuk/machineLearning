"""Clean-run check (task 5): fresh venvs OUTSIDE the repository, install by the shipped instructions only.

Usage: python clean_run.py <long_venv_root> <short_venv_root>
  A: handoff/requirements.txt in <long_venv_root>  (the location the audit was asked to use)
  B: the same file in <short_venv_root>            (separates a path-length failure from a content failure)
  C: fixed/requirements.txt in <short_venv_root>   (minimal inference requirements of the fixed version)
Writes results/clean_run.json. Nothing is written inside the repository except that file.
"""
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
HANDOFF = REPO / "lab06" / "handoff"


def run(cmd, cwd=None, timeout=1800):
    t = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, round(time.time() - t, 1), (p.stdout + p.stderr)


def attempt(name, venv_dir, req, script_dir, script, extra_files=()):
    venv_dir = Path(venv_dir)
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    rec = {"name": name, "venv_path_length": len(str(venv_dir)), "requirements": str(req.relative_to(REPO)).replace("\\", "/")}
    t0 = time.time()
    rc, s, out = run([sys.executable, "-m", "venv", str(venv_dir)])
    rec.update(venv_exit=rc, venv_s=s)
    py = venv_dir / "Scripts" / "python.exe"
    rc, s, out = run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)])
    rec.update(pip_exit=rc, pip_s=s, pip_tail=[l for l in out.strip().splitlines() if l.startswith(("ERROR", "HINT", "Successfully"))][:4])
    rc, _, out = run([str(py), "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"])
    pkgs = {l.split("==")[0].lower(): l.split("==")[1] for l in out.splitlines() if "==" in l}
    rec["installed_core"] = {k: pkgs.get(k) for k in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "flaml", "jupyter")}
    rec["n_installed"] = len(pkgs)
    work = venv_dir.parent / (venv_dir.name + "_work")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(script_dir, work, ignore=shutil.ignore_patterns("__pycache__"))
    rc, s, out = run([str(py), script], cwd=work)
    rec.update(run_exit=rc, run_s=s, run_script=script, run_output_tail=out.strip().splitlines()[-2:])
    rc, _, out = run([str(py), "--version"])
    rec["python"] = out.strip()
    rec["total_s"] = round(time.time() - t0, 1)
    rec["success"] = bool(rec["venv_exit"] == 0 and rec["pip_exit"] == 0 and rec["run_exit"] == 0)
    shutil.rmtree(work, ignore_errors=True)
    return rec


def main():
    long_root, short_root = Path(sys.argv[1]), Path(sys.argv[2])
    res = {"host_python": platform.python_version(), "platform": platform.platform(), "attempts": []}
    res["attempts"].append(attempt("A_full_requirements_long_path", long_root / "venv_pr7", HANDOFF / "requirements.txt", HANDOFF, "inference.py"))
    res["attempts"].append(attempt("B_full_requirements_short_path", short_root / "v7", HANDOFF / "requirements.txt", HANDOFF, "inference.py"))
    res["attempts"].append(attempt("C_fixed_minimal_requirements", short_root / "v7min", ROOT / "fixed" / "requirements.txt", ROOT / "fixed", "serving.py"))
    for d in (short_root / "v7", short_root / "v7min", long_root / "venv_pr7"):
        shutil.rmtree(d, ignore_errors=True)
    out = ROOT / "results" / "clean_run.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
