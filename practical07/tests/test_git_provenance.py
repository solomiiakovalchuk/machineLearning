"""Regression tests for the read-only git provenance helpers (temporary repositories only)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_core as A  # noqa: E402


def run(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, **kw).stdout.strip()


def new_repo(tmp):
    run(tmp, "init", "-q", ".")
    run(tmp, "config", "user.email", "t@example.com")
    run(tmp, "config", "user.name", "t")
    run(tmp, "config", "core.autocrlf", "false")


def commit(tmp, files):
    for name, text in files.items():
        (Path(tmp) / name).write_bytes(text.encode())
    run(tmp, "add", "-A")
    run(tmp, "commit", "-q", "-m", "c")
    return run(tmp, "rev-parse", "--short", "HEAD")


class TestGitProvenance(unittest.TestCase):
    def test_worktree_match_survives_autocrlf(self):
        """Bytes are identical, so the comparison must be True even if core.autocrlf=true is set afterwards."""
        with tempfile.TemporaryDirectory() as t:
            new_repo(t)
            commit(t, {"f.txt": "a\r\nb\r\n"})
            run(t, "config", "core.autocrlf", "true")
            self.assertEqual(A.worktree_matches_commit(t, "HEAD", "f.txt"), "identical")

    def test_worktree_match_detects_real_change(self):
        with tempfile.TemporaryDirectory() as t:
            new_repo(t)
            commit(t, {"f.txt": "a\r\nb\r\n"})
            (Path(t) / "f.txt").write_bytes(b"a\r\nc\r\n")
            self.assertEqual(A.worktree_matches_commit(t, "HEAD", "f.txt"), "different")

    def test_worktree_match_line_endings_only(self):
        """LF in the blob, CRLF in the working tree (autocrlf checkout): same content, different bytes."""
        with tempfile.TemporaryDirectory() as t:
            new_repo(t)
            commit(t, {"f.txt": "a\nb\n"})
            (Path(t) / "f.txt").write_bytes(b"a\r\nb\r\n")
            self.assertEqual(A.worktree_matches_commit(t, "HEAD", "f.txt"), "line_endings_only")


if __name__ == "__main__":
    unittest.main()
