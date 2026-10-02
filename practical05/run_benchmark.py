"""Запуск benchmark (дорога частина): python run_benchmark.py [--out results]."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.benchmark import run_benchmark  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "results"), help="каталог результатів")
    args = parser.parse_args()
    results = run_benchmark(args.out)
    print(results.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
