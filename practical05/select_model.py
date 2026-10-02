"""Відбір моделі за сценарієм (дешева частина; читає готовий benchmark_results.csv).

python select_model.py --config configs/realtime_workstation.json
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model_selector import (NoEligibleModelError, load_config, rank_models,  # noqa: E402
                                save_selection)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON-конфігурація сценарію")
    parser.add_argument("--results", default=str(ROOT / "results" / "benchmark_results.csv"))
    parser.add_argument("--models-dir", default=str(ROOT / "results" / "models"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        results = pd.read_csv(args.results)
        ranking, rejected = rank_models(results, config)
    except NoEligibleModelError as exc:
        print(f"ПОМИЛКА: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"ПОМИЛКА: {exc}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 200)
    print(f"Сценарій: {config['scenario']} (конфігурація v{config['version']})")
    print("\nРейтинг допущених моделей:")
    print(ranking.round(4).to_string(index=False))
    print("\nВідхилені моделі:")
    print(rejected.to_string(index=False) if len(rejected) else "(немає)")

    sel = save_selection(ranking, rejected, config, args.out, args.models_dir, config_path=args.config)
    gap = f"{sel['score_gap']:.4f}" if sel["score_gap"] is not None else "-"
    backup = sel["backup"]["model"] if sel["backup"] else "немає"
    print(f"\nОсновна модель: {sel['primary']['model']} (score {sel['primary']['score']:.4f}); "
          f"резервна: {backup}; відрив: {gap}")
    print(f"Файли збережено в {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
