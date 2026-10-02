"""Відбір моделі за сценарієм: обмеження (eligibility) + зважена оцінка (ranking).

Модуль не знає нічого про sklearn-моделі: він працює з будь-якою таблицею
(DataFrame зі стовпцем `model` та числовими метриками) і конфігурацією сценарію.

Порядок: 1) жорсткі обмеження - модель, що порушила хоч одне, виключається
(і причини фіксуються); 2) серед допущених - нормалізація критеріїв у [0, 1]
за якорями worst/best із конфігурації і зважена сума; 3) детермінований
tie-break: за рівних балів - ім'я моделі за зростанням.
Якщо допущених моделей немає - NoEligibleModelError ("найменш погана" не обирається).
"""
from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd

DIRECTIONS = ("maximize", "minimize")
WEIGHT_TOL = 1e-9
REQUIRED_KEYS = ("scenario", "description", "version", "constraints", "criteria")
SCORE_ROUND = 12         # для tie-break: бали, що відрізняються < 1e-12, вважаються рівними


class NoEligibleModelError(ValueError):
    """Жодна модель не задовольняє обмеженням сценарію."""

    def __init__(self, violations: Mapping[str, List[str]]):
        self.violations = {k: list(v) for k, v in violations.items()}
        lines = [f"  {m}: {'; '.join(v)}" for m, v in self.violations.items()]
        super().__init__("жодна модель не пройшла обмеження сценарію:\n" + "\n".join(lines))


# ---------------------------------------------------------------- конфігурація
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_config(cfg: dict) -> dict:
    """Перевіряє структуру конфігурації; зайві ключі верхнього рівня ігноруються."""
    if not isinstance(cfg, dict):
        raise ValueError("конфігурація має бути JSON-об'єктом")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"у конфігурації відсутні обов'язкові ключі: {missing}")

    constraints = cfg["constraints"]
    if not isinstance(constraints, dict):
        raise ValueError("constraints має бути об'єктом {метрика: {min/max}}")
    for metric, limit in constraints.items():
        if not isinstance(limit, dict) or not ({"min", "max"} & set(limit)):
            raise ValueError(f"обмеження '{metric}': потрібен хоча б один із ключів min/max")
        unknown = set(limit) - {"min", "max"}
        if unknown:
            raise ValueError(f"обмеження '{metric}': невідомі ключі {sorted(unknown)}")
        for key, val in limit.items():
            if not _is_number(val):
                raise ValueError(f"обмеження '{metric}': {key}={val!r} має бути скінченним числом")
        if "min" in limit and "max" in limit and limit["min"] > limit["max"]:
            raise ValueError(f"обмеження '{metric}': min={limit['min']} > max={limit['max']}")

    criteria = cfg["criteria"]
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("criteria має бути непорожнім об'єктом {метрика: специфікація}")
    for name, spec in criteria.items():
        if not isinstance(spec, dict):
            raise ValueError(f"критерій '{name}': специфікація має бути об'єктом")
        absent = [k for k in ("direction", "weight", "worst", "best") if k not in spec]
        if absent:
            raise ValueError(f"критерій '{name}': відсутні ключі {absent}")
        if spec["direction"] not in DIRECTIONS:
            raise ValueError(
                f"критерій '{name}': direction={spec['direction']!r}, очікується одне з {DIRECTIONS}")
        for key in ("weight", "worst", "best"):
            if not _is_number(spec[key]):
                raise ValueError(f"критерій '{name}': {key}={spec[key]!r} має бути скінченним числом")
        if spec["weight"] < 0:
            raise ValueError(f"критерій '{name}': weight={spec['weight']} < 0")
        if spec["direction"] == "maximize" and not spec["best"] > spec["worst"]:
            raise ValueError(
                f"критерій '{name}': для maximize потрібно best > worst "
                f"(best={spec['best']}, worst={spec['worst']})")
        if spec["direction"] == "minimize" and not spec["best"] < spec["worst"]:
            raise ValueError(
                f"критерій '{name}': для minimize потрібно best < worst "
                f"(best={spec['best']}, worst={spec['worst']})")

    total = sum(spec["weight"] for spec in criteria.values())
    if abs(total - 1.0) > WEIGHT_TOL:
        raise ValueError(f"сума ваг критеріїв має дорівнювати 1, а не {total!r}")
    return cfg


def load_config(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(cfg)


# ---------------------------------------------------------------- таблиця результатів
def validate_results(results: pd.DataFrame, config: dict) -> None:
    """Перевіряє таблицю: стовпець model, унікальні імена, потрібні метрики числові й скінченні."""
    if "model" not in results.columns:
        raise ValueError("у таблиці результатів немає стовпця 'model'")
    dup = results["model"][results["model"].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"дублікати імен моделей у таблиці: {dup}")
    needed = list(dict.fromkeys(list(config["constraints"]) + list(config["criteria"])))
    for col in needed:
        if col not in results.columns:
            raise ValueError(f"у таблиці результатів немає стовпця метрики '{col}'")
        for model, value in zip(results["model"], results[col]):
            if isinstance(value, (bool, str, bytes)):
                raise ValueError(f"метрика '{col}' моделі '{model}' не числова: {value!r}")
            try:
                num = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"метрика '{col}' моделі '{model}' не числова: {value!r}") from None
            if not math.isfinite(num):
                raise ValueError(f"метрика '{col}' моделі '{model}' не скінченна (NaN/inf): {value!r}")


# ---------------------------------------------------------------- обмеження
def _fmt(x: float) -> str:
    return f"{float(x):.4g}"


def check_constraints(row, constraints: Mapping[str, Mapping[str, float]]) -> Tuple[bool, List[str]]:
    """Перевіряє один рядок на обмеження. Межі включні: value == limit допускається.

    Повертає (допущено, [усі порушення]) - перелік повний, не зупиняється на першому.
    """
    violations: List[str] = []
    for metric, limit in constraints.items():
        if metric not in row:
            raise ValueError(f"метрика '{metric}' відсутня в рядку")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError(f"метрика '{metric}' не скінченна: {value!r}")
        if "min" in limit and value < limit["min"]:
            violations.append(f"{metric}={_fmt(value)} < min {float(limit['min'])}")
        if "max" in limit and value > limit["max"]:
            violations.append(f"{metric}={_fmt(value)} > max {float(limit['max'])}")
    return (not violations), violations


# ---------------------------------------------------------------- нормалізація
def normalize_value(value: float, spec: Mapping) -> float:
    """Лінійна нормалізація у [0, 1]: worst -> 0, best -> 1; значення поза якорями обрізаються."""
    direction = spec.get("direction")
    worst, best = float(spec["worst"]), float(spec["best"])
    if direction == "maximize":
        if not best > worst:
            raise ValueError(f"maximize: потрібно best > worst (best={best}, worst={worst})")
        z = (float(value) - worst) / (best - worst)
    elif direction == "minimize":
        if not best < worst:
            raise ValueError(f"minimize: потрібно best < worst (best={best}, worst={worst})")
        z = (worst - float(value)) / (worst - best)
    else:
        raise ValueError(f"невідомий direction={direction!r}, очікується одне з {DIRECTIONS}")
    return min(1.0, max(0.0, z))


# ---------------------------------------------------------------- ранжування
def rank_models(results: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Повертає (ranking, rejected).

    ranking: rank, model, score, значення критеріїв, z_<критерій>; сортування за score
    спадно, за рівних балів - за ім'ям моделі зростаюче (детермінований tie-break).
    rejected: model, violations ("; "), n_violations.
    """
    validate_config(config)
    validate_results(results, config)
    criteria = config["criteria"]

    eligible_rows, rejected_rows = [], []
    for _, row in results.iterrows():
        ok, violations = check_constraints(row, config["constraints"])
        if ok:
            eligible_rows.append(row)
        else:
            rejected_rows.append({"model": row["model"], "violations": "; ".join(violations),
                                  "n_violations": len(violations)})
    rejected = pd.DataFrame(rejected_rows, columns=["model", "violations", "n_violations"])
    if not eligible_rows:
        raise NoEligibleModelError({r["model"]: r["violations"].split("; ") for r in rejected_rows})

    records = []
    for row in eligible_rows:
        rec = {"model": row["model"]}
        score = 0.0
        for name, spec in criteria.items():
            rec[name] = float(row[name])
            z = normalize_value(row[name], spec)
            rec[f"z_{name}"] = z
            score += spec["weight"] * z
        rec["score"] = score
        records.append(rec)
    ranking = pd.DataFrame(records)
    ranking["_key"] = ranking["score"].round(SCORE_ROUND)
    ranking = ranking.sort_values(["_key", "model"], ascending=[False, True], kind="mergesort")
    ranking = ranking.drop(columns="_key").reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    cols = ["rank", "model", "score"] + list(criteria) + [f"z_{c}" for c in criteria]
    return ranking[cols], rejected.sort_values("model").reset_index(drop=True)


# ---------------------------------------------------------------- чутливість до ваг
def reweight(config: dict, criterion: str, new_weight: float) -> dict:
    """Копія конфігурації: вага criterion = new_weight, решта масштабується пропорційно (сума = 1)."""
    criteria = config["criteria"]
    if criterion not in criteria:
        raise ValueError(f"критерію '{criterion}' немає в конфігурації")
    if not _is_number(new_weight) or not 0.0 <= new_weight <= 1.0:
        raise ValueError(f"нова вага має бути в [0, 1], отримано {new_weight!r}")
    others = sum(s["weight"] for n, s in criteria.items() if n != criterion)
    new_cfg = copy.deepcopy(config)
    if others <= 0:
        if abs(new_weight - 1.0) > WEIGHT_TOL:
            raise ValueError("інші ваги нульові: пропорційне масштабування неможливе")
        new_cfg["criteria"][criterion]["weight"] = 1.0
        return new_cfg
    factor = (1.0 - new_weight) / others
    for name, spec in new_cfg["criteria"].items():
        spec["weight"] = float(new_weight) if name == criterion else spec["weight"] * factor
    return new_cfg


def sensitivity_analysis(results: pd.DataFrame, config: dict, criterion: str,
                         grid: Iterable[float]) -> pd.DataFrame:
    """Для кожної ваги з grid - ранжування допущених моделей: weight, model, score, rank, winner."""
    rows = []
    for w in grid:
        ranking, _ = rank_models(results, reweight(config, criterion, w))
        for _, r in ranking.iterrows():
            rows.append({"weight": float(w), "model": r["model"], "score": float(r["score"]),
                         "rank": int(r["rank"]), "winner": int(r["rank"]) == 1})
    return pd.DataFrame(rows, columns=["weight", "model", "score", "rank", "winner"])


# ---------------------------------------------------------------- збереження
def _md_table(df: pd.DataFrame, floats: int = 4) -> str:
    def fmt(v):
        return f"{v:.{floats}f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def save_selection(ranking: pd.DataFrame, rejected: pd.DataFrame, config: dict, out_dir,
                   models_dir=None, config_path=None) -> dict:
    """Записує ranking/rejected CSV, selection JSON, короткий MD-звіт і копії pipeline.

    Якщо models_dir задано - primary/backup pipeline копіюються з нього у
    selected_<scenario>.joblib / backup_<scenario>.joblib.
    """
    scenario = config["scenario"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(out_dir / f"ranking_{scenario}.csv", index=False)
    rejected.to_csv(out_dir / f"rejected_{scenario}.csv", index=False)

    primary = ranking.iloc[0]
    backup = ranking.iloc[1] if len(ranking) > 1 else None
    selection = {
        "scenario": scenario,
        "config_version": config["version"],
        "config_path": str(config_path) if config_path is not None else None,
        "primary": {"model": primary["model"], "score": float(primary["score"])},
        "backup": ({"model": backup["model"], "score": float(backup["score"])}
                   if backup is not None else None),
        "score_gap": float(primary["score"] - backup["score"]) if backup is not None else None,
        "rejected": [{"model": r.model, "reasons": r.violations.split("; ")}
                     for r in rejected.itertuples(index=False)],
    }

    if models_dir is not None:
        models_dir = Path(models_dir)
        shutil.copyfile(models_dir / f"{primary['model']}.joblib", out_dir / f"selected_{scenario}.joblib")
        selection["primary"]["file"] = f"selected_{scenario}.joblib"
        if backup is not None:
            shutil.copyfile(models_dir / f"{backup['model']}.joblib", out_dir / f"backup_{scenario}.joblib")
            selection["backup"]["file"] = f"backup_{scenario}.joblib"

    (out_dir / f"selection_{scenario}.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Вибір моделі: {scenario}", "", config["description"], "",
             f"Версія конфігурації: {config['version']}.", "",
             f"**Основна модель:** `{primary['model']}` (score = {primary['score']:.4f}).", ""]
    if backup is not None:
        lines += [f"**Резервна модель:** `{backup['model']}` (score = {backup['score']:.4f}, "
                  f"відрив основної = {selection['score_gap']:.4f}).", ""]
    else:
        lines += ["Резервної моделі немає: допущена лише одна.", ""]
    lines += ["## Рейтинг допущених моделей", "", _md_table(ranking), ""]
    lines += ["## Відхилені моделі", ""]
    lines += [_md_table(rejected), ""] if len(rejected) else ["Усі моделі пройшли обмеження.", ""]
    lines += ["## Обмеження сценарію", ""]
    for metric, lim in config["constraints"].items():
        parts = ([f"min {lim['min']}"] if "min" in lim else []) + ([f"max {lim['max']}"] if "max" in lim else [])
        lines.append(f"- `{metric}`: {', '.join(parts)}")
    lines += ["", "## Критерії та ваги", ""]
    for name, spec in config["criteria"].items():
        lines.append(f"- `{name}` ({spec['direction']}): вага {spec['weight']}, "
                     f"worst = {spec['worst']}, best = {spec['best']}")
    (out_dir / f"MODEL_SELECTION_{scenario}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selection
