"""Generate notebook/practical07.ipynb; execute separately:
jupyter nbconvert --to notebook --execute --inplace notebook/practical07.ipynb
The notebook only reads results/ and figures/ (heavy work is pr07_audit.py) and recomputes the headline test metrics live."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md('''# Практична робота № 7
## Критичний аудит ML-рішення, створеного за допомогою ШІ або AutoML
**Ковальчук · ІПЗс-24-1 · варіант 5 · Бедратюк Л.П.**

Об'єкт аудиту — рішення ЛР № 6 (AutoML FLAML → `Pipeline(SimpleImputer → StandardScaler → LogisticRegression(C=0.1))`) у зафіксованій версії
коміту `81445e2`; `lab06/` не змінювався. Notebook лише читає артефакти `results/` і `figures/`, а ключові метрики перераховує наживо;
повний запуск — `python pr07_audit.py`. Код і текст підготовлено за допомогою Claude Code; рішення й межі допомоги — в `AI_USAGE.md`.
Позитивний клас y=1 — malignant (у `load_breast_cancer` malignant закодовано як 0, тому ціль аудиту — `1 − target`).''')
code('''import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
from sklearn.datasets import load_breast_cancer

ROOT = Path.cwd() if (Path.cwd() / "results").is_dir() else Path.cwd().parent
assert (ROOT / "results" / "verdict.json").is_file(), "run pr07_audit.py first"
sys.path.insert(0, str(ROOT))
import audit_core as A

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)
R = ROOT / "results"
CFG = json.loads((ROOT / "configs" / "audit_config.json").read_text(encoding="utf-8"))
SUM = json.loads((R / "summary.json").read_text(encoding="utf-8"))
def table(name, **kw):
    return pd.read_csv(R / name, **kw)
def jload(name):
    return json.loads((R / name).read_text(encoding="utf-8"))
def picture(name):
    display(Image(filename=str(ROOT / "figures" / name)))
def say(text):
    display(Markdown(text))''')

md("## 1. Об'єкт аудиту, версія й комплектність")
code('''fro = table("frozen_object.csv")
say(f"Зафіксований коміт: `{CFG['object']['frozen_commit']}`. Файлів handoff: {len(fro)}; "
    f"байт-ідентичні: {SUM['frozen_info']['files_identical_bytes']}, відрізняються лише закінченнями рядків (autocrlf): "
    f"{SUM['frozen_info']['files_line_endings_only']}, змінені: {SUM['frozen_info']['files_different']}.")
display(fro[["file", "bytes", "worktree_vs_commit", "last_changed_in_commit", "sha256"]].assign(sha256=lambda d: d.sha256.str[:16] + "…"))
inv = table("inventory.csv")
say(f"Інвентаризація: присутні {int(inv.present.sum())} із {len(inv)} елементів.")
display(inv)''')

md("## 2. Реєстр тверджень")
code('''claims = table("claims.csv")
display(claims[["id", "claim", "evidence_level", "criteria_passed", "verdict"]])
say("Деталі (джерело, незалежна перевірка, критерії прийняття, доказ) — у `results/claims.csv`.")
picture("09_claims_findings.png")''')
md('''**Рисунок 9.** Ліворуч — розподіл вердиктів тверджень, праворуч — знахідок за критичністю. Жодне твердження не спростовано повністю,
але більшість підтверджена лише частково: числа відтворюються, а умови їх застосування (контракт входу, незалежність test, цілісність пакета) мають прогалини.''')

md("## 3. Чистий запуск")
code('''cr = jload("clean_run.json")
rows = [{"спроба": a["name"], "довжина шляху venv": a["venv_path_length"], "pip exit": a["pip_exit"], "pip, с": a["pip_s"],
         "пакетів": a["n_installed"], "запуск exit": a["run_exit"], "успіх": a["success"]} for a in cr["attempts"]]
display(pd.DataFrame(rows))
say(f"Python: {cr['attempts'][0]['python']}. У спробі A pip завершився помилкою довгого шляху (Windows), проте `inference.py` у частково встановленому "
    "venv дав коректний результат — тому успіх фіксується лише за трьома ознаками разом (venv, pip, запуск).")''')

md("## 4. Дані, leakage, протокол")
code('''da = jload("data_audit.json")
display(pd.Series({k: v for k, v in da.items() if k != "raw_sha256"}).to_frame("value"))
display(table("leakage_checks.csv"))
display(table("dataflow.csv"))
picture("01_leakage_correlations.png")''')
md('''**Рисунок 1.** Найбільша |r| з target на train — менша за поріг сигналу 0.9, тож проксі-ознак немає; кореляція лишається лише сигналом,
а висновок спирається на походження ознак і структуру протоколу (scaler навчено на train, preprocessing усередині Pipeline).
Негативні рядки таблиці — відсутність SHA-256 моделі та відсутність групових/часових ідентифікаторів.''')

md("## 5. Незалежні метрики, baseline, відтворення")
code('''ind = table("independent_metrics.csv")
display(ind.round(4))
rep = table("reproduction_delta.csv")
display(rep)
# live recomputation, independent of the stored tables
d = load_breast_cancer()
y = A.audit_target(d.target)
split = pd.read_csv(ROOT.parent / "lab06" / "handoff" / "split_ids.csv")
te = split[split.part == "test"].id.to_numpy()
model = joblib.load(ROOT.parent / "lab06" / "handoff" / "pipeline.joblib")
live = A.evaluate_model(model, d.data[te], y[te], 0.5)
say(f"Наживо: AP={live['ap']:.4f}, ROC-AUC={live['roc_auc']:.4f}, CM={live['confusion_matrix']}; "
    f"max |Δ| проти test_metrics.json = {rep[rep.metric != 'confusion_matrix'].abs_delta.max():.1e} (допуск {CFG['tolerances']['metric']:.0e}).")
picture("02_metrics_vs_baseline.png")
picture("04_confusion_matrices.png")''')
md('''**Рисунки 2 і 4.** Dummy-baseline дає AP, що дорівнює частці malignant у test, ROC-AUC 0.5 і жодного знайденого malignant (42 FN). Модель на test
пропускає 2 malignant і не має хибнопозитивних. Метрики відтворено до машинної точності; це доводить відтворюваність заявлених чисел, але не узагальнення.''')

md("## 6. Невизначеність: repeated CV і bootstrap на test")
code('''s = table("repeated_cv_summary.csv")
display(s[s.metric.isin(["ap", "roc_auc", "balanced_accuracy", "recall"])].round(4))
display(pd.Series(jload("test_bootstrap_ci.json")).to_frame("95% bootstrap CI на test"))
picture("03_repeated_cv.png")''')
md('''**Рисунок 3.** П'ятнадцять fold-оцінок AP фіксованої конфігурації лежать вузько біля 0.994, а інтервал середнього не містить baseline; test AP потрапляє
в діапазон CV. Інтервал описує мінливість folds (вони залежні) і не усуває систематичних обмежень протоколу. На test (42 malignant) інтервал recall значно ширший.''')

md("## 7. Перевірка рекомендацій ШІ / AutoML")
code('''display(table("ai_recommendation_checks.csv"))
display(table("selection_control.csv").round(4).head(8))
picture("10_selection_control.png")''')
md('''**Рисунок 10.** Лінії C=0.1 і C=1.0 майже збігаються, а найкраще C змінюється між seeds, тому вибір C=0.1 практично довільний (і так сформульовано в lab06).''')

md("## 8. Стійкість")
code('''runs = table("stability_runs.csv")
display(runs[["ap", "roc_auc", "recall", "precision", "disagreement_D", "mean_abs_dp", "max_abs_dp"]].describe().loc[["mean", "std", "min", "max"]].round(4))
imp = table("importance_summary.csv")
display(imp.head(8).round(4))
say(f"Ранжування важливості між перенавчаннями: Spearman {SUM['rank_stability']['rank_mean_spearman']:.3f}, top-5 Jaccard {SUM['rank_stability']['rank_mean_topk_jaccard']:.3f}.")
picture("05_stability.png")
picture("06_permutation_importance.png")''')
md('''**Рисунки 5 і 6.** AP майже не змінюється між перенавчаннями, а recall і конкретні мітки — змінюються (D до кількох відсотків, найбільший |Δp| на окремому об'єкті — десятки відсотків).
Важливість `texture_worst` найчастіше перша, але решта рангів нестійка (низький Jaccard), тож це чутливість моделі до перестановки, а не причина.''')

md("## 9. Аналіз помилок")
code('''err = table("error_objects.csv")
display(err.round(3))
picture("07_error_analysis.png")''')
md('''**Рисунок 7.** Обидва пропущені malignant (ID 73 і 205) лежать у зоні нижче порога; ID 205 ще й нестабільний між перенавчаннями. У CV на train найупевненіші помилки — теж FN.
Baseline цих помилок не виправляє: він пропускає всі malignant.''')

md("## 10. Негативні сценарії й виправлена версія")
code('''neg = table("negative_tests.csv")
display(neg[["scenario", "expected", "legacy_status", "legacy_ok", "hardened_status", "hardened_ok"]])
fe = jload("fixed_evaluation.json")
display(pd.Series({k: v for k, v in fe.items() if k not in ("sha256_file",)}).to_frame("fixed"))
picture("08_negative_tests.png")''')
md('''**Рисунок 8.** Оригінальний `inference.py` відхиляє нечислові, NaN/inf, від'ємні й неповні входи, але мовчки приймає переставлені колонки, числові рядки, значення поза діапазоном,
геометрично неузгоджені записи, надмірний batch і підмінену модель. Виправлена версія відповідає очікуванням у всіх сценаріях, а її прогнози на test ідентичні.''')

md("## 11. Знахідки та вердикт")
code('''display(table("findings.csv")[["id", "severity", "status", "problem"]])
v = jload("verdict.json")
say(f"**Вердикт для початкового рішення: {v['verdict']}.** Після виправлень: {v['verdict_after_fixes']}. "
    f"Тверджень: {v['claims']}; знахідок: {v['findings_by_severity']}; блокуючі: {v['basis']['blocking_findings']}.")''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
(ROOT / "notebook").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebook" / "practical07.ipynb")
print("notebook written")
