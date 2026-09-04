# machine-learning-course

Лабораторні та практичні роботи з дисципліни **«Основи машинного навчання»**.

**Варіант 5** — набір даних Breast Cancer Wisconsin (Diagnostic), цільова ознака `diagnosis`.

> Перед здачею перейменуйте репозиторій на `machine-learning-course-<прізвище>`
> і впишіть автора нижче.

**Автор:** _(прізвище, група)_

## Структура

| Директорія | Робота | Зміст |
|---|---|---|
| [`practical01/`](practical01/) | Практична № 1 | аналіз даних (EDA) та побудова preprocessing-pipeline |
| `lab01/` | Лабораторна № 1 | _(далі)_ регресійне моделювання на підготовлених даних |

Кожна робота містить власний `README.md` (звіт), `notebook/` з виконаним
Jupyter Notebook, `figures/` з рисунками та `AI_USAGE.md`.

## Відтворення

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace practical01/notebook/practical01.ipynb
```

Набір даних окремо завантажувати не потрібно — він вбудований у `scikit-learn`
(`load_breast_cancer`). Усі випадкові операції зафіксовано `random_state=42`.

## Використання генеративного ШІ

Дозволено без обмежень; суттєві випадки зафіксовано у `*/AI_USAGE.md`.
Відповідальність за коректність усіх рішень, коду та висновків несе автор роботи.
