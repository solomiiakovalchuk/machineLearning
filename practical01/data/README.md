# Дані — Breast Cancer Wisconsin (Diagnostic)

## Джерело

Набір **не зберігається** у репозиторії — він вбудований у `scikit-learn` і
завантажується у коді:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame            # 569 × 31 (30 ознак + target)
```

Першоджерело: *Breast Cancer Wisconsin (Diagnostic) Data Set*, W. Wolberg,
W. Street, O. Mangasarian — UCI Machine Learning Repository
(<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>).

## Опис

* **569** об'єктів (новоутворення молочної залози), **30** числових ознак.
* Ознаки — характеристики клітинних ядер за оцифрованим зображенням FNA-біопсії:
  для 10 базових властовостей (`radius`, `texture`, `perimeter`, `area`,
  `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`,
  `fractal_dimension`) наведено 3 агрегати: `_mean`, `_se` (стандартна похибка),
  `_worst` (середнє 3 найбільших).
* **Цільова змінна `diagnosis`** у кодуванні `scikit-learn`:
  `0 = malignant` (злоякісне), `1 = benign` (доброякісне).
  ⚠️ Це протилежне до оригінального кодування UCI (`M` / `B`).
* Пропущених значень немає; дублікатів немає.

У цій роботі назви ознак приведено до формату `<властивість>_<mean|se|worst>`
(напр. `radius_mean`, `area_se`, `concave_points_worst`).

## Згенеровані артефакти (створюються ноутбуком)

| Файл | Вміст |
|---|---|
| `preprocessor.joblib` | навчений `ColumnTransformer` + метадані спліту (`random_state=42`, `test_size=0.2`, `stratify=True`) |
| `split_processed.npz` | `X_train (455×30)`, `X_test (114×30)`, `y_train`, `y_test`, `feature_names` — вхід для ЛР № 1 |

Обидва відтворюються запуском `notebook/practical01.ipynb` і не потребують
зберігання у git (див. `.gitignore`).
