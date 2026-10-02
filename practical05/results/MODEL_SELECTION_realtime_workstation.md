# Вибір моделі: realtime_workstation

Синхронна підказка на робочій станції цитолога: прогноз для одного зразка одразу після сегментації ядер; спільний CPU, 1 потік; модель постачається в пакеті оновлення ПЗ станції.

Версія конфігурації: 1.0.0.

**Основна модель:** `logreg` (score = 0.6186).

**Резервна модель:** `svm_rbf` (score = 0.4633, відрив основної = 0.1553).

## Рейтинг допущених моделей

| rank | model | score | cv_f1_mean | cv_recall_mean | latency_p95_ms_b1 | model_size_mb | cv_f1_std | z_cv_f1_mean | z_cv_recall_mean | z_latency_p95_ms_b1 | z_model_size_mb | z_cv_f1_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | logreg | 0.6186 | 0.9703 | 0.9647 | 4.0682 | 0.0023 | 0.0151 | 0.7031 | 0.2941 | 0.6591 | 0.9953 | 0.6983 |
| 2 | svm_rbf | 0.4633 | 0.9588 | 0.9588 | 6.8897 | 0.0248 | 0.0193 | 0.5880 | 0.1765 | 0.3456 | 0.9504 | 0.6136 |

## Відхилені моделі

| model | violations | n_violations |
|---|---|---|
| hist_gb | cv_recall_mean=0.9471 < min 0.95; latency_p95_ms_b1=12.19 > max 10.0 | 2 |
| knn | cv_recall_mean=0.9294 < min 0.95 | 1 |
| random_forest | latency_p95_ms_b1=91.94 > max 10.0 | 1 |
| tree | cv_recall_mean=0.9353 < min 0.95 | 1 |

## Обмеження сценарію

- `cv_recall_mean`: min 0.95
- `latency_p95_ms_b1`: max 10.0
- `model_size_mb`: max 0.5

## Критерії та ваги

- `cv_f1_mean` (maximize): вага 0.3, worst = 0.9, best = 1.0
- `cv_recall_mean` (maximize): вага 0.25, worst = 0.95, best = 1.0
- `latency_p95_ms_b1` (minimize): вага 0.25, worst = 10.0, best = 1.0
- `model_size_mb` (minimize): вага 0.1, worst = 0.5, best = 0.0
- `cv_f1_std` (minimize): вага 0.1, worst = 0.05, best = 0.0
