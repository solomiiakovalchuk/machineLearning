# Вибір моделі: batch_second_reading

Нічне пакетне повторне прочитання всіх зразків за день (пакети по 256) на сервері лабораторії; пріоритет якості, затримка некритична.

Версія конфігурації: 1.0.0.

**Основна модель:** `logreg` (score = 0.8284).

**Резервна модель:** `svm_rbf` (score = 0.7776, відрив основної = 0.0508).

## Рейтинг допущених моделей

| rank | model | score | cv_roc_auc_mean | cv_f1_mean | cv_f1_std | fit_time_s | z_cv_roc_auc_mean | z_cv_f1_mean | z_cv_f1_std | z_fit_time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | logreg | 0.8284 | 0.9959 | 0.9703 | 0.0151 | 0.0130 | 0.9174 | 0.7031 | 0.6983 | 0.9987 |
| 2 | svm_rbf | 0.7776 | 0.9956 | 0.9588 | 0.0193 | 0.0848 | 0.9112 | 0.5880 | 0.6136 | 0.9915 |
| 3 | random_forest | 0.6832 | 0.9904 | 0.9533 | 0.0254 | 1.3735 | 0.8070 | 0.5332 | 0.4917 | 0.8626 |

## Відхилені моделі

| model | violations | n_violations |
|---|---|---|
| hist_gb | cv_recall_mean=0.9471 < min 0.95 | 1 |
| knn | cv_recall_mean=0.9294 < min 0.95 | 1 |
| tree | cv_recall_mean=0.9353 < min 0.95; cv_f1_mean=0.9085 < min 0.94 | 2 |

## Обмеження сценарію

- `cv_recall_mean`: min 0.95
- `cv_f1_mean`: min 0.94
- `latency_p95_ms_b256`: max 200.0
- `model_size_mb`: max 5.0

## Критерії та ваги

- `cv_roc_auc_mean` (maximize): вага 0.45, worst = 0.95, best = 1.0
- `cv_f1_mean` (maximize): вага 0.3, worst = 0.9, best = 1.0
- `cv_f1_std` (minimize): вага 0.15, worst = 0.05, best = 0.0
- `fit_time_s` (minimize): вага 0.1, worst = 10.0, best = 0.0
