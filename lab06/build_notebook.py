"""Generate the explanatory notebook from immutable experiment results."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Лабораторна робота № 6
## AutoML і ручне моделювання: контрольоване порівняння
**Ковальчук · ІПЗс-24-1 · варіант 5 · Бедратюк Л.П.**

UCI Optdigits (`optdigits.tra`): цифра **5 проти решти**. Новий набір обрано, щоб не повторювати вже відкритий test Breast Cancer.
Методичка не задає окремого набору для варіанта 5; це навчальний вибір задачі.
Код і перевірки підготовлені з Codex; журнал відрізняє виконані агентом перевірки від особистої перевірки студентки.''')
code('''from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
from IPython.display import display,Image
ROOT=Path.cwd() if (Path.cwd()/'results').is_dir() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
import experiment as E
def table(name): return pd.read_csv(ROOT/'results'/f'{name}.csv')
def record(name): return json.loads((ROOT/'results'/f'{name}.json').read_text(encoding='utf-8'))
def picture(name): display(Image(filename=str(ROOT/'figures'/f'{name}.png')))
display(record('environment'))
display(record('train_eda'))''')
md('''## 1. Постановка та протокол до результатів
Користувач — оператор OCR; рішення — показати ймовірність «це 5» і запросити ручне підтвердження.
Не застосовувати для автоматичного читання фінансових документів. Вхід — 64 пікселі 0–16, target = digit==5.
AP є основною метрикою через частку позитивних близько 10%; поріг 0.5 заданий до запуску.
Додаткові: ROC-AUC, F1, precision, recall, Brier, log loss.

Holdout 20%, stratify, seed=42. На решті — nested CV: 5 зовнішніх × 3 внутрішніх folds.
Кожна гілка має 12 кандидатів, один потік, однакові індекси та Pipeline(imputer, scaler, model).
Практична межа ε=0.01 AP; якщо середня парна перевага не перевищує її, tie-break: розмір → p95 latency → Manual.
Це правило рішення, не статистичний доказ еквівалентності.''')
code('''display(record('search_contract'))
splits=table('split_ids')
assert not set(splits[splits.part.eq('train')].id)&set(splits[splits.part.eq('test')].id)
display(splits.groupby('part').size().to_frame('n'))''')
md('''## 2. Дві процедури пошуку
Manual: чотири логістичні регресії (C), чотири RF (глибина/лист), чотири HistGB (крок/ітерації/листя).
FLAML 2.3.6: lrl2, rf, extra_tree, histgb, kneighbor; CFO, roundrobin, max_iter=12, metric='ap',
log_type='all', sample=False, ensemble=False, skip_transform=True, retrain_full=True.
Власні адаптери BaseEstimator **тільки** забезпечують однаковий Pipeline всередині fold; конфігурації обирає FLAML.
FrozenCV повертає зафіксовані позиційні індекси і не потребує y у split(X), якого викликає FLAML.

Попередній API smoke-test на перших 600 sklearn Digits перетнувся з початковим holdout.
Цей експеримент відкинуто; його результат збережено в commit `1a3af08`.
Фінальний експеримент використовує **optdigits.tra**, іншу частину UCI (3823 записи),
яку не використовував smoke-test. Перетин ознак із попереднім джерелом перевіряється до split.
Метрика, ε, простір кандидатів, бюджет і правило вибору при цьому не змінювалися.''')
code('''display(record('protocol_repair'))
provenance=record('data_provenance');display(provenance)
assert provenance['exact_feature_overlap_previous_source']==0
display(table('candidate_history').groupby(['stage','branch']).agg(trials=('trial','size'),best_inner_ap=('inner_ap','max')))
display(table('budget_audit').round(5))
audit=table('flaml_fit_audit')
assert (audit.n==audit.scaler_n).all()
assert len(audit)==296
assert (table('budget_audit').cv_fits==36).all()''')
md('''Завершених пошукових CV-fit однаково: 36 на пошук; FLAML додатково робить один автоматичний refit.
Обидві гілки потім однаково перевчаються для outer-прогнозу / фіналіста. Отже, рівність бюджету стосується
порівнянь кандидатів; накладна робота AutoML більша на один fit, час наведений повністю, а не «нормований» приховано.
Невдалий trial, неповний бюджет або нечислова втрата зупиняють запуск.

## 3. Зовнішня якість, парні різниці та невизначеність''')
code('''display(table('fold_metrics').round(6))
display(table('cv_summary').round(6))
display(record('paired_comparison'))
picture('01_ap_distribution')
picture('02_paired_ap')
oof=table('oof_predictions')
assert len(oof)==3058 and oof.id.is_unique and not oof.isna().any().any()
for row in table('fold_metrics').itertuples():
    g=oof[oof.fold.eq(row.fold)]
    assert np.isclose(E.probability_metrics(g.y,g[row.branch])['ap'],row.ap)''')
md('''Обидві гілки порівнюються з Dummy prior. Рішення спирається на парну різницю та наперед заданий ε.
П'ять залежних fold-ів — малий матеріал для невизначеності; bootstrap тут описує мінливість середніх
на цих fold-ах, а не дає незалежний популяційний інтервал. Нуль усередині інтервалу не доводить рівність.

## 4. Стійкість пошуку й OOF-поведінка
Три фінальні searches змінюють seed splitter та FLAML; усі моделі з випадковістю мають random_state=42.
Перший seed=42 визначений операційним заздалегідь; інші не використовуються для перевибору після test.
Розбіжність фінальних моделей на train — описова, оптимістична для узагальнення; якість оцінюється outer OOF.''')
code('''display(table('search_stability').round(6))
display(record('seed_disagreement'))
display(table('behavior_summary'))
display(table('largest_disagreements').round(6))
picture('04_confusion_matrices')
picture('03_calibration')
picture('06_probability_disagreement')
picture('07_disagreement_objects')''')
md('''Стійкість сімейства та параметрів оцінюється за трьома searches у таблиці вище.
OOF-категорії відрізняють «дві неправильні» від «дві правильні»: сама згода не є якістю.
Великі розбіжності перевіряються за конкретними ID, істинним класом, ймовірностями та зображенням.
Висока прогнозована ймовірність не є гарантією; calibration plot і Brier перевіряють інший аспект, ніж AP.

## 5. Важливість на відкладених folds
Permutation importance рахується тільки на outer-validation, 5 перестановок, seed=42.
Показую середні top-10 та Jaccard. Корельовані пікселі можуть ділити важливість; мінусове падіння можливе через шум.
Це чутливість прогнозу, а не причинний ефект пікселя.''')
code('''imp=table('permutation_importance').groupby(['branch','feature']).importance.mean().unstack('branch')
display(imp.nlargest(10,'manual').round(6))
display(imp.nlargest(10,'automl').round(6))
print('Top-10 Jaccard:',record('summary')['top10_jaccard'])
picture('05_permutation_importance')''')
md('''## 6. Інженерні характеристики і рішення
Обидва фіналісти збережені як повні sklearn Pipeline без стану AutoML-пошуку.
Latency: 7 прогрівів, 45 вимірювань, один потік; пакети 1 і 256. Невелика різниця затримки залежить від машини.
Рішення записане до першого фінального predict(test), test_metrics_present=False; SHA-256 контролює незмінність.
Хеш сам по собі не доводить час створення або те, що людина ніколи не бачила даних.''')
code('''display(table('engineering').round(6))
decision=record('decision_record');display(decision)
digest=E.sha256_file(ROOT/'results/decision_record.json')
print('Decision SHA-256:',digest)
assert digest==(ROOT/'results/decision.sha256').read_text().strip()
assert decision['test_metrics_present'] is False
eng=table('engineering')
if abs(decision['cv_evidence']['mean_difference'])<=decision['epsilon']:
    expected=eng.sort_values(['size_bytes','single_p95_ms','branch']).iloc[0].branch
    assert expected==decision['selected_branch']''')
md('''Фактичне послідовне правило видно в config, engineering.csv і перевірці вище. Первинний JSON збережено без редагування після test.
На іншому обладнанні вибір у tie-регіоні може змінитись; для перевірки саме цього рішення слід читати збережені виміри.

## 7. Test і пакет передачі
Фінальний test оцінюється для вибраної гілки один раз у початковому запуску нового джерела.
Повторне відтворення коду знову обчислить оцінку, але не створює новий незалежний експеримент.''')
code('''display(record('test_metrics'))
t=table('test_predictions');actual=E.probability_metrics(t.y,t.probability)
for key,value in actual.items(): assert np.isclose(value,record('test_metrics')[key])
print('Handoff files:',sorted(p.name for p in (ROOT/'handoff').iterdir()))
ex=json.loads((ROOT/'handoff/inference_example.json').read_text())
import joblib
from inference import predict
loaded=joblib.load(ROOT/'handoff/pipeline.joblib')
np.testing.assert_allclose(predict(loaded,ex['input']),ex['expected_probability'],rtol=1e-10,atol=1e-12)
assert record('test_metrics')['decision_sha256']==digest''')
md('''## 8. Висновок
Гіпотезу «найбільша середня AP автоматично визначає кращу інженерну модель» відхилено заданим ε і tie-break.
Manual зберігає сімейство RF у трьох searches, але параметри змінюються. AutoML обирає три різні сімейства:
Extra Trees, KNN і RF; гіпотезу про стійкість сімейства AutoML відхилено, навіть за невеликої розбіжності train-прогнозів.
Фактичні test AP, F1, recall і матриця помилок наведені вище; модель після test не змінюється.
Порушення незалежності початкового smoke-test виправлено новим джерелом. Немає writer-ID для групового поділу, історична колекція мала.
До ЛР № 7 передано модель, схему, конфігурацію, split, результати, model card, залежності й контрольний інференс.
Потрібні незалежні автори/сканери та перевірка порога для реального OCR-сценарію. Команди й тести — у README.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
(ROOT/'notebook').mkdir(exist_ok=True);nbf.write(nb,ROOT/'notebook/lab06.ipynb')
