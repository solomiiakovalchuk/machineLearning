"""Generate the explanatory notebook. It only reads saved results (CSV/JSON/PNG) and verifies them."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Лабораторна робота № 6
## AutoML і ручне моделювання: контрольоване порівняння
**Ковальчук · ІПЗс-24-1 · варіант 5 · Бедратюк Л.П.**

Breast Cancer Wisconsin (Diagnostic), `sklearn.load_breast_cancer`; позитивний клас y=1 — malignant.
Набір обрано для наступності з варіантом 5, ЛР № 5 і ПР № 5–6; методичка ЛР № 6 (п. 4.1) таблиці варіантів не містить.
Цей набір уже використовувався в попередніх роботах курсу, тому test незалежний лише в межах цього експерименту.
Не діагностичний інструмент: клінічне використання заборонено.

Notebook **нічого не навчає і не шукає заново**: він читає збережені CSV/JSON/PNG і перевіряє їхню узгодженість.
Код і результати отримано в сесії Claude Code (див. `AI_USAGE.md`); студент відповідає за перевірку результату.''')
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
Користувач — експерт/оператор лабораторії, який переглядає записи FNA; результат — ймовірність malignant і ранжування для ручної перевірки.
Це не діагноз і не автоматичне клінічне рішення.
AP — основна метрика: вона не залежить від порога й оцінює, наскільки високо ранжовані позитивні записи.
Додаткові: ROC-AUC, F1, precision, recall, balanced accuracy, Brier, log loss; поріг 0.5 заданий до запуску.

Holdout 20% (stratify, seed=42). На решті — nested CV: 5 зовнішніх × 3 внутрішніх folds.
Кожна гілка має 12 кандидатів, один потік, однакові індекси та `SimpleImputer → StandardScaler → модель`.
Практична межа ε=0.01 AP; якщо середня парна різниця не перевищує її, tie-break: розмір → p95 latency → Manual.
Це правило рішення, а не статистичний доказ еквівалентності.''')
code('''cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
display(cfg)
assert cfg['test_used_for_search'] is False and cfg['test_used_for_selection'] is False
contract=record('search_contract')
display(pd.DataFrame([{'candidate_id':s['candidate_id'],'family':s['family'],'params':json.dumps(s['params'])} for s in contract['manual']]))
display(pd.DataFrame({k:{p:str(v) for p,v in d.items()} for k,d in contract['flaml_space'].items()}).T)
splits=table('split_ids')
assert not set(splits[splits.part.eq('train')].id)&set(splits[splits.part.eq('test')].id)
assert sorted(splits.id)==list(range(569))
X,y,names=E.load_data()
tr=splits[splits.part.eq('train')].id.to_numpy();te=splits[splits.part.eq('test')].id.to_numpy()
assert (len(tr),len(te))==(455,114) and (int(y[tr].sum()),int(y[te].sum()))==(170,42)
display(pd.DataFrame({'n':[len(tr),len(te)],'malignant':[int(y[tr].sum()),int(y[te].sum())]},index=['train','test']))''')
md('''## 2. Дані, походження та повторне використання
Джерело, ліцензія CC BY 4.0, хеш сирих даних і примітка про повторне використання — у `data_provenance.json`.
Для 114 test-ID перетин зі split ЛР № 5 порахований лише за ID (ознаки й моделі ЛР № 5 не читалися).
Конфігурація простору, метрики, ε і tie-break зафіксована до цього запуску й не підбиралась за результатами попередніх робіт.''')
code('''provenance=record('data_provenance');display(provenance)
assert provenance['raw_x_target_sha256']==E.raw_data_sha256()
assert provenance['n']==569 and provenance['duplicate_rows']==0
ov=provenance['lab05_split_overlap']
assert ov['lab06_test_ids_in_lab05_train']+ov['lab06_test_ids_in_lab05_test']==ov['lab06_test_n']
print('lab06 test IDs: in lab05 train =',ov['lab06_test_ids_in_lab05_train'],', in lab05 test =',ov['lab06_test_ids_in_lab05_test'])''')
md('''## 3. Дві процедури пошуку та бюджет
Manual: 12 наперед заданих конфігурацій (4 LogisticRegression, 4 RandomForest, 4 HistGradientBoosting), усі оцінені на тих самих inner folds.
AutoML: готовий FLAML 2.3.6 (CFO, `learner_selector='roundrobin'`, `max_iter=12`, `metric='ap'`, `log_type='all'`), п'ять сімейств.
Власні адаптери лише забезпечують fold-local Pipeline; конфігурації обирає FLAML.''')
code('''hist=table('candidate_history')
display(hist.groupby(['stage','branch']).agg(trials=('trial','size'),best_inner_ap=('inner_ap','max')).round(6))
budget=table('budget_audit')
display(budget.round(4))
audit=table('flaml_fit_audit')
assert (audit.n==audit.scaler_n).all() and len(audit)==296
assert (budget.candidates==12).all() and (budget.cv_fits==36).all()
assert (budget[budget.branch.eq('automl')].framework_refits==1).all() and (budget[budget.branch.eq('manual')].framework_refits==0).all()
assert len(hist)==192
for path in sorted((ROOT/'results/logs').glob('*.jsonl')):
    n=sum('validation_loss' in json.loads(s) for s in path.read_text(encoding='utf-8').splitlines())
    assert n==12,(path.name,n)
print('Search seconds total:',budget.groupby('branch').search_seconds.sum().round(3).to_dict())
print('Mean trial seconds by family:');display(hist.groupby(['branch','family']).seconds.mean().round(3))''')
md('''Бюджет порівняння кандидатів однаковий: 12 кандидатів × 3 inner folds = 36 CV-fit на пошук. FLAML додатково робить один
внутрішній refit; наш контур потім однаково перевчає переможця кожної гілки. Час наведено без нормування: він залежить
від складу простору й початкових значень, а не лише від «якості» методу. Невдалий trial, нечислова втрата чи неповний бюджет зупиняють запуск.

## 4. Зовнішня якість, парні різниці та невизначеність''')
code('''fm=table('fold_metrics');display(fm.round(4))
display(table('cv_summary').round(4))
cmp_=record('paired_comparison');display(cmp_)
picture('01_ap_distribution')
picture('02_paired_ap')
oof=table('oof_predictions')
assert len(oof)==455 and oof.id.is_unique and not oof.isna().any().any()
for row in fm.itertuples():
    g=oof[oof.fold.eq(row.fold)]
    assert np.isclose(E.probability_metrics(g.y,g[row.branch])['ap'],row.ap)
ap=fm.pivot(index='fold',columns='branch',values='ap')
d=(ap.automl-ap.manual).to_numpy()
assert np.allclose(d,cmp_['difference_automl_minus_manual']) and np.isclose(d.mean(),cmp_['mean_difference'])
assert np.allclose(E.bootstrap_mean_interval(d),cmp_['bootstrap_interval_95'])
print('mean d =',round(d.mean(),6),' within epsilon:',abs(d.mean())<=cmp_['epsilon'],' max |d_j| =',round(np.abs(d).max(),6))
assert (ap.manual>ap.baseline).all() and (ap.automl>ap.baseline).all()''')
md('''Обидві гілки на всіх п'яти outer-folds значно перевищують prior baseline. Середня парна різниця AutoML − Manual менша за ε,
інтервал bootstrap містить нуль, і жодна окрема різниця не перевищує ε. П'ять залежних folds — малий матеріал:
bootstrap тут описовий, а не доказ еквівалентності. Для folds 1 і 4 обидві гілки вибрали однаковий pipeline, тому різниця дорівнює нулю.

## 5. Стійкість пошуку й OOF-поведінка
Три фінальні searches змінюють seed splitter і FLAML; моделі з випадковістю мають random_state=42.
Seed=42 наперед визначено операційним; 43 і 44 не використовуються для перевибору після test.
Розбіжність фінальних моделей на train — описова й оптимістична для узагальнення; якість оцінюється outer OOF.''')
code('''display(table('search_stability').round(6))
display(record('seed_disagreement'))
ow=pd.DataFrame([{'fold':r['fold'],'branch':r['branch'],'family':r['spec']['family'],'params':json.dumps(r['spec']['params']),'inner_ap':round(r['inner_ap'],6)} for r in json.loads((ROOT/'results/outer_winners.json').read_text(encoding='utf-8'))])
display(ow)
stab=table('search_stability')
assert (stab.family=='lrl2').all()
display(table('behavior_summary'))
display(table('largest_disagreements').round(6))
b=oof.copy();pm=b.manual>=.5;pa=b.automl>=.5;cm_=pm==b.y;ca=pa==b.y
cat=np.select([cm_&ca,~cm_&~ca,cm_&~ca],['both_correct','both_wrong','only_manual_correct'],default='only_automl_correct')
bs=table('behavior_summary').set_index('category').n.to_dict()
assert pd.Series(cat).value_counts().to_dict()==bs and sum(bs.values())==455
beh=table('oof_behavior')
print('records with probability gap >0.2:',int((beh.probability_gap>0.2).sum()),'| >0.5:',int((beh.probability_gap>0.5).sum()))
picture('04_confusion_matrices')
picture('03_calibration')
picture('06_probability_disagreement')
picture('07_disagreement_objects')
picture('08_candidate_history')''')
md('''Обидві гілки у всіх трьох фінальних searches вибрали `lrl2`; змінюються лише параметри (Manual: C=1.0; AutoML: C=0.1,
`class_weight` null для seed 42 і balanced для 43/44). Тому на цьому наборі гіпотеза про нестійке сімейство переможця **не підтверджується**.
Це властивість саме цього набору й простору, а не загальне твердження про AutoML.
У OOF мітки на порозі 0.5 різняться лише в 4 записах із 455; 12 записів помиляються в обох гілках.
Усі п'ять найбільших розбіжностей ймовірностей — з fold 2, де Manual обрав HistGradientBoosting, а AutoML — логістичну регресію.
Історія кандидатів (рисунок 8) — діагностика пошуку; вона обчислена на тих самих inner folds, що й добір, тому
не придатна для повторного вибору моделі.

## 6. Важливість на відкладених folds
Permutation importance рахується тільки на outer-validation, 5 перестановок, seed=42, показник — падіння AP.
Корельовані ознаки можуть ділити важливість; це чутливість прогнозу, а не причинний ефект.''')
code('''imp=table('permutation_importance');assert len(imp)==300
means=imp.groupby(['branch','feature']).importance.mean().unstack('branch')
display(means.nlargest(10,'manual').round(6))
display(means.nlargest(10,'automl').round(6))
a=set(means.manual.nlargest(10).index);c=set(means.automl.nlargest(10).index)
print('Top-10 Jaccard:',len(a&c),'/',len(a|c),'=',round(len(a&c)/len(a|c),6),sorted(a&c))
assert np.isclose(len(a&c)/len(a|c),record('summary')['top10_jaccard'])
picture('05_permutation_importance')''')
md('''## 7. Інженерні характеристики і рішення до test
Обидва фіналісти збережені як повні sklearn Pipeline без стану AutoML-пошуку.
Latency: 7 прогрівів, 45 вимірювань, один потік; пакети 1 і 256; одне середовище, p95 з 45 вимірювань чутливий до поодиноких викидів.
Рішення записане й захешоване до першого `predict` на test; `test_metrics_present=False`.''')
code('''eng=table('engineering');display(eng.round(5))
decision=record('decision_record');display({k:decision[k] for k in ['selected_branch','candidate_id','family','params','epsilon','tie_break','reason','test_metrics_present']})
digest=E.sha256_file(ROOT/'results/decision_record.json')
print('Decision SHA-256:',digest)
assert digest==(ROOT/'results/decision.sha256').read_text().strip()
assert decision['test_metrics_present'] is False
assert decision['config_sha256']==E.sha256_file(ROOT/'configs/experiment.json')
delta=decision['cv_evidence']['mean_difference']
if abs(delta)<=decision['epsilon']:
    expected=eng.sort_values(['size_bytes','single_p95_ms','branch']).iloc[0].branch
    assert expected==decision['selected_branch']
print('size difference, bytes:',int(eng.size_bytes.max()-eng.size_bytes.min()),'| first criterion decided the tie-break')''')
md('''Рішення прийнято механічно за наперед заданим правилом: різниця AP в межах ε, тож обрано менший файл — AutoML (2171 проти 2174 байт).
Різниця в 3 байти практично не має змісту, а якість еквівалентна; обидва фіналісти — логістична регресія L2, тому критерій
розміру тут вироджений. Правило не змінювалось після результатів. p95 single-latency відрізняється, але до цього критерію
правило не дійшло; p50 майже однаковий.

## 8. Test і пакет передачі
Фінальний test оцінюється для вибраної гілки один раз. Інший фіналіст оцінений на test лише після заморожування рішення,
виключно для аудиту; він не впливав на вибір.''')
code('''tm=record('test_metrics');display(tm)
t=table('test_predictions');actual=E.probability_metrics(t.y,t.probability)
for key,value in actual.items(): assert np.isclose(value,tm[key])
assert sorted(t.id)==sorted(te) and tm['n']==114 and tm['positive_n']==42
assert tm['decision_sha256']==digest
other=record('test_audit_other_finalist');display(other)
assert other['AUDIT_ONLY'] is True and other['branch']!=decision['selected_branch'] and other['decision_sha256']==digest
print('Handoff files:',sorted(p.name for p in (ROOT/'handoff').iterdir()))
assert E.sha256_file(ROOT/'handoff/decision_record.json')==digest
ex=json.loads((ROOT/'handoff/inference_example.json').read_text())
import joblib
from inference import predict
loaded=joblib.load(ROOT/'handoff/pipeline.joblib')
np.testing.assert_allclose(predict(loaded,ex['input']),ex['expected_probability'],rtol=1e-10,atol=1e-12)
cv=table('cv_summary').set_index('branch').loc[decision['selected_branch']]
print('CV AP',round(cv.ap_mean,4),'+-',round(cv.ap_std,4),'| test AP',round(tm['ap'],4))''')
md('''## 9. Висновок
Обидві гілки перевищили baseline. Середня парна різниця AP AutoML − Manual (0.0020) менша за ε=0.01, інтервал bootstrap містить нуль,
тож якість практично еквівалентна; це не доказ еквівалентності. Наперед заданий tie-break за розміром обрав AutoML (логістична регресія L2, C=0.1)
через 3 байти різниці — вибір практично довільний. Це не означає, що AutoML кращий.
Обидві гілки стабільно обирали `lrl2` для seeds 42, 43, 44; це властивість цього набору.
Test: AP 0.9966, F1 0.9756, recall 0.9524, матриця [[72, 0], [2, 40]]; другий фіналіст оцінений лише для аудиту.
Обмеження для ЛР № 7: одне джерело з 569 записів без групування за пацієнтами, фіксований поріг 0.5, повторне використання набору в курсі,
мала історія пошуку, невелика нестійкість рангу важливості. Поточний test не слід використовувати для подальшого підбору порога чи калібрування.
Команди й тести — у README.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
(ROOT/'notebook').mkdir(exist_ok=True);nbf.write(nb,ROOT/'notebook/lab06.ipynb')
