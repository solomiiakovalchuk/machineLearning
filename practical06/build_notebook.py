"""Generate the explanatory notebook; execute separately with nbconvert."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Практична робота № 6
## Інтерпретація та перевірка корисності результатів навчання без учителя
**Ковальчук · ІПЗс-24-1 · варіант 5 · Бедратюк Л.П.**

Працюю з фіксованими артефактами ЛР № 5: Breast Cancer Wisconsin, log1p → StandardScaler → PCA(6), K-Means(2).
Новий добір q/K/порогів за мітками не виконується. Мета — перевірити пояснення і обмежені дії, а не дати діагноз.

Виконані перевірки й код підготовлено за допомогою Codex; студентська перевірка та розуміння залишаються необхідними. Фактичні рішення та межі допомоги — в AI_USAGE.md.''')
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd()/'results').is_dir() else Path.cwd().parent
assert (ROOT/'results/summary.json').is_file()
sys.path.insert(0,str(ROOT))
import audit as A
def table(name): return pd.read_csv(ROOT/'results'/f'{name}.csv')
def picture(name): display(Image(filename=str(ROOT/'figures'/f'{name}.png')))
summary=json.loads((ROOT/'results/summary.json').read_text())
display(pd.Series(summary).to_frame('value'))''')
md('''## 1. Постановка, дані, протокол
Профілі рахую в **початкових сирих ознаках**, стандартизую відносно train; відстані — у робочому PCA(6).
Спершу перевірка контракту, профілі, представники і стійкість; потім PCA та scores; далі фіксовані правила;
тільки після цього відкриваю контрольні мітки спотворень для аудиту. Діагноз не є target правил.
Основний запуск: `python pr06_analysis.py`; цей notebook читає збережені результати й перевіряє їх.
Повторні підвибірки генерує RNG(42), кожен K-Means має random_state=42; preprocessing і PCA незмінні.''')
code('''validation=json.loads((ROOT/'results/validation.json').read_text())
display(validation)
assert validation['valid'] and validation['train_n']==455 and validation['test_n']==114
assert validation['k']==2 and validation['q']==6
display(table('cluster_summary').round(6))''')
md('''## 2. Профілі й розміри кластерів
Стандартизований профіль: (середнє кластера − середнє train) / SD(train), ddof=0.
Для сталої ознаки профіль нульовий. Повна long-таблиця зберігає середні, медіани та внутрішній SD.
Показую п'ять найбільших додатних і від'ємних відхилень окремо. Номер кластера не надає йому медичної назви.''')
code('''display(table('profile_top_signed').round(4))
picture('01_cluster_profiles')
assert len(table('cluster_profiles'))==60''')
md('''На heatmap видно протилежні напрямки числових профілів; це опис вимірювань, а не причин або діагнозів.
Середня відстань та RMS у PCA показують ширину групи окремо від її розміру.

## 3. Типові, віддалені й граничні
Типовий — найближчий **реальний запис** до центру; віддалений — найбільша відстань; граничний — малий
(d₂−d₁)/max(d₂, eps). Центроїд може мати значення, яких немає в жодного запису. Граничний не обов'язково віддалений.''')
code('''reps=table('representatives')
display(reps.round(5))
assert (reps.groupby(['cluster','kind']).size()==3).all()
picture('02_distance_boundary')''')
md('''## 4. Стійкість призначень і пояснень
Перевіряю дві різні зміни: п'ять попередніх seeds та 20 підвибірок по 90% train.
Номери кластерів вирівнюю угорським алгоритмом на train, а test прогнозую через те саме вирівнювання.
Контрольна межа S < 0.8; саме значення 0.8 не потрапляє до нестійких. Повторення seed не замінює зміни складу даних.''')
code('''display(pd.Series({k:summary[k] for k in ['seed_ari','seed_stability_min','subsample_ari','stability_mean','stability_min','unstable_n']}))
display(table('least_stable_10').round(6))
picture('03_stability')
ps=table('profile_stability')
display(ps.groupby('cluster').agg(jaccard_min=('jaccard','min'),jaccard_mean=('jaccard','mean'),max_profile_change=('max_abs_profile_change','max')))
picture('04_profile_stability')''')
md('''П'ять seeds дали стійкість кожного об'єкта 1.0, але підвибірки виявили два об'єкти нижче 0.8.
Top-5 ознак обох кластерів зберігаються (Jaccard=1), хоча числові профілі трохи змінюються.
Тому гіпотезу «стійке розбиття гарантує кожне призначення» відхилено, а гіпотезу про незмінність коротких профілів підтримано.
Таблиця десяти найменш стабільних містить і стабільні записи: нестійких лише два, не треба називати нестійкими всі десять.

## 5. PCA: ваги та крайні об'єкти
Ваги стосуються **стандартизованих log1p-ознак**. PC1 описує спільне збільшення розмірів та нерівності контуру;
PC2 — контраст фрактальних характеристик і розмірів; PC3 — контраст похибок вимірювань і частини worst-характеристик.
Це інтерпретаційні гіпотези, які перевіряю внесками крайніх об'єктів. Знак PCA умовний, причинність не випливає з ваг.''')
code('''loads=table('pca_loadings');ext=table('pca_extrema')
for pc in [1,2,3]:
    g=loads[loads.component.eq(pc)]
    print('PC',pc,'positive and negative loadings')
    display(pd.concat([g[g.weight>0].nlargest(5,'weight'),g[g.weight<0].nsmallest(5,'weight')]).round(5))
    print('Extreme records with strongest contributions')
    for side in ['min','max']:
        e=ext[ext.component.eq(pc)&ext.side.eq(side)]
        display(e.assign(abs_contribution=e.contribution.abs()).nlargest(5,'abs_contribution').drop(columns='abs_contribution').round(5))
picture('05_pca_signed_loadings')''')
md('''Великої ваги недостатньо: об'єкт може мати мале значення відповідної ознаки, тому його внесок буде малим.
Перевірка всіх внесків, а не лише top-5, наведена в CSV. Контрприклад предметному діагнозу: низька PCA-координата
не виключає злоякісності, бо компонента оптимізує дисперсію без діагнозу. Двовимірної проєкції для висновку недостатньо.

## 6. Scores, правила та картки
IF-PCA оцінює нетиповість усередині підпростору, reconstruction — відхилення поза ним.
Процентиль — частка train scores ≤ поточного; це не ймовірність спотворення.
Пріоритети та пороги читаються з JSON, усі одночасні умови журналюються.''')
code('''display(json.loads((ROOT/'configs/audit.json').read_text()))
display(json.loads((ROOT/'results/repair_evidence.json').read_text()))
picture('06_anomaly_signals')
cards=table('anomaly_cards')
display(cards)
assert len(cards)==15 and cards.id.nunique()==15
scores=table('scores_actions')
rules=json.loads((ROOT/'configs/audit.json').read_text())['policy']
check=A.apply_action_policy(scores,rules)
assert check.action.tolist()==scores.action.tolist()
assert not any('label' in rule['column'] or 'diagnos' in rule['column'] for rule in rules)''')
md('''Картки: п'ять top combined, п'ять розбіжностей IF-PCA/reconstruction та п'ять false positives;
усі 15 ID різні. Гіпотеза перевірки називає найбільший залишок ознаки, але не стверджує причину.

## 7. Наслідки та практична корисність
Контрольні спотворення відкриваються **після** обчислення action. FP тут означає «не ін'єктовано», а не «клінічно здоровий».
Фіксована політика направляє до оператора 24 об'єкти: 18 TP, 6 FP; пропущено 6 FN. Precision=recall=0.75.
Вісім конфліктів вирішено заданим порядком. Сценарні ціни — навчальні припущення, не виміряні економічні втрати.''')
code('''display(table('action_summary').round(5))
picture('07_actions')
display(pd.DataFrame([
 {'scenario':'Quality triage','user':'Оператор якості вимірювань','max_review_fraction':.30,'allowed_latency_ms':1000,'fp_cost':3,'fn_cost':20,'decision':'Обмежено: ручна перевірка, не відбракування'},
 {'scenario':'Automatic rejection','user':'Автоматичний імпортер записів','max_review_fraction':.10,'allowed_latency_ms':100,'fp_cost':100,'fn_cost':20,'decision':'Відхилено: FP і FN несумісні з автоматичною дією'}
]))
assert summary['review_n']==summary['tp']+summary['fp']
assert summary['test_n']==sum(summary[k] for k in ['tp','fp','fn','tn'])''')
md('''## 8. Висновки й межі
Результат придатний як **офлайнова допомога оператору**: за припущення 3 хвилини на запис це 72 хвилини роботи.
Автоматично можна обчислювати scores, пояснення й маршрут; виключення запису або медичне рішення потребує людини.
Операційна затримка всього контуру не вимірювалась: задані 1000/100 мс залишаються вимогами майбутнього прототипу.
Зупинка: порушення схеми/діапазонів, зміна розподілу, перевищення бюджету перевірок; пороги переглядати на новій валідації.
Обидва top-5 профілі стійкі, але ID 318 та 86 мають S=0.4. Предметні назви кластерів і причина реальної аномалії залишаються відкритими.
Тести: `python -m unittest discover -s tests -v`. Вхідний bundle відтворюється командою `python ../lab05/lr05_experiment.py` з кореня ПР.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
(ROOT/'notebook').mkdir(exist_ok=True)
nbf.write(nb,ROOT/'notebook/practical06.ipynb')
