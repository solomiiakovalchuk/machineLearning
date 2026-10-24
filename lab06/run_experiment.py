"""One-command nested CV, decision freeze and LR7 handoff. Test is opened once."""
from __future__ import annotations
import json, platform, time, shutil, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn, scipy, flaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import average_precision_score,confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
import experiment as E

ROOT=Path(__file__).resolve().parent


def run():
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    R,F,H=ROOT/'results',ROOT/'figures',ROOT/'handoff'
    for p in [R,F,H,R/'logs',R/'models']:p.mkdir(exist_ok=True,parents=True)
    def dump(value,path):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item()),encoding='utf-8')
    def save(df,name):df.to_csv(R/f'{name}.csv',index=False)
    def figure(name):plt.tight_layout();plt.savefig(F/f'{name}.png',dpi=150);plt.close()
    Xall,digits=load_digits(return_X_y=True);yall=(digits==5).astype(int)
    train_ids,test_ids=train_test_split(np.arange(len(yall)),test_size=cfg['test_size'],stratify=yall,random_state=42)
    assert not set(train_ids)&set(test_ids)
    # No EDA, model fitting or prediction accesses Xall[test_ids] before decision freeze below.
    X,y=Xall[train_ids],yall[train_ids]
    names=[f'pixel_{i//8}_{i%8}' for i in range(64)]
    split=pd.DataFrame({'id':np.r_[train_ids,test_ids],'part':['train']*len(train_ids)+['test']*len(test_ids)})
    save(split,'split_ids')
    dump({'train_n':len(X),'features':X.shape[1],'positive_n':int(y.sum()),'missing_n':int(np.isnan(X).sum()),
          'constant_features':[names[i] for i in np.flatnonzero(X.std(axis=0)==0)],
          'min':float(X.min()),'max':float(X.max()),'positive_class':'digit 5'},R/'train_eda.json')
    dump({'manual':E.manual_specs(),'flaml_space':E.SPACE,'config':cfg},R/'search_contract.json')
    oof=pd.DataFrame({'id':train_ids,'y':y,'fold':-1,'manual':np.nan,'automl':np.nan,'baseline':np.nan})
    folds=[];histories=[];budgets=[];importance=[];fitted_specs=[];split_contract=[];fit_audits=[]
    outer=list(StratifiedKFold(cfg['outer_folds'],shuffle=True,random_state=42).split(X,y))
    def search_pair(x,labels,inner,seed,stage):
        results={}
        for branch in ['manual','automl']:
            if branch=='manual':
                spec,history,seconds=E.evaluate_candidate_specs(E.manual_specs(),x,labels,inner)
                extra_fits=0
            else:
                spec,history,seconds,audit=E.evaluate_flaml(x,labels,inner,seed,cfg['budget'],R/'logs'/f'{stage}.jsonl')
                extra_fits=len(audit)-len(history)*len(inner)
                assert extra_fits>=0
                fit_audits.extend([dict(stage=stage,**row) for row in audit])
            history['branch']=branch;history['stage']=stage;history['search_seed']=seed;histories.append(history)
            budgets.append({'stage':stage,'branch':branch,'candidates':len(history),'inner_folds':len(inner),
                            'cv_fits':len(history)*len(inner),'framework_refits':extra_fits,'search_seconds':seconds,'threads':1})
            results[branch]=(spec,history.inner_ap.max(),seconds)
        return results
    for fold,(tr,va) in enumerate(outer):
        inner=list(StratifiedKFold(cfg['inner_folds'],shuffle=True,random_state=42).split(X[tr],y[tr]))
        split_contract.append({'stage':f'outer_{fold}','outer_train_ids':train_ids[tr].tolist(),'outer_val_ids':train_ids[va].tolist(),
                               'inner':[{'train_ids':train_ids[tr[a]].tolist(),'val_ids':train_ids[tr[b]].tolist()} for a,b in inner]})
        choices=search_pair(X[tr],y[tr],inner,42,f'outer_{fold}')
        oof.loc[va,'fold']=fold
        baseline=DummyClassifier(strategy='prior').fit(X[tr],y[tr]);basep=baseline.predict_proba(X[va])[:,1]
        oof.loc[va,'baseline']=basep
        folds.append({'fold':fold,'branch':'baseline',**E.probability_metrics(y[va],basep,cfg['threshold'])})
        for branch,(spec,inner_ap,seconds) in choices.items():
            model=E.build_pipeline(spec).fit(X[tr],y[tr]);p=model.predict_proba(X[va])[:,1]
            oof.loc[va,branch]=p
            folds.append({'fold':fold,'branch':branch,**E.probability_metrics(y[va],p,cfg['threshold'])})
            fitted_specs.append({'fold':fold,'branch':branch,'inner_ap':inner_ap,'spec':spec})
            pi=permutation_importance(model,X[va],y[va],scoring='average_precision',n_repeats=5,random_state=42,n_jobs=1)
            for j,feature in enumerate(names):importance.append({'fold':fold,'branch':branch,'feature':feature,'importance':pi.importances_mean[j]})
        print(f'Outer fold {fold+1}/5: '+', '.join(f'{b}={r[0]["family"]}' for b,r in choices.items()),flush=True)
    assert len(oof)==len(X) and oof.id.is_unique and not oof.isna().any().any() and (oof.fold>=0).all()
    foldtable=pd.DataFrame(folds);save(foldtable,'fold_metrics');save(oof,'oof_predictions')
    imp=pd.DataFrame(importance);save(imp,'permutation_importance')
    grouped=foldtable.groupby('branch')[['ap','roc_auc','f1','recall','precision','brier','log_loss']].agg(['mean','std'])
    grouped.columns=['_'.join(c) for c in grouped.columns];save(grouped.reset_index(),'cv_summary')
    ap=foldtable.pivot(index='fold',columns='branch',values='ap');diff=(ap.automl-ap.manual).to_numpy()
    interval=E.bootstrap_mean_interval(diff)
    compare={'difference_automl_minus_manual':diff.tolist(),'mean_difference':float(diff.mean()),
             'bootstrap_interval_95':interval.tolist(),'epsilon':cfg['epsilon'],
             'interval_limitation':'Five dependent folds; descriptive fold bootstrap, not independent-sample CI or equivalence proof.'}
    dump(compare,R/'paired_comparison.json')
    # Three final inner searches; first one is the predeclared operational search.
    final_models={};final_specs={};seed_rows=[];seed_predictions={'manual':[],'automl':[]}
    engineering=[]
    for seed in cfg['stability_seeds']:
        inner=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))
        choices=search_pair(X,y,inner,seed,f'final_seed_{seed}')
        for branch,(spec,score,seconds) in choices.items():
            start=time.perf_counter();model=E.build_pipeline(spec).fit(X,y);fit_seconds=time.perf_counter()-start
            seed_predictions[branch].append(model.predict_proba(X)[:,1])
            seed_rows.append({'seed':seed,'branch':branch,'family':spec['family'],'inner_ap':score,'params':json.dumps(spec['params'],default=lambda x:x.item())})
            if seed==42:
                final_models[branch]=model;final_specs[branch]=spec
                joblib.dump(model,R/'models'/f'{branch}.joblib',compress=3)
                engineering.append({'branch':branch,'search_seconds':seconds,'fit_seconds':fit_seconds,
                                    'size_bytes':E.serialized_size_bytes(model),
                                    **{f'single_{key}':v for key,v in E.prediction_latency_ms(model,X[:1]).items()},
                                    **{f'batch256_{key}':v for key,v in E.prediction_latency_ms(model,X[:256]).items()}})
        print(f'Final inner search seed {seed} complete',flush=True)
    save(pd.DataFrame(seed_rows),'search_stability')
    disagreements={branch:E.pairwise_prediction_disagreement(p) for branch,p in seed_predictions.items()}
    dump({'on':'full train; descriptive fitted-model comparison, not a generalization estimate',
          'pairwise_disagreement':disagreements},R/'seed_disagreement.json')
    eng=pd.DataFrame(engineering);save(eng,'engineering')
    delta=compare['mean_difference'];eps=cfg['epsilon']
    if delta>eps:selected='automl';reason='Mean paired AP advantage exceeds predeclared epsilon'
    elif delta < -eps:selected='manual';reason='Mean paired AP advantage exceeds predeclared epsilon'
    else:
        selected=eng.sort_values(['size_bytes','single_p95_ms','branch']).iloc[0].branch
        reason='Mean AP difference within decision tie region; choose smaller serialized pipeline. This is not proof of equivalence.'
    decision={'selected_branch':selected,'spec':final_specs[selected],'reason':reason,'epsilon':eps,
              'cv_evidence':compare,'test_metrics_present':False,'config_sha256':E.sha256_file(ROOT/'configs/experiment.json'),
              'limitations':['single small historical Digits collection','no writer IDs; writer-independent performance unverified',
                             'fixed threshold 0.5; AP advantage does not guarantee recall advantage','limited 12-trial search']}
    dump(decision,R/'decision_record.json');decision_hash=E.sha256_file(R/'decision_record.json')
    (R/'decision.sha256').write_text(decision_hash+'\n')
    print('Decision frozen:',selected,decision_hash,flush=True)
    # Test opened once, only the chosen pipeline. Never use the other finalist on test to choose again.
    xt,yt=Xall[test_ids],yall[test_ids]
    p=final_models[selected].predict_proba(xt)[:,1]
    test={'branch':selected,'n':len(yt),'positive_n':int(yt.sum()),'threshold':cfg['threshold'],
          **E.probability_metrics(yt,p,cfg['threshold']),
          'confusion_matrix':confusion_matrix(yt,p>=cfg['threshold']).tolist(), 'decision_sha256':decision_hash}
    dump(test,R/'test_metrics.json');save(pd.DataFrame({'id':test_ids,'y':yt,'probability':p}),'test_predictions')
    assert E.sha256_file(R/'decision_record.json')==decision_hash
    for branch,m in final_models.items():
        np.testing.assert_allclose(joblib.load(R/'models'/f'{branch}.joblib').predict_proba(X[:5]),m.predict_proba(X[:5]))
    save(pd.concat(histories,ignore_index=True),'candidate_history');save(pd.DataFrame(budgets),'budget_audit')
    save(pd.DataFrame(fit_audits),'flaml_fit_audit');dump(split_contract,R/'cv_splits.json');dump(fitted_specs,R/'outer_winners.json')
    dump({'python':platform.python_version(),'platform':platform.platform(),'sklearn':sklearn.__version__,
          'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'flaml':flaml.__version__,
          'threads':1,'warnings_policy':'Warnings are treated as errors by the validation runner.'},R/'environment.json')
    # Behavior diagnostics, entirely OOF.
    om=oof.manual>=.5;oa=oof.automl>=.5;correct_m=om==y;correct_a=oa==y
    categories=np.select([correct_m&correct_a,~correct_m&~correct_a,correct_m&~correct_a],
                         ['both_correct','both_wrong','only_manual_correct'],default='only_automl_correct')
    behavior=oof.copy();behavior['category']=categories;behavior['probability_gap']=(oof.manual-oof.automl).abs()
    save(behavior,'oof_behavior');save(behavior.sort_values(['probability_gap','id'],ascending=[False,True]).head(5),'largest_disagreements')
    save(pd.Series(categories).value_counts().rename_axis('category').reset_index(name='n'),'behavior_summary')
    plt.figure(figsize=(7,4));plt.boxplot([ap.manual,ap.automl,ap.baseline],tick_labels=['Manual','AutoML','Baseline'])
    plt.xlabel('Search branch');plt.ylabel('Outer validation AP');plt.title('Nested CV distribution, identical folds');figure('01_ap_distribution')
    plt.figure(figsize=(7,4))
    for i,row in ap.iterrows():plt.plot([0,1],[row.manual,row.automl],'o-',label=f'Fold {i+1}')
    plt.xticks([0,1],['Manual','AutoML']);plt.xlabel('Search branch');plt.ylabel('Outer validation AP');plt.title('Paired fold comparison');plt.legend();figure('02_paired_ap')
    plt.figure(figsize=(7,4));plt.plot([0,1],[0,1],'--',label='Ideal')
    for branch in ['manual','automl']:
        fraction,prob=calibration_curve(y,oof[branch],n_bins=8,strategy='quantile');plt.plot(prob,fraction,'o-',label=branch)
    plt.xlabel('Mean OOF probability');plt.ylabel('Observed positive fraction');plt.title('OOF calibration, quantile bins');plt.legend();figure('03_calibration')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,branch in zip(axes,['manual','automl']):
        matrix=confusion_matrix(y,oof[branch]>=.5);ax.imshow(matrix,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Prediction (1=digit 5)',ylabel='True label',title=f'{branch}: OOF, threshold=0.5')
    figure('04_confusion_matrices')
    means=imp.groupby(['branch','feature']).importance.mean().unstack('branch')
    top=set(means.manual.nlargest(10).index)|set(means.automl.nlargest(10).index)
    means.loc[sorted(top)].plot.barh(figsize=(9,7));plt.xlabel('Held-out AP decrease after permutation');plt.ylabel('Pixel feature');plt.title('Outer-fold permutation importance, top-10 union');figure('05_permutation_importance')
    plt.figure(figsize=(7,4));plt.scatter(oof.manual,oof.automl,c=y,cmap='coolwarm',s=12)
    plt.plot([0,1],[0,1],'--',label='Agreement');plt.xlabel('Manual OOF probability');plt.ylabel('AutoML OOF probability');plt.title('Largest differences are inspected by object ID');plt.legend();plt.colorbar(label='True digit 5');figure('06_probability_disagreement')
    fig,axes=plt.subplots(1,5,figsize=(12,3))
    for ax,row in zip(axes,behavior.nlargest(5,'probability_gap').itertuples()):
        ax.imshow(Xall[row.id].reshape(8,8),cmap='gray_r',vmin=0,vmax=16)
        ax.set(title=f'ID {row.id}, y={row.y}\nM={row.manual:.3f}, A={row.automl:.3f}',xlabel='Pixel column',ylabel='Pixel row')
    figure('07_disagreement_objects')
    # Handoff contains a raw-input sklearn pipeline, not a FLAML search object.
    for source,name in [(R/'models'/f'{selected}.joblib','pipeline.joblib'),(R/'decision_record.json','decision_record.json'),
                        (R/'decision.sha256','decision.sha256'),(ROOT/'configs/experiment.json','config.json'),
                        (R/'split_ids.csv','split_ids.csv'),(R/'fold_metrics.csv','cv_metrics.csv'),(R/'test_metrics.json','test_metrics.json')]:shutil.copyfile(source,H/name)
    dump({'features':[{'name':n,'dtype':'float64','min':0,'max':16} for n in names],
          'order_required':True,'shape':[None,64],'positive_class':'digit 5',
          'missing':'rejected at serving boundary; training pipeline includes median imputation'},H/'data_schema.json')
    example={'input':X[:3].tolist(),'expected_probability':final_models[selected].predict_proba(X[:3])[:,1].tolist(),'threshold':.5}
    dump(example,H/'inference_example.json')
    shutil.copyfile(ROOT/'inference.py',H/'inference.py');shutil.copyfile(ROOT/'requirements.txt',H/'requirements.txt')
    card=f'''# Model card — Digits, digit 5 vs rest

Навчальний прототип маршрутизації зображень 8×8 для оператора OCR. Один об'єкт — 64 пікселі 0–16.
Позитивний клас: цифра 5; походження: sklearn load_digits / UCI Optical Recognition of Handwritten Digits.
Джерело: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
Ліцензія першоджерела UCI: CC BY 4.0; див. https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Вибрана гілка: {selected}; сімейство: {final_specs[selected]['family']}.
Причина: {reason}
Рішення SHA-256: {decision_hash}. Test AP={test['ap']:.6f}, F1={test['f1']:.6f}, recall={test['recall']:.6f}.
Поріг 0.5 зафіксований до оцінювання. Pipeline навчається на train, не перевчався на test.
CV, параметри, схема, середовище і приклад інференсу — у цьому пакеті та ../results/.

Обмеження: мала історична колекція, немає ID авторів для group split; нові почерки та сканери не перевірені.
Це не розпізнавач усіх цифр. Не використовувати для автоматичного ухвалення фінансових чи інших значущих рішень.
За невідомої схеми, виходу пікселів за діапазон або дрейфу — зупинити інференс і переглянути модель з оператором.
До ЛР7: перевіряти схему, версії, хеш рішення, контрольний прогноз; перед реальним застосуванням потрібен незалежний набір авторів.
'''
    (H/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    dump({'selected':selected,'decision_sha256':decision_hash,'cv':grouped.reset_index().to_dict(orient='records'),
          'test':test,'comparison':compare,'top10_jaccard':E_top_jaccard(means)},R/'summary.json')
    print(json.dumps({'comparison':compare,'selected':selected,'test':test},indent=2))


def E_top_jaccard(means):
    a=set(means.manual.nlargest(10).index);b=set(means.automl.nlargest(10).index)
    return len(a&b)/len(a|b)


if __name__=='__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with threadpool_limits(1):run()
