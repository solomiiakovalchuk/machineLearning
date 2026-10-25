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
    E.check_test_flags(cfg)
    Xall,yall,names=E.load_data()
    train_ids,test_ids=train_test_split(np.arange(len(yall)),test_size=cfg['test_size'],stratify=yall,random_state=42)
    assert not set(train_ids)&set(test_ids)
    # No EDA, model fitting or prediction accesses Xall[test_ids] before decision freeze below.
    X,y=Xall[train_ids],yall[train_ids]
    provenance={'source':'UCI Breast Cancer Wisconsin (Diagnostic)','url':'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
                'authors':'W. Wolberg, W. Street, O. Mangasarian','license':'CC BY 4.0','access':'sklearn.datasets.load_breast_cancer',
                'sklearn_version':sklearn.__version__,'n':int(len(yall)),'n_features':int(Xall.shape[1]),
                'class_counts':{'malignant':int(yall.sum()),'benign':int(len(yall)-yall.sum())},
                'raw_x_target_sha256':E.raw_data_sha256(),'raw_hash_definition':'sha256(ascontiguousarray(data).tobytes() + ascontiguousarray(target).tobytes()) of load_breast_cancer',
                'duplicate_rows':int(len(Xall)-len(np.unique(Xall,axis=0))),'object_id':'row index in load_breast_cancer',
                'reuse_note':('The same public dataset was used in earlier course labs (lab01, lab05, practical05/06), so the lab06 test part is untouched '
                              'only within this experiment. The lab06 search space, metric, epsilon and tie-break were fixed in the config before this run '
                              'and were not tuned on results of earlier labs.')}
    lab05_split=ROOT.parent/'lab05/results/split_ids.csv'
    if lab05_split.exists():
        prev=pd.read_csv(lab05_split)
        if {'id','part'}<=set(prev.columns):
            provenance['lab05_split_overlap']={'source':'lab05/results/split_ids.csv (ids only, no features or models read)',
                'lab06_test_ids_in_lab05_train':int(np.isin(test_ids,prev[prev.part.eq('train')].id).sum()),
                'lab06_test_ids_in_lab05_test':int(np.isin(test_ids,prev[prev.part.eq('test')].id).sum()),'lab06_test_n':int(len(test_ids))}
    dump(provenance,R/'data_provenance.json')
    split=pd.DataFrame({'id':np.r_[train_ids,test_ids],'part':['train']*len(train_ids)+['test']*len(test_ids)})
    save(split,'split_ids')
    dump({'train_n':len(X),'features':X.shape[1],'positive_n':int(y.sum()),'missing_n':int(np.isnan(X).sum()),
          'constant_features':[names[i] for i in np.flatnonzero(X.std(axis=0)==0)],
          'min':float(X.min()),'max':float(X.max()),'positive_class':'malignant (y=1)'},R/'train_eda.json')
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
    if delta>eps:selected='automl';reason=f'AutoML mean paired outer-CV AP advantage {delta:+.6f} exceeds predeclared epsilon {eps}'
    elif delta < -eps:selected='manual';reason=f'Manual mean paired outer-CV AP advantage {-delta:.6f} exceeds predeclared epsilon {eps}'
    else:
        selected=eng.sort_values(['size_bytes','single_p95_ms','branch']).iloc[0].branch
        reason=(f'Mean paired AP difference {delta:+.6f} is within epsilon {eps}; tie-break applied: '
                f'{cfg["tie_break"]}. This is not proof of equivalence.')
    sel_ap=foldtable[foldtable.branch.eq(selected)].ap
    sel_spec=final_specs[selected]
    decision={'selected_branch':selected,'candidate_id':sel_spec['candidate_id'],'family':sel_spec['family'],'params':sel_spec['params'],
              'spec':sel_spec,'cv_ap_mean':float(sel_ap.mean()),'cv_ap_std':float(sel_ap.std(ddof=1)),
              'paired_difference':compare,'cv_evidence':compare,'epsilon':eps,'tie_break':cfg['tie_break'],'reason':reason,
              'test_metrics_present':False,'config_sha256':E.sha256_file(ROOT/'configs/experiment.json'),
              'limitations':['small single-source dataset of 569 FNA images; performance on other laboratories, scanners or populations is unverified',
                             'dataset was already seen in earlier course labs; the lab06 test part is untouched only within this experiment',
                             'random stratified split without patient or time grouping information; dependence between records cannot be excluded',
                             'fixed threshold 0.5; AP advantage does not guarantee recall advantage',
                             'limited 12-trial search budget per branch and per search',
                             'educational model; forbidden for diagnosis or clinical decisions']}
    dump(decision,R/'decision_record.json');decision_hash=E.sha256_file(R/'decision_record.json')
    (R/'decision.sha256').write_text(decision_hash+'\n')
    print('Decision frozen:',selected,decision_hash,flush=True)
    # Test opened once for the chosen pipeline; the decision file is already hashed.
    xt,yt=Xall[test_ids],yall[test_ids]
    p=final_models[selected].predict_proba(xt)[:,1]
    test={'branch':selected,'n':len(yt),'positive_n':int(yt.sum()),'threshold':cfg['threshold'],
          **E.probability_metrics(yt,p,cfg['threshold']),
          'confusion_matrix':confusion_matrix(yt,p>=cfg['threshold']).tolist(), 'decision_sha256':decision_hash}
    dump(test,R/'test_metrics.json');save(pd.DataFrame({'id':test_ids,'y':yt,'probability':p}),'test_predictions')
    assert E.sha256_file(R/'decision_record.json')==decision_hash
    # Audit only: the other finalist on test after the freeze. Never used for selection; the decision file is final.
    other='manual' if selected=='automl' else 'automl'
    po=final_models[other].predict_proba(xt)[:,1]
    dump({'AUDIT_ONLY':True,'note':'Evaluated after the decision freeze. Not used for selection, tuning or reporting of the selected model.',
          'branch':other,'candidate_id':final_specs[other]['candidate_id'],'family':final_specs[other]['family'],
          'n':len(yt),'positive_n':int(yt.sum()),'threshold':cfg['threshold'],**E.probability_metrics(yt,po,cfg['threshold']),
          'confusion_matrix':confusion_matrix(yt,po>=cfg['threshold']).tolist(),'decision_sha256':decision_hash},R/'test_audit_other_finalist.json')
    assert E.sha256_file(R/'decision_record.json')==decision_hash
    for branch,m in final_models.items():
        np.testing.assert_allclose(joblib.load(R/'models'/f'{branch}.joblib').predict_proba(X[:5]),m.predict_proba(X[:5]))
    history_all=pd.concat(histories,ignore_index=True)
    save(history_all,'candidate_history');save(pd.DataFrame(budgets),'budget_audit')
    save(pd.DataFrame(fit_audits),'flaml_fit_audit');dump(split_contract,R/'cv_splits.json');dump(fitted_specs,R/'outer_winners.json')
    dump({'python':platform.python_version(),'platform':platform.platform(),'sklearn':sklearn.__version__,
          'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'flaml':flaml.__version__,
          'threads':1,'warnings_policy':'Warnings are treated as errors by the validation runner.'},R/'environment.json')
    # Behavior diagnostics, entirely OOF.
    om=oof.manual>=.5;oa=oof.automl>=.5;correct_m=om==y;correct_a=oa==y
    categories=np.select([correct_m&correct_a,~correct_m&~correct_a,correct_m&~correct_a],
                         ['both_correct','both_wrong','only_manual_correct'],default='only_automl_correct')
    behavior=oof.copy();behavior['category']=categories;behavior['probability_gap']=(oof.manual-oof.automl).abs()
    save(behavior,'oof_behavior');top5=behavior.sort_values(['probability_gap','id'],ascending=[False,True]).head(5)
    save(top5,'largest_disagreements')
    save(pd.Series(categories).value_counts().rename_axis('category').reset_index(name='n'),'behavior_summary')
    lo=max(0.,min(ap.manual.min(),ap.automl.min())-.02);hi=min(max(ap.manual.max(),ap.automl.max()),1.)+.01
    fig,axes=plt.subplots(1,2,figsize=(9,4),gridspec_kw={'width_ratios':[2,1]})
    axes[0].boxplot([ap.manual,ap.automl],tick_labels=['Manual','AutoML'])
    axes[0].set(xlabel='Search branch',ylabel='Outer validation AP',title='Nested CV: model comparison',ylim=(lo,hi))
    axes[1].boxplot([ap.baseline],tick_labels=['Prior baseline'])
    axes[1].set(xlabel='Prior baseline',ylabel='Outer validation AP',title='Baseline on separate scale',ylim=(0,max(.1,float(ap.baseline.max())*1.3)))
    figure('01_ap_distribution')
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
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Prediction (1=malignant)',ylabel='True label (1=malignant)',title=f'{branch}: OOF, threshold=0.5')
    figure('04_confusion_matrices')
    means=imp.groupby(['branch','feature']).importance.mean().unstack('branch')
    top=set(means.manual.nlargest(10).index)|set(means.automl.nlargest(10).index)
    means.loc[sorted(top)].plot.barh(figsize=(9,7));plt.xlabel('Held-out AP decrease after permutation');plt.ylabel('Feature');plt.title('Outer-fold permutation importance, top-10 union');plt.legend(title='Branch');figure('05_permutation_importance')
    plt.figure(figsize=(7,4));plt.scatter(oof.manual,oof.automl,c=y,cmap='coolwarm',s=12)
    plt.plot([0,1],[0,1],'--',label='Agreement');plt.xlabel('Manual OOF probability');plt.ylabel('AutoML OOF probability');plt.title('Largest differences are inspected by object ID');plt.legend();plt.colorbar(label='True label (1=malignant)');figure('06_probability_disagreement')
    # Training-only standardisation (mean/std of the training part) of the five largest-disagreement OOF objects.
    mu,sd=X.mean(axis=0),X.std(axis=0);sd=np.where(sd==0,1.,sd)
    zs=(Xall[top5.id.to_numpy()]-mu)/sd
    cols=np.argsort(-np.abs(zs).max(axis=0))[:10]
    fig,ax=plt.subplots(figsize=(11,4.5));vmax=max(1.,float(np.abs(zs[:,cols]).max()))
    im=ax.imshow(zs[:,cols],cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto')
    for i in range(zs.shape[0]):
        for j,c in enumerate(cols):ax.text(j,i,f'{zs[i,c]:.1f}',ha='center',va='center',fontsize=8)
    ax.set(xticks=range(len(cols)),yticks=range(len(zs)),title='Largest-disagreement OOF objects: train z-scores, 10 most extreme features',
           xlabel='Feature',ylabel='Object')
    ax.set_xticklabels([names[c] for c in cols],rotation=45,ha='right')
    ax.set_yticklabels([f'ID {r.id}, y={r.y}\nM={r.manual:.3f}, A={r.automl:.3f}' for r in top5.itertuples()])
    fig.colorbar(im,ax=ax,label='z-score (training mean/std)');figure('07_disagreement_objects')
    final_hist=history_all[history_all.stage.eq('final_seed_42')]
    palette=dict(zip(sorted(final_hist.family.unique()),plt.cm.tab10.colors))
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,branch in zip(axes,['manual','automl']):
        h=final_hist[final_hist.branch.eq(branch)].sort_values('trial')
        ax.plot(h.trial,h.inner_ap,color='gray',lw=.6,zorder=0)
        for family,g in h.groupby('family'):ax.scatter(g.trial,g.inner_ap,color=palette[family],label=family,s=40)
        ax.set(xlabel='Trial order',ylabel='Inner CV AP',title=f'{branch}: seed-42 final search');ax.legend(title='Family',fontsize=8)
    figure('08_candidate_history')
    # Handoff contains a raw-input sklearn pipeline, not a FLAML search object.
    for source,name in [(R/'models'/f'{selected}.joblib','pipeline.joblib'),(R/'decision_record.json','decision_record.json'),
                        (R/'decision.sha256','decision.sha256'),(ROOT/'configs/experiment.json','config.json'),
                        (R/'split_ids.csv','split_ids.csv'),(R/'fold_metrics.csv','cv_metrics.csv'),(R/'test_metrics.json','test_metrics.json'),
                        (R/'data_provenance.json','data_provenance.json'),(R/'environment.json','environment.json')]:shutil.copyfile(source,H/name)
    dump({'features':[{'name':n,'dtype':'float64','required':True,'min':float(X[:,i].min()),'max':float(X[:,i].max())} for i,n in enumerate(names)],
          'order_required':True,'shape':[None,len(names)],'positive_class':'malignant (y=1)','threshold':cfg['threshold'],
          'constraints':'finite, non-negative, ordered as listed',
          'missing':'rejected at serving boundary; training pipeline includes median imputation'},H/'data_schema.json')
    example={'input':X[:3].tolist(),'expected_probability':final_models[selected].predict_proba(X[:3])[:,1].tolist(),'threshold':.5}
    dump(example,H/'inference_example.json')
    shutil.copyfile(ROOT/'inference.py',H/'inference.py');shutil.copyfile(ROOT/'requirements.txt',H/'requirements.txt')
    cm=test['confusion_matrix'];params_text=json.dumps(sel_spec['params'],default=lambda x:x.item())
    card=f'''# Model card — Breast Cancer Wisconsin (Diagnostic), malignant vs benign

## Призначення
Навчальний прототип ранжування записів FNA-аналізу за ймовірністю класу malignant для подальшого розгляду експертом.
Це НЕ діагностичний інструмент і не підтримка клінічних рішень.

## Дані
UCI Breast Cancer Wisconsin (Diagnostic), https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (CC BY 4.0;
W. Wolberg, W. Street, O. Mangasarian), через `sklearn.datasets.load_breast_cancer`. 569 записів, 30 числових ознак, 212 malignant (y=1) / 357 benign.
Split: train {len(train_ids)} / test {len(test_ids)} (stratified, seed 42). Той самий публічний набір використовувався в попередніх ЛР курсу,
тому test «не торкнутий» лише в межах цього експерименту. Хеш сирих даних і примітка — у data_provenance.json.

## Вибрана модель
Гілка: {selected}; кандидат: {sel_spec['candidate_id']}; сімейство: {sel_spec['family']}; параметри: {params_text}.
Причина вибору: {reason}
SHA-256 рішення: {decision_hash} (рішення записане й захешоване до відкриття test).

## Якість
Вкладена CV (5 зовнішніх фолдів, AP): {selected} {decision['cv_ap_mean']:.4f} ± {decision['cv_ap_std']:.4f};
різниця AutoML − Manual = {delta:+.4f}, 95% bootstrap [{interval[0]:+.4f}, {interval[1]:+.4f}] (5 залежних фолдів, лише описово).
Test (один раз, поріг {cfg['threshold']}): AP={test['ap']:.4f}, ROC-AUC={test['roc_auc']:.4f}, F1={test['f1']:.4f}, recall={test['recall']:.4f}, precision={test['precision']:.4f}, Brier={test['brier']:.4f}.
Матриця помилок test [[TN, FP], [FN, TP]] = {cm}.

## Поріг
Фіксований {cfg['threshold']} (y=1 — malignant), заданий до оцінювання; підбір порога чи калібрування на test не виконувалися.

## Обмеження
- мала колекція (569 зображень FNA з одного джерела); перенесення на інші лабораторії, сканери й популяції не перевірене;
- набір уже використовувався в попередніх ЛР; test «не торкнутий» лише в межах цього експерименту;
- випадковий stratified split без інформації про пацієнтів чи час; залежність між записами не виключена;
- фіксований поріг 0.5; перевага за AP не гарантує перевагу за recall;
- пошук обмежений 12 випробуваннями; 5 залежних фолдів не дають незалежного довірчого інтервалу.

## Заборонені застосування
Діагностика, скринінг, тріаж пацієнтів, будь-які клінічні чи інші значущі автоматичні рішення без експерта.

## Що перевірити в ЛР7
Схему входу (30 ознак у порядку data_schema.json, скінченні, невід'ємні), версії бібліотек (environment.json), SHA-256 рішення,
контрольний прогноз з inference_example.json, дрейф ознак відносно train min/max, та що відповіді сервісу не подаються як діагноз.
'''
    (H/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    dump({'selected':selected,'candidate_id':sel_spec['candidate_id'],'family':sel_spec['family'],'reason':reason,
          'decision_sha256':decision_hash,'cv':grouped.reset_index().to_dict(orient='records'),
          'selected_cv_ap_mean':decision['cv_ap_mean'],'selected_cv_ap_std':decision['cv_ap_std'],
          'test':test,'comparison':compare,'top10_jaccard':E_top_jaccard(means),'stability':seed_rows,
          'engineering':eng.to_dict(orient='records'),'seed_disagreement':disagreements,'test_audit_other_finalist':other},R/'summary.json')
    print(json.dumps({'comparison':compare,'selected':selected,'test':test},indent=2))


def E_top_jaccard(means):
    a=set(means.manual.nlargest(10).index);b=set(means.automl.nlargest(10).index)
    return len(a&b)/len(a|b)


if __name__=='__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with threadpool_limits(1):run()
