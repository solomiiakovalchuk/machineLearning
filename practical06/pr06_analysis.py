"""Audit fixed LR5 artifacts and write every PR6 result with one command."""
from __future__ import annotations
import json, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import audit as A

ROOT=Path(__file__).resolve().parent
LR5=ROOT.parent/'lab05'
sys.path.insert(0,str(LR5))
from src import unsupervised_pipeline as U


def run():
    cfg=json.loads((ROOT/'configs/audit.json').read_text(encoding='utf-8'))
    out=ROOT/'results'; figs=ROOT/'figures'
    out.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    def save(df,name): df.to_csv(out/f'{name}.csv',index=False)
    def fig(name):
        plt.tight_layout(); plt.savefig(figs/f'{name}.png',dpi=150); plt.close()
    paths=[LR5/'artifacts/lr05_bundle.joblib',LR5/'configs/experiment_config.json',
           LR5/'results/split_ids.csv',LR5/'results/selected_k.json']
    b=U.load_bundle(paths[0]); d=U.load_dataset(b['config'])
    split=pd.read_csv(paths[2])
    assert set(split.query("part == 'train'").id)==set(d.id_train)
    assert set(split.query("part == 'test'").id)==set(d.id_test)
    raw=d.X_train
    xs=b['preprocessor'].transform(raw); z=b['pca'].transform(xs)
    km=b['kmeans_pca']; ref=km.predict(z); k=km.n_clusters
    batch=U.inject_controlled_anomalies(d.X_test,d.id_test,b['config'])
    te=b['preprocessor'].transform(batch.X); zte=b['pca'].transform(te)
    scores=U.predict_new(b,batch.X).rename(columns={'cluster_pca':'cluster'})
    for det in U.DETECTORS:
        reference=np.sort(b['detectors'].train_scores[det])
        scores[f'pct_{det}']=U.empirical_percentile(reference,scores[f'score_{det}'])
    report=A.validate_artifacts(paths,d.id_train,d.id_test,xs,ref,scores,k)
    report.update(q=int(b['pca'].n_components_),python=platform.python_version(),
                  input_commit='1bd0b4a',reference_seed=42)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    diag=A.compute_boundary_diagnostics(z,km.cluster_centers_)
    diag.insert(0,'id',d.id_train)
    profiles=A.build_cluster_profiles(raw,ref); save(profiles,'cluster_profiles')
    selected=[]
    for c,g in profiles.groupby('cluster'):
        for direction,asc in [('positive',False),('negative',True)]:
            h=g[g.profile>0] if direction=='positive' else g[g.profile<0]
            h=h.sort_values('profile',ascending=asc).head(5).copy();h['direction']=direction;selected.append(h)
    save(pd.concat(selected),'profile_top_signed')
    reps=A.find_cluster_representatives(d.id_train,diag.drop(columns='id'));save(reps,'representatives')
    full_runs=[]; test_runs=[]; seed_runs=[]
    for seed in b['config']['kmeans']['seeds']:
        seed_runs.append(KMeans(n_clusters=k,n_init=10,random_state=seed).fit(z).predict(z))
    seed_stability,_,seed_ari=A.compare_cluster_runs(ref,seed_runs)
    rng=np.random.default_rng(cfg['seed'])
    for _ in range(cfg['repeats']):
        idx=rng.choice(len(z),round(len(z)*cfg['subsample_fraction']),replace=False)
        model=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z[idx])
        candidate=model.predict(z); aligned=A.align(ref,candidate)
        mapping={c:aligned[np.flatnonzero(candidate==c)[0]] for c in np.unique(candidate)}
        full_runs.append(candidate);test_runs.append(np.array([mapping[c] for c in model.predict(zte)]))
    stability,aligned,ari=A.compare_cluster_runs(ref,full_runs)
    diag['stability']=stability;save(diag,'object_stability')
    save(diag.sort_values(['stability','margin']).head(10),'least_stable_10')
    base_top=A.top_features(profiles); stability_rows=[]; profile_runs=[]
    for run,labels in enumerate(aligned):
        p=A.build_cluster_profiles(raw,labels);p['run']=run;profile_runs.append(p)
        tops=A.top_features(p)
        for c in range(k):
            a=profiles.query('cluster == @c').set_index('feature').profile
            v=p.query('cluster == @c').set_index('feature').profile
            stability_rows.append({'run':run,'cluster':c,'jaccard':A.jaccard(base_top[c],tops[c]),
                                   'max_abs_profile_change':(a-v).abs().max(),
                                   'top_features':'|'.join(sorted(tops[c]))})
    pst=pd.DataFrame(stability_rows);save(pst,'profile_stability');save(pd.concat(profile_runs),'profile_runs')
    centers=cdist(km.cluster_centers_,km.cluster_centers_);np.fill_diagonal(centers,np.inf)
    summary=[]
    for c in range(k):
        g=diag.query('cluster == @c'); pr=pst.query('cluster == @c')
        summary.append({'cluster':c,'n':len(g),'fraction':len(g)/len(z),'mean_distance':g.distance.mean(),
                        'rms_distance':np.sqrt(np.mean(g.distance**2)),'neighbor_cluster':int(centers[c].argmin()),
                        'center_distance':centers[c].min(),'mean_stability':g.stability.mean(),
                        'unstable_n':int((g.stability<cfg['stability_threshold']).sum()),
                        'boundary_n':int((g.margin<cfg['boundary_threshold']).sum()),
                        'jaccard_mean':pr.jaccard.mean(),'max_profile_change':pr.max_abs_profile_change.max()})
    save(pd.DataFrame(summary),'cluster_summary')
    # PCA weights are for standardized log1p features, not the raw-scale profile above.
    loads=[];extrema=[]
    for pc in range(3):
        weights=b['pca'].components_[pc]
        for f,w in zip(d.feature_names,weights):loads.append({'component':pc+1,'feature':f,'weight':w})
        for side,idx in [('min',int(z[:,pc].argmin())),('max',int(z[:,pc].argmax()))]:
            for j,f in enumerate(d.feature_names):
                extrema.append({'component':pc+1,'side':side,'id':int(d.id_train[idx]),'coordinate':z[idx,pc],
                                'feature':f,'raw_value':raw.iloc[idx,j],'scaled_log_value':xs[idx,j],
                                'contribution':weights[j]*xs[idx,j]})
    save(pd.DataFrame(loads),'pca_loadings');save(pd.DataFrame(extrema),'pca_extrema')
    # Train-only alignment; test labels and injection indicators are unavailable to policy.
    td=A.compute_boundary_diagnostics(zte,km.cluster_centers_)
    scores['margin']=td.margin.to_numpy()
    scores['stability']=np.mean(np.array(test_runs)==scores.cluster.to_numpy(),axis=0)
    residual=(te-b['pca'].inverse_transform(zte))**2
    scores['largest_residual_feature']=np.array(d.feature_names)[residual.argmax(axis=1)]
    scores['largest_residual']=residual.max(axis=1)
    policy=A.apply_action_policy(scores,cfg['policy']); scores=pd.concat([scores,policy],axis=1)
    save(scores,'scores_actions')
    # Only now attach control labels to audit consequences, never to choose thresholds.
    reviewed=scores.action!='no_action';truth=batch.is_anomaly.astype(bool)
    audit=scores[['id','action','conflict']].copy();audit['controlled_anomaly']=truth;audit['kind']=batch.kind
    audit['diagnosis_audit_only']=d.y_test.to_numpy();save(audit,'action_label_audit')
    actions=audit.groupby('action').agg(n=('id','size'),tp=('controlled_anomaly','sum'),conflicts=('conflict','sum')).reset_index()
    actions['fp']=actions.n-actions.tp;actions['fraction']=actions.n/len(scores);save(actions,'action_summary')
    tp=int((reviewed&truth).sum());fp=int((reviewed&~truth).sum());fn=int((~reviewed&truth).sum())
    metrics={'train_n':len(raw),'test_n':len(scores),'q':int(b['pca'].n_components_),'k':k,
             'seed_ari':seed_ari,'seed_stability_min':float(seed_stability.min()),'subsample_ari':ari,
             'stability_mean':float(stability.mean()),'stability_min':float(stability.min()),
             'unstable_n':int((stability<cfg['stability_threshold']).sum()),'review_n':int(reviewed.sum()),
             'review_fraction':float(reviewed.mean()),'tp':tp,'fp':fp,'fn':fn,'tn':int((~reviewed&~truth).sum()),
             'conflicts':int(scores.conflict.sum()),'precision':tp/(tp+fp) if tp+fp else 0,
             'recall':tp/(tp+fn),'operator_minutes':int(reviewed.sum())*3,
             'quality_scenario_cost_units':fp*3+fn*20,'rejection_scenario_cost_units':fp*100+fn*20}
    # Distinct cards across the three categories; sorting within ties is explicit.
    cards=[];used=set()
    ranked=scores.sort_values(['pct_combined','score_rec','id'],ascending=[False,False,True])
    disagree=scores.assign(gap=(scores.pct_ifpca-scores.pct_rec).abs()).sort_values(['gap','id'],ascending=[False,True])
    falsepos=scores.loc[reviewed&~truth].sort_values(['pct_combined','id'],ascending=[False,True])
    for category,pool in [('top_combined',ranked),('detector_disagreement',disagree),('false_positive',falsepos)]:
        chosen=pool.loc[~pool.id.isin(used)].head(5)
        if len(chosen)!=5:raise ValueError('not enough distinct explanation cards')
        for _,row in chosen.iterrows():cards.append(A.explain_anomaly(row,category));used.add(row.id)
    save(pd.DataFrame(cards),'anomaly_cards')
    (out/'anomaly_cards.json').write_text(json.dumps(cards,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    # Figures, all with titles, axis labels and explicit scales.
    heat=profiles.pivot(index='cluster',columns='feature',values='profile')
    cols=heat.abs().max().nlargest(15).index
    plt.figure(figsize=(12,3));plt.imshow(heat[cols],aspect='auto',cmap='coolwarm',vmin=-1.5,vmax=1.5)
    plt.xticks(range(len(cols)),cols,rotation=65,ha='right');plt.yticks(range(k),range(k))
    plt.colorbar(label='Raw feature deviation / train SD');plt.xlabel('Original feature');plt.ylabel('Cluster');plt.title('Cluster profiles in original features');fig('01_cluster_profiles')
    plt.figure(figsize=(8,4));plt.scatter(diag.margin,diag.distance,c=diag.stability,cmap='viridis',vmin=0,vmax=1)
    plt.colorbar(label='Assignment stability');plt.axvline(cfg['boundary_threshold'],ls='--',label='Boundary threshold')
    plt.xlabel('Normalized boundary margin');plt.ylabel('Distance in PCA(6)');plt.title('Distance and boundary measure different properties');plt.legend();fig('02_distance_boundary')
    plt.figure(figsize=(8,4));plt.hist(stability,bins=np.linspace(0,1,21),label='20 subsamples');plt.axvline(.8,color='red',ls='--',label='Control threshold')
    plt.xlabel('Assignment stability');plt.ylabel('Train objects');plt.title('Object stability, fixed preprocessing and K');plt.legend();fig('03_stability')
    plt.figure(figsize=(8,4))
    for c,g in pst.groupby('cluster'):plt.plot(g.run,g.jaccard,'o-',label=f'Cluster {c}')
    plt.xlabel('Subsample repeat');plt.ylabel('Jaccard top-5');plt.title('Stability of explanations');plt.legend();fig('04_profile_stability')
    figobj,axes=plt.subplots(1,3,figsize=(15,5))
    for pc,ax in enumerate(axes):
        w=pd.Series(b['pca'].components_[pc],index=d.feature_names);chosen=pd.concat([w.nlargest(5),w.nsmallest(5)]).sort_values()
        ax.barh(chosen.index,chosen,color=['steelblue' if v>0 else 'coral' for v in chosen]);ax.set(xlabel='Weight on scaled log1p feature',ylabel='Feature',title=f'PC{pc+1} signed loadings')
    fig('05_pca_signed_loadings')
    plt.figure(figsize=(7,5));plt.scatter(scores.pct_ifpca,scores.pct_rec,c=truth,cmap='coolwarm',label='Test objects')
    plt.axhline(.95,ls='--',label='Reconstruction reference');plt.xlabel('IF-PCA train percentile');plt.ylabel('Reconstruction train percentile');plt.title('Complementary anomaly signals');plt.legend();plt.colorbar(label='Controlled injection (audit only)');fig('06_anomaly_signals')
    plt.figure(figsize=(8,4));plt.bar(actions.action,actions.fp,label='Not injected');plt.bar(actions.action,actions.tp,bottom=actions.fp,label='Injected')
    plt.xticks(rotation=20);plt.xlabel('First matching action');plt.ylabel('Test objects');plt.title('Policy workload and audited composition');plt.legend();fig('07_actions')
    print(json.dumps(metrics,indent=2));return metrics


if __name__=='__main__':
    with threadpool_limits(1):run()
