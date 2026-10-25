"""Controlled Manual / FLAML experiment. All preprocessing is inside fold fits."""
from __future__ import annotations
import hashlib, io, json, time
from itertools import combinations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (average_precision_score,roc_auc_score,f1_score,precision_score,
                            recall_score,balanced_accuracy_score,brier_score_loss,log_loss)
from sklearn.datasets import load_breast_cancer
from flaml import AutoML, tune
from flaml.automl.model import BaseEstimator


def to_snake(col):
    """'mean radius' -> 'radius_mean', 'radius error' -> 'radius_se', 'worst radius' -> 'radius_worst' (lab05 convention)."""
    parts=col.split()
    if parts[0] in ('mean','worst'):return '_'.join(parts[1:])+'_'+parts[0]
    if parts[-1]=='error':return '_'.join(parts[:-1])+'_se'
    return '_'.join(parts)


def raw_data_sha256():
    """SHA-256 of the raw sklearn feature matrix bytes followed by the raw target bytes."""
    raw=load_breast_cancer()
    h=hashlib.sha256();h.update(np.ascontiguousarray(raw.data).tobytes());h.update(np.ascontiguousarray(raw.target).tobytes())
    return h.hexdigest()


def load_data():
    """UCI Breast Cancer Wisconsin (Diagnostic) via sklearn. Object ID = row index. y=1 is malignant."""
    raw=load_breast_cancer()
    X=np.asarray(raw.data,dtype=float);y=(np.asarray(raw.target)==0).astype(int)
    names=[to_snake(c) for c in raw.feature_names]
    if X.shape!=(569,30) or y.shape!=(569,) or len(set(names))!=30:raise ValueError('unexpected Breast Cancer shape')
    if not np.isfinite(X).all() or (X<0).any():raise ValueError('features must be finite and non-negative')
    if len(np.unique(X,axis=0))!=len(X):raise ValueError('duplicate feature rows require group-aware splitting')
    return X,y,names


def check_test_flags(cfg):
    """Programmatic guard: the held-out test part must not be used for search or for selection."""
    for key in ['test_used_for_search','test_used_for_selection']:
        if cfg.get(key) is not False:raise ValueError(f'{key} must be exactly false')

SPACE={
    'lrl2': {'C':[0.01,0.1,1.,10.], 'class_weight':[None,'balanced']},
    'rf': {'n_estimators':[32,64,128], 'max_depth':[5,10,None], 'min_samples_leaf':[1,3,5]},
    'extra_tree': {'n_estimators':[32,64,128], 'max_depth':[5,10,None], 'min_samples_leaf':[1,3,5]},
    'histgb': {'max_iter':[50,100,150], 'learning_rate':[0.03,0.1,0.2], 'max_leaf_nodes':[7,15,31]},
    'kneighbor': {'n_neighbors':[3,5,9,15], 'weights':['uniform','distance']}}


def manual_specs():
    specs=[]
    for C in [0.01,0.1,1.,10.]:specs.append({'family':'lrl2','params':{'C':C}})
    for depth,leaf in [(5,1),(10,1),(None,1),(None,3)]:
        specs.append({'family':'rf','params':{'n_estimators':64,'max_depth':depth,'min_samples_leaf':leaf}})
    for lr,it,leaves in [(0.03,100,7),(0.1,100,7),(0.1,100,15),(0.1,150,31)]:
        specs.append({'family':'histgb','params':{'learning_rate':lr,'max_iter':it,'max_leaf_nodes':leaves}})
    for i,spec in enumerate(specs):spec['candidate_id']=f'manual_{i:02d}'
    return specs


def build_pipeline(spec):
    family,p=spec['family'],dict(spec.get('params',{}))
    for key in ['n_jobs','FLAML_sample_size','_estimator_type']:p.pop(key,None)
    if family=='lrl2':model=LogisticRegression(max_iter=5000,random_state=42,**p)
    elif family=='rf':model=RandomForestClassifier(n_jobs=1,random_state=42,**p)
    elif family=='extra_tree':model=ExtraTreesClassifier(n_jobs=1,random_state=42,**p)
    elif family=='histgb':model=HistGradientBoostingClassifier(early_stopping=False,random_state=42,**p)
    elif family=='kneighbor':model=KNeighborsClassifier(n_jobs=1,**p)
    else:raise ValueError(f'unknown family: {family}')
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('scaler',StandardScaler()),('model',model)])


def probability_metrics(y_true,probability,threshold=.5):
    y,p=np.asarray(y_true),np.asarray(probability,float)
    if p.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probabilities')
    pred=p>=threshold
    return {'ap':average_precision_score(y,p),'roc_auc':roc_auc_score(y,p),
            'f1':f1_score(y,pred,zero_division=0),'precision':precision_score(y,pred,zero_division=0),
            'recall':recall_score(y,pred,zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,pred),
            'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1])}


def evaluate_candidate_specs(specs,X,y,cv_splits,primary_metric='ap'):
    if primary_metric!='ap':raise ValueError('this contract requires AP')
    rows=[];start=time.perf_counter()
    for i,spec in enumerate(specs):
        values=[];t=time.perf_counter()
        for tr,va in cv_splits:
            m=build_pipeline(spec).fit(X[tr],y[tr]);values.append(average_precision_score(y[va],m.predict_proba(X[va])[:,1]))
        rows.append({'trial':i,'candidate_id':spec.get('candidate_id',f'manual_{i:02d}'),'family':spec['family'],'params':json.dumps(spec['params'],sort_keys=True),
                     'inner_ap':np.mean(values),'inner_ap_std':np.std(values,ddof=1),
                     'fold_ap':json.dumps(values),'seconds':time.perf_counter()-t,'cv_fits':len(cv_splits)})
    history=pd.DataFrame(rows);winner=int(history.inner_ap.to_numpy().argmax())
    return specs[winner],history,time.perf_counter()-start


class FrozenCV(KFold):
    """Immutable positional split contract; FLAML calls split(X) without y."""
    def __init__(self,splits):
        super().__init__(n_splits=len(splits),shuffle=False)
        self.splits=[(np.array(a,copy=True),np.array(b,copy=True)) for a,b in splits]
    def split(self,X,y=None,groups=None):
        for a,b in self.splits:yield a.copy(),b.copy()


class PipelineLearner(BaseEstimator):
    family=None
    fit_audit=[]
    @classmethod
    def search_space(cls,**kwargs):
        return {k:{'domain':tune.choice(v),'init_value':v[0]} for k,v in SPACE[cls.family].items()}
    def _fit(self,X_train,y_train,**kwargs):
        t=time.perf_counter()
        self._model=build_pipeline({'family':self.family,'params':self.params}).fit(np.asarray(X_train),np.asarray(y_train))
        PipelineLearner.fit_audit.append({'family':self.family,'n':len(y_train),
                                         'scaler_n':int(self._model['scaler'].n_samples_seen_)})
        return time.perf_counter()-t


class LR(PipelineLearner):family='lrl2'
class RF(PipelineLearner):family='rf'
class ET(PipelineLearner):family='extra_tree'
class HG(PipelineLearner):family='histgb'
class KN(PipelineLearner):family='kneighbor'
LEARNERS={'lrl2':LR,'rf':RF,'extra_tree':ET,'histgb':HG,'kneighbor':KN}


def evaluate_flaml(X,y,inner_cv,seed,budget,log_path):
    path=Path(log_path);path.parent.mkdir(exist_ok=True,parents=True)
    if path.exists():path.unlink()
    automl=AutoML();PipelineLearner.fit_audit=[]
    for name,cls in LEARNERS.items():automl.add_learner(name,cls)
    start=time.perf_counter()
    automl.fit(X_train=X,y_train=y,task='classification',metric='ap',
               estimator_list=list(LEARNERS),max_iter=budget,eval_method='cv',
               split_type=FrozenCV(inner_cv),learner_selector='roundrobin',
               log_file_name=str(path),log_type='all',ensemble=False,
               n_jobs=1,seed=seed,sample=False,auto_augment=False,
               skip_transform=True,retrain_full=True,verbose=0,early_stop=False)
    elapsed=time.perf_counter()-start
    records=[json.loads(l) for l in path.read_text().splitlines() if 'validation_loss' in json.loads(l)]
    if len(records)!=budget or not np.isfinite([r['validation_loss'] for r in records]).all():
        raise ValueError(f'completed {len(records)} / {budget} trials')
    history=pd.DataFrame([{'trial':r['record_id'],'candidate_id':f"automl_seed{seed}_trial{r['record_id']}",'family':r['learner'],'params':json.dumps(r['config'],sort_keys=True),
                           'inner_ap':1-r['validation_loss'],'seconds':r['trial_time'],'cv_fits':len(inner_cv)} for r in records])
    best=min(records,key=lambda r:r['validation_loss'])  # first best trial, as FLAML keeps strict improvements
    spec={'family':automl.best_estimator,'params':dict(automl.best_config),'candidate_id':f"automl_seed{seed}_trial{best['record_id']}"}
    if best['learner']!=spec['family']:raise AssertionError('best FLAML record does not match best estimator')
    audit=list(PipelineLearner.fit_audit)
    if any(r['n']!=r['scaler_n'] for r in audit):raise AssertionError('scaler fit outside learner fold')
    return spec,history,elapsed,audit


def bootstrap_mean_interval(values,n_bootstrap=10000,seed=42):
    v=np.asarray(values,float)
    if v.ndim!=1 or len(v)==0 or not np.isfinite(v).all():raise ValueError('invalid bootstrap input')
    samples=np.random.default_rng(seed).choice(v,size=(n_bootstrap,len(v)),replace=True).mean(axis=1)
    return np.quantile(samples,[.025,.975])


def pairwise_prediction_disagreement(probability_runs,threshold=.5):
    p=np.asarray(probability_runs)
    if p.ndim!=2 or len(p)<2:raise ValueError('need >=2 runs')
    return [float(np.mean((a>=threshold)!=(b>=threshold))) for a,b in combinations(p,2)]


def serialized_size_bytes(model):
    stream=io.BytesIO();joblib.dump(model,stream,compress=3);return len(stream.getvalue())


def prediction_latency_ms(model,X,repeats=45):
    for _ in range(7):model.predict_proba(X)
    values=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();model.predict_proba(X);values.append((time.perf_counter_ns()-t)/1e6)
    return {'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95))}


def sha256_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
