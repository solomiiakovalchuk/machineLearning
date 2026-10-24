"""Pure audit functions for the fixed LR5 experiment; no diagnostic labels in policy."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score


def validate_artifacts(paths, train_ids, test_ids, X, labels, scores, k):
    for path in paths:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError(f'missing or empty artifact: {path}')
        try:
            if Path(path).suffix == '.json': json.loads(Path(path).read_text(encoding='utf-8'))
            if Path(path).suffix == '.csv': pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f'corrupt artifact: {path}') from exc
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError('duplicate object id')
    if set(train_ids) & set(test_ids):
        raise ValueError('train/test overlap')
    X, labels = np.asarray(X), np.asarray(labels)
    signals=['if30','ifpca','rec','combined']
    required={'id',*[f'score_{s}' for s in signals],*[f'pct_{s}' for s in signals]}
    if not required.issubset(scores.columns):
        raise ValueError('missing required score columns')
    if not all(pd.api.types.is_numeric_dtype(scores[c]) for c in required):
        raise ValueError('nonnumeric score column')
    if X.ndim != 2 or len(X) != len(train_ids) or labels.shape != (len(X),):
        raise ValueError('incompatible matrix shapes')
    if not np.isfinite(X).all() or not np.isfinite(scores.select_dtypes('number')).all().all():
        raise ValueError('nonfinite data or scores')
    if scores.id.duplicated().any() or set(scores.id) != set(test_ids):
        raise ValueError('score ids do not match test')
    if not np.issubdtype(labels.dtype, np.integer) or set(labels) != set(range(k)):
        raise ValueError('invalid cluster ids')
    for col in [c for c in scores if c.startswith('pct_')]:
        if not scores[col].between(0, 1).all():
            raise ValueError(f'percentile outside [0,1]: {col}')
    return {'valid': True, 'train_n': len(train_ids), 'test_n': len(test_ids), 'k': k,
            'input_sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}


def build_cluster_profiles(X, labels):
    X = pd.DataFrame(X).copy()
    labels = np.asarray(labels)
    if len(X) != len(labels) or not np.isfinite(X).all().all():
        raise ValueError('invalid profile input')
    mean, sd = X.mean(), X.std(ddof=0)
    rows = []
    for c in np.unique(labels):
        group = X.iloc[np.flatnonzero(labels == c)]
        for feature in X:
            rows.append({'cluster': int(c), 'feature': feature, 'n': len(group),
                         'fraction': len(group)/len(X), 'mean': group[feature].mean(),
                         'median': group[feature].median(), 'std': group[feature].std(ddof=0),
                         'profile': (group[feature].mean()-mean[feature])/sd[feature] if sd[feature] else 0.})
    return pd.DataFrame(rows)


def compute_boundary_diagnostics(X, centers):
    X, centers = np.asarray(X, float), np.asarray(centers, float)
    if X.ndim != 2 or centers.ndim != 2 or X.shape[1] != centers.shape[1] or len(centers)<2:
        raise ValueError('invalid distance shapes')
    distances = cdist(X, centers)
    order = np.argsort(distances, axis=1, kind='stable')
    d1, d2 = np.take_along_axis(distances, order[:,:2], axis=1).T
    return pd.DataFrame({'cluster': order[:,0], 'distance': d1, 'second_distance': d2,
                         'margin': (d2-d1)/np.maximum(d2,np.finfo(float).eps)})


def find_cluster_representatives(ids, diagnostics, n=3):
    tab = diagnostics.copy(); tab.insert(0,'id',ids)
    rows=[]
    for c,g in tab.groupby('cluster'):
        if len(g)<n: raise ValueError('cluster too small')
        for kind,col,asc in [('typical','distance',True),('distant','distance',False),('boundary','margin',True)]:
            chosen=g.sort_values([col,'id'],ascending=[asc,True]).head(n).copy()
            chosen.insert(1,'kind',kind); rows.append(chosen)
    return pd.concat(rows,ignore_index=True)


def align(reference, candidate):
    reference,candidate=np.asarray(reference),np.asarray(candidate)
    if reference.shape!=candidate.shape: raise ValueError('assignment shape mismatch')
    a,b=np.unique(reference),np.unique(candidate)
    if len(a)!=len(b): raise ValueError('different cluster count')
    table=np.array([[np.sum((candidate==j)&(reference==i)) for i in a] for j in b])
    r,c=linear_sum_assignment(-table); mapping={b[i]:a[j] for i,j in zip(r,c)}
    return np.array([mapping[x] for x in candidate])


def compare_cluster_runs(reference, assignments):
    if len(assignments)<1: raise ValueError('no repeats')
    aligned=np.array([align(reference,a) for a in assignments])
    return np.mean(aligned==reference,axis=0), aligned, float(np.mean([
        adjusted_rand_score(a,b) for a,b in combinations(assignments,2)])) if len(assignments)>1 else 1.


def top_features(profiles, n=5):
    return {int(c): set(g.assign(abs_profile=g.profile.abs()).sort_values(
        ['abs_profile','feature'],ascending=[False,True]).head(n).feature) for c,g in profiles.groupby('cluster')}


def jaccard(a,b):
    return len(set(a)&set(b))/len(set(a)|set(b)) if set(a)|set(b) else 1.


def apply_action_policy(scores, rules):
    """First matched rule determines action, every matching condition is retained."""
    for rule in rules:
        if rule['operator'] not in ('ge','lt') or not np.isfinite(rule['threshold']):
            raise ValueError('invalid policy operator or threshold')
        if rule['column'] not in scores or not pd.api.types.is_numeric_dtype(scores[rule['column']]):
            raise ValueError('policy column must be numeric')
        if not np.isfinite(scores[rule['column']]).all():raise ValueError('nonfinite policy input')
    result=[]
    for _,row in scores.iterrows():
        triggered=[]
        for rule in rules:
            value=float(row[rule['column']]); threshold=rule['threshold']
            if (rule['operator']=='ge' and value>=threshold) or (rule['operator']=='lt' and value<threshold):
                triggered.append(rule['action'])
        result.append({'action':triggered[0] if triggered else 'no_action',
                       'triggered_rules':'|'.join(triggered),'conflict':len(triggered)>1})
    return pd.DataFrame(result,index=scores.index)


def explain_anomaly(row, category):
    keys=['id','cluster','stability','margin','score_if30','score_ifpca','score_rec','score_combined',
          'pct_if30','pct_ifpca','pct_rec','pct_combined','action','triggered_rules',
          'largest_residual_feature','largest_residual']
    card={k:row[k].item() if isinstance(row[k],np.generic) else row[k] for k in keys}
    card.update(category=category,
                hypothesis=f"Перевірити вимірювання {row['largest_residual_feature']}; найбільший квадрат залишку = {row['largest_residual']:.4f}",
                recommended_action=row['action'],
                limitation='Нетиповість не встановлює причину і не є діагнозом; контрольні спотворення штучні.')
    return card
