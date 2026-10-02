"""Дані, спліт і кандидати-моделі для практичної № 5 (єдине джерело).

Параметри кандидатів НЕ підбираються тут: вони взяті з попередніх робіт
(див. поле source у describe_candidates). Поріг рішення фіксований заздалегідь.
Тестова частина даних у benchmark / відборі не використовується.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
THRESHOLD = 0.5          # фіксований до порівняння моделей
CV_SPLITS = 5


def to_snake(col: str) -> str:
    """<property>_<mean|se|worst> — той самий код, що в практичній № 1."""
    parts = col.split()
    if parts[0] in ("mean", "worst"):
        return "_".join(parts[1:]) + "_" + parts[0]
    if parts[-1] == "error":
        return "_".join(parts[:-1]) + "_se"
    return col


@dataclass(frozen=True)
class Split:
    """Спліт 80/20 (stratify, random_state=42); y: 1 = malignant."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_idx: np.ndarray
    test_idx: np.ndarray


def load_data() -> Split:
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = [to_snake(c) for c in X.columns]
    t = raw.target                                   # sklearn: 0 = malignant
    X_train, X_test, t_train, t_test = train_test_split(
        X, t, test_size=0.2, stratify=t, random_state=RANDOM_STATE
    )
    return Split(
        X_train=X_train, X_test=X_test,
        y_train=1 - t_train, y_test=1 - t_test,
        train_idx=X_train.index.to_numpy(), test_idx=X_test.index.to_numpy(),
    )


def make_cv() -> StratifiedKFold:
    """Однакові фолди для всіх моделей."""
    return StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def _imputer():
    return ("imputer", SimpleImputer(strategy="median"))


def _scaler():
    return ("scaler", StandardScaler())


def _logreg() -> Pipeline:
    return Pipeline([_imputer(), _scaler(),
                     ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE))])


def _tree() -> Pipeline:
    return Pipeline([_imputer(),
                     ("clf", DecisionTreeClassifier(max_depth=3, min_samples_leaf=16,
                                                    random_state=RANDOM_STATE))])


def _knn() -> Pipeline:
    return Pipeline([_imputer(), _scaler(),
                     ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))])


def _svm_rbf() -> Pipeline:
    # SVC(probability=True) у sklearn 1.9 дає FutureWarning; еквівалент -
    # Platt scaling на внутрішніх 5-fold значеннях рішення + один фінальний SVC.
    svc = SVC(C=1.0, gamma="scale", random_state=RANDOM_STATE)
    return Pipeline([_imputer(), _scaler(),
                     ("clf", CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False))])


def _random_forest() -> Pipeline:
    return Pipeline([_imputer(),
                     ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                    random_state=RANDOM_STATE, n_jobs=1))])


def _hist_gb() -> Pipeline:
    return Pipeline([_imputer(),
                     ("clf", HistGradientBoostingClassifier(
                         max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                         min_samples_leaf=20, l2_regularization=1.0,
                         early_stopping=False, random_state=RANDOM_STATE))])


# назва -> фабрика (кожен виклик повертає новий незавчений pipeline)
CANDIDATES: Dict[str, Callable[[], Pipeline]] = {
    "logreg": _logreg,
    "tree": _tree,
    "knn": _knn,
    "svm_rbf": _svm_rbf,
    "random_forest": _random_forest,
    "hist_gb": _hist_gb,
}

_SOURCE = {
    "logreg": "ПР № 4",
    "tree": "ЛР № 4",
    "knn": "ЛР № 3 (тип моделі)",
    "svm_rbf": "ЛР № 3 (тип моделі)",
    "random_forest": "ЛР № 4",
    "hist_gb": "ЛР № 4",
}

# які гіперпараметри останнього кроку потрапляють у метадані
_PARAM_KEYS = {
    "logreg": ["C", "max_iter", "random_state"],
    "tree": ["max_depth", "min_samples_leaf", "random_state"],
    "knn": ["n_neighbors", "weights"],
    "svm_rbf": ["method", "cv", "ensemble", "estimator__C", "estimator__gamma", "estimator__random_state"],
    "random_forest": ["n_estimators", "max_features", "random_state", "n_jobs"],
    "hist_gb": ["max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                "l2_regularization", "early_stopping", "random_state"],
}


def describe_candidates() -> List[dict]:
    """Опис кандидатів для experiment_metadata.json."""
    out = []
    for name, factory in CANDIDATES.items():
        pipe = factory()
        last = pipe.steps[-1][1]
        params = last.get_params()
        out.append({
            "name": name,
            "steps": [s for s, _ in pipe.steps],
            "classifier": type(last).__name__,
            "params": {k: params[k] if isinstance(params[k], (int, float, str, bool, type(None)))
                       else str(params[k]) for k in _PARAM_KEYS[name]},
            "source_work": _SOURCE[name],
        })
    return out
