"""Run from lab06/handoff/: python inference.py. Includes serving schema checks."""
import json
from pathlib import Path
import joblib
import numpy as np


def predict(model,X):
    X=np.asarray(X,float)
    if X.ndim!=2 or X.shape[1]!=30 or len(X)==0:raise ValueError('expected n x 30 ordered features (see data_schema.json)')
    if not np.isfinite(X).all() or (X<0).any():raise ValueError('features must be finite and non-negative')
    return model.predict_proba(X)[:,1]


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    example=json.loads((root/'inference_example.json').read_text())
    model=joblib.load(root/'pipeline.joblib');p=predict(model,example['input'])
    np.testing.assert_allclose(p,example['expected_probability'],rtol=1e-10,atol=1e-12)
    print(p.tolist())
