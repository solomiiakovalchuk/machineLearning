"""Run from lab06/handoff/: python inference.py. Includes serving schema checks."""
import json
from pathlib import Path
import joblib
import numpy as np


def predict(model,X):
    X=np.asarray(X,float)
    if X.ndim!=2 or X.shape[1]!=64:raise ValueError('expected n x 64 ordered pixels')
    if not np.isfinite(X).all() or (X<0).any() or (X>16).any():raise ValueError('pixels must be finite and in [0,16]')
    return model.predict_proba(X)[:,1]


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    example=json.loads((root/'inference_example.json').read_text())
    model=joblib.load(root/'pipeline.joblib');p=predict(model,example['input'])
    np.testing.assert_allclose(p,example['expected_probability'],rtol=1e-10,atol=1e-12)
    print(p.tolist())
