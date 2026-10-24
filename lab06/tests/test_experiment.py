import sys,json,unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiment as E
from inference import predict


class TestFunctions(unittest.TestCase):
    def test_specs_and_families(self):
        self.assertEqual(len(E.manual_specs()),12)
        self.assertEqual(len(set(s['family'] for s in E.manual_specs())),3)
        self.assertEqual(set(E.LEARNERS),set(E.SPACE))
    def test_pipeline(self):
        m=E.build_pipeline({'family':'lrl2','params':{}})
        self.assertEqual(list(m.named_steps),['imputer','scaler','model'])
        with self.assertRaises(ValueError):E.build_pipeline({'family':'absent'})
    def test_metrics(self):
        m=E.probability_metrics([0,1],[.1,.9]);self.assertEqual(m['ap'],1);self.assertEqual(m['f1'],1)
        self.assertAlmostEqual(m['brier'],.01)
    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):E.probability_metrics([0,1],[0,np.nan])
        with self.assertRaises(ValueError):E.probability_metrics([0,1],[0,1.1])
    def test_threshold_equality(self):
        self.assertEqual(E.probability_metrics([0,1],[.1,.5])['recall'],1.)
    def test_bootstrap_constant(self):np.testing.assert_allclose(E.bootstrap_mean_interval([.2,.2]),[.2,.2])
    def test_disagreement(self):self.assertEqual(E.pairwise_prediction_disagreement([[.1,.9],[.8,.9]]),[.5])
    def test_frozen_cv(self):
        cv=E.FrozenCV([(np.array([0,1]),np.array([2])),(np.array([1,2]),np.array([0]))])
        a=list(cv.split(np.zeros((3,2))));a[0][0][0]=999
        self.assertEqual(list(cv.split(np.zeros((3,2))))[0][0][0],0)
    def test_fold_preprocessing(self):
        X=np.array([[0.,1.],[2.,1.],[1000.,9.],[2000.,9.]])
        m=E.build_pipeline({'family':'lrl2','params':{}}).fit(X[:2],[0,1])
        np.testing.assert_allclose(m['scaler'].mean_,[1,1])
        self.assertFalse(np.allclose(m['scaler'].mean_,X.mean(axis=0)))
    def test_candidate_search(self):
        x,y=make_classification(n_samples=60,n_features=4,n_redundant=0,random_state=42)
        cv=list(StratifiedKFold(3,shuffle=True,random_state=42).split(x,y))
        s,h,_=E.evaluate_candidate_specs(E.manual_specs()[:2],x,y,cv)
        self.assertEqual(len(h),2);self.assertTrue(h.inner_ap.between(0,1).all())
    def test_latency_size_hash(self):
        x,y=make_classification(n_samples=20,n_features=4,random_state=42)
        m=E.build_pipeline({'family':'lrl2'}).fit(x,y)
        self.assertGreater(E.serialized_size_bytes(m),0)
        self.assertGreater(E.prediction_latency_ms(m,x[:1],5)['p95_ms'],0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a';p.write_bytes(b'abc')
            self.assertEqual(E.sha256_file(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    def test_serving_schema(self):
        with self.assertRaises(ValueError):predict(None,np.zeros((1,63)))
        with self.assertRaises(ValueError):predict(None,np.full((1,64),17))


class TestArtifacts(unittest.TestCase):
    def test_oof_once(self):
        o=pd.read_csv(ROOT/'results/oof_predictions.csv')
        self.assertEqual(len(o),1437);self.assertTrue(o.id.is_unique);self.assertFalse(o.isna().any().any())
        self.assertEqual(set(o.fold),set(range(5)))
    def test_budget_actual_fits(self):
        b=pd.read_csv(ROOT/'results/budget_audit.csv');self.assertEqual(len(b),16)
        self.assertTrue((b.candidates==12).all());self.assertTrue((b.cv_fits==36).all())
        a=pd.read_csv(ROOT/'results/flaml_fit_audit.csv');self.assertEqual(len(a),296)
        self.assertTrue((a.n==a.scaler_n).all())
        self.assertTrue((b[b.branch.eq('automl')].framework_refits==1).all())
    def test_candidate_history(self):
        h=pd.read_csv(ROOT/'results/candidate_history.csv')
        self.assertEqual(len(h),192);self.assertTrue(h.inner_ap.between(0,1).all())
        for path in (ROOT/'results/logs').glob('*.jsonl'):
            records=[json.loads(s) for s in path.read_text().splitlines() if 'validation_loss' in json.loads(s)]
            self.assertEqual(len(records),12)
    def test_split_nesting(self):
        splits=pd.read_csv(ROOT/'results/split_ids.csv');train=set(splits[splits.part.eq('train')].id);test=set(splits[splits.part.eq('test')].id)
        self.assertFalse(train&test)
        for fold in json.loads((ROOT/'results/cv_splits.json').read_text()):
            tr=set(fold['outer_train_ids']);va=set(fold['outer_val_ids']);self.assertFalse(tr&va);self.assertEqual(tr|va,train)
            for inner in fold['inner']:
                a=set(inner['train_ids']);b=set(inner['val_ids']);self.assertFalse(a&b);self.assertEqual(a|b,tr)
    def test_decision_integrity(self):
        r=ROOT/'results';d=json.loads((r/'decision_record.json').read_text());t=json.loads((r/'test_metrics.json').read_text())
        self.assertFalse(d['test_metrics_present']);self.assertEqual(d['selected_branch'],t['branch'])
        self.assertEqual(E.sha256_file(r/'decision_record.json'),(r/'decision.sha256').read_text().strip())
        self.assertEqual(t['decision_sha256'],E.sha256_file(r/'decision_record.json'))
    def test_selected_rule(self):
        eng=pd.read_csv(ROOT/'results/engineering.csv');d=json.loads((ROOT/'results/decision_record.json').read_text())
        delta=d['cv_evidence']['mean_difference'];eps=d['epsilon']
        expected='automl' if delta>eps else 'manual' if delta < -eps else eng.sort_values(['size_bytes','single_p95_ms','branch']).iloc[0].branch
        self.assertEqual(d['selected_branch'],expected)
    def test_metrics_recompute(self):
        o=pd.read_csv(ROOT/'results/oof_predictions.csv');f=pd.read_csv(ROOT/'results/fold_metrics.csv')
        for row in f.itertuples():
            g=o[o.fold.eq(row.fold)];m=E.probability_metrics(g.y,g[row.branch])
            self.assertAlmostEqual(m['ap'],row.ap,places=12)
        t=pd.read_csv(ROOT/'results/test_predictions.csv');metrics=json.loads((ROOT/'results/test_metrics.json').read_text())
        for key,value in E.probability_metrics(t.y,t.probability).items():self.assertAlmostEqual(value,metrics[key],places=12)
    def test_handoff_prediction(self):
        h=ROOT/'handoff';ex=json.loads((h/'inference_example.json').read_text());m=joblib.load(h/'pipeline.joblib')
        np.testing.assert_allclose(predict(m,ex['input']),ex['expected_probability'],rtol=1e-10,atol=1e-12)
        self.assertEqual(E.sha256_file(h/'decision_record.json'),E.sha256_file(ROOT/'results/decision_record.json'))
    def test_heldout_importance(self):
        p=pd.read_csv(ROOT/'results/permutation_importance.csv')
        self.assertEqual(len(p),5*2*64)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):unittest.main()
