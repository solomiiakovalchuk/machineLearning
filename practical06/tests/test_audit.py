import sys, json, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audit as A


class TestAudit(unittest.TestCase):
    def valid(self,**kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{}')
            s={'id':[3]}
            for signal in ['if30','ifpca','rec','combined']:
                s[f'score_{signal}']=[.2];s[f'pct_{signal}']=[.2]
            args=dict(paths=[p],train_ids=[1,2],test_ids=[3],X=[[0,1],[2,3]],labels=[0,1],
                      scores=pd.DataFrame(s),k=2)
            args.update(kwargs);return A.validate_artifacts(**args)
    def test_valid_contract(self):self.assertTrue(self.valid()['valid'])
    def test_missing(self):
        with self.assertRaises(ValueError):self.valid(paths=[ROOT/'absent.json'])
    def test_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text('{bad')
            with self.assertRaises(ValueError):self.valid(paths=[p])
    def test_duplicate(self):
        with self.assertRaises(ValueError):self.valid(train_ids=[1,1])
    def test_overlap(self):
        with self.assertRaises(ValueError):self.valid(test_ids=[1])
    def test_shapes(self):
        with self.assertRaises(ValueError):self.valid(X=[[1,2]])
    def test_nonfinite(self):
        with self.assertRaises(ValueError):self.valid(X=[[np.inf,1],[2,3]])
    def test_cluster_ids(self):
        with self.assertRaises(ValueError):self.valid(labels=[1,2])
    def test_percentile_range(self):
        s={'id':[3]}
        for signal in ['if30','ifpca','rec','combined']:
            s[f'score_{signal}']=[.2];s[f'pct_{signal}']=[.2]
        s['pct_rec']=[1.1]
        with self.assertRaises(ValueError):self.valid(scores=pd.DataFrame(s))
    def test_score_schema(self):
        with self.assertRaises(ValueError):self.valid(scores=pd.DataFrame({'id':[3],'score_rec':['text']}))
    def test_alignment(self):np.testing.assert_array_equal(A.align([0,0,1,1],[1,1,0,0]),[0,0,1,1])
    def test_distance(self):
        d=A.compute_boundary_diagnostics([[1,0]],[[0,0],[4,0]])
        self.assertEqual(d.distance.iloc[0],1);self.assertAlmostEqual(d.margin.iloc[0],2/3)
    def test_representative_counts(self):
        d=A.compute_boundary_diagnostics([[i,0] for i in range(10)],[[0,0],[9,0]])
        r=A.find_cluster_representatives(range(10),d)
        self.assertTrue((r.groupby(['cluster','kind']).size()==3).all())
    def test_jaccard(self):self.assertAlmostEqual(A.jaccard({1,2},{2,3}),1/3)
    def test_constant_and_nonmutation(self):
        x=pd.DataFrame({'a':[1.,1.,1.,1.],'b':[0.,1.,3.,4.]});old=x.copy(deep=True)
        p=A.build_cluster_profiles(x,[0,0,1,1]);self.assertTrue((p.query("feature == 'a'").profile==0).all())
        pd.testing.assert_frame_equal(x,old)
    def test_policy_priority(self):
        rules=[{'action':'manual_review','column':'p','operator':'ge','threshold':.9},
               {'action':'data_quality_check','column':'p','operator':'ge','threshold':.8}]
        r=A.apply_action_policy(pd.DataFrame({'p':[1.,.5]}),rules)
        self.assertEqual(r.action.tolist(),['manual_review','no_action']);self.assertTrue(r.conflict.iloc[0])
    def test_card_complete(self):
        c=pd.read_csv(ROOT/'results/anomaly_cards.csv').iloc[0].to_dict()
        card=A.explain_anomaly(c,'test')
        self.assertTrue({'id','cluster','margin','stability','hypothesis','limitation','triggered_rules','recommended_action'}<=set(card))
    def test_saved_bundle_and_percentiles(self):
        sys.path.insert(0,str(ROOT.parent/'lab05'))
        from src import unsupervised_pipeline as U
        b=U.load_bundle(ROOT.parent/'lab05/artifacts/lr05_bundle.joblib')
        for detector in U.DETECTORS:
            ref=np.sort(b['detectors'].train_scores[detector])
            result=U.empirical_percentile(ref,ref)
            self.assertEqual(result[-1],1.)
            self.assertTrue(np.all(np.diff(result)>=0))
        self.assertEqual(b['pca'].n_components_,6)
    def test_output_completeness(self):
        cards=pd.read_csv(ROOT/'results/anomaly_cards.csv')
        self.assertEqual(len(cards),15);self.assertEqual(cards.id.nunique(),15)
        self.assertEqual(len(list((ROOT/'figures').glob('*.png'))),7)
        self.assertEqual(len(pd.read_csv(ROOT/'results/object_stability.csv')),455)
        self.assertEqual(len(pd.read_csv(ROOT/'results/cluster_profiles.csv')),60)
    def test_disagreement_cards_really_disagree(self):
        cards=pd.read_csv(ROOT/'results/anomaly_cards.csv')
        scores=pd.read_csv(ROOT/'results/scores_actions.csv').set_index('id')
        ids=cards.loc[cards.category.eq('detector_disagreement'),'id']
        self.assertEqual(len(ids),5)
        self.assertTrue((scores.loc[ids,'flag_ifpca']!=scores.loc[ids,'flag_rec']).all())

if __name__=='__main__':unittest.main()
