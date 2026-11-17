import json

import joblib
import numpy as np
import pandas as pd

from detection.detectors.TabuPIO.model import HybridModel
from detection.detectors.TabuPIO import preprocess
from detection.detectors.TabuPIO.train import run_paper


def test_scaler_reference_is_distinct_from_ocsvm_fit_population():
    benign=np.array([[0.,0.],[1.,1.],[2.,2.]])
    reference=np.vstack([benign,[100.,200.]])
    model=HybridModel(['a','b'],[True,False],.1,'unresolved').fit(benign,scaler_reference=reference)
    assert model.scaler.data_max_.tolist()==[100.]
    assert model.classifier.shape_fit_[0]==3


def test_paper_selection_and_scaler_do_not_use_test_data(tmp_path,monkeypatch):
    monkeypatch.setattr(preprocess,'EXPECTED',{
        'train':{'Normal':8,'Attack':8},'test':{'Normal':8,'Attack':8}})
    data=tmp_path/'data';data.mkdir()
    rng=np.random.default_rng(9)
    frame=pd.DataFrame(np.vstack([rng.normal(0,.01,(8,42)),rng.normal(2,.01,(8,42))]),
                       columns=preprocess.FEATURES)
    frame['label']=[0]*8+[1]*8
    frame.to_csv(data/'train-numeric.csv',index=False)
    frame.to_csv(data/'test-numeric.csv',index=False)
    (data/'preprocessing.json').write_text(json.dumps({'features':preprocess.FEATURES,'feature_count':42}))
    run_paper(data,tmp_path/'first')
    frame[preprocess.FEATURES]=1000.
    frame['label']=1-frame.label
    frame.to_csv(data/'test-numeric.csv',index=False)
    run_paper(data,tmp_path/'second')
    a=joblib.load(tmp_path/'first/model-0.joblib')['model']
    b=joblib.load(tmp_path/'second/model-0.joblib')['model']
    assert a.nu==b.nu
    np.testing.assert_array_equal(a.scaler.data_max_,b.scaler.data_max_)
    np.testing.assert_array_equal(a.classifier.support_vectors_,b.classifier.support_vectors_)
