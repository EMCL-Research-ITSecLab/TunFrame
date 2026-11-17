import json
from pathlib import Path
import threading

import joblib
import numpy as np
import pandas as pd
import pytest

from detection.detectors.TabuPIO.model import (HybridModel, SearchSettings,
    classification_metrics, fitness, length_alarm, select_features)
from detection.detectors.TabuPIO.detector import TabuPIO
from detection.detectors.TabuPIO.train import load_splits


def test_length_boundaries():
    lengths = [99,100,115,116,144,145,185,186,219,220,340,341]
    assert length_alarm(lengths).tolist() == [False,True,True,False,False,True,True,False,False,True,True,False]
    with pytest.raises(ValueError):
        length_alarm([float('nan')])


def test_confusion_partition_and_undefined_metrics():
    m = classification_metrics([1,1,0,0], [1,0,1,0])
    assert [m[k] for k in ['tp','fn','fp','tn']] == [1,1,1,1]
    assert sum(m[k] for k in ['tp','fn','fp','tn']) == m['n']
    assert m['f1'] == .5
    assert classification_metrics([],[])['recall'] is None
    assert classification_metrics([0],[0])['precision'] is None
    assert fitness(1,2,dict(recall=.5,fpr=.1)) == pytest.approx(1.028)
    assert np.isinf(fitness(1,2,dict(recall=0,fpr=0)))






def test_model_roundtrip_and_shortcut(tmp_path):
    x=np.array([[0.,1.],[.1,.9],[.2,.8],[.3,.7]])
    model=HybridModel(['a','b'],[True,False],.1,'dns').fit(x)
    test=np.array([[.1,.9],[100.,100.]])
    expected=model.predict(test,[101,500])
    path=tmp_path/'model.joblib'; joblib.dump(model,path)
    assert np.array_equal(joblib.load(path).predict(test,[101,500]),expected)
    assert model.predict(test,[101,500],'length_only').tolist() == [1,0]
    class ExplodingClassifier:
        def predict(self,x): raise AssertionError('Shortcut must bypass inference')
    model.classifier=ExplodingClassifier()
    assert model.predict(test,[101,150]).tolist() == [1,1]
    assert model.predict([[float('nan'),0]],[101]).tolist() == [1]
    with pytest.raises(ValueError): model.predict([[float('nan'),0]],[500])


def test_optimizer_determinism_and_minimization():
    target=np.array([True,False,True,False])
    objective=lambda mask: float(np.count_nonzero(mask != target))
    settings=SearchSettings(population=8,iterations=5,local_iterations=3,seed=7)
    first, history=select_features(objective,4,settings)
    second,_=select_features(objective,4,settings)
    assert np.array_equal(first,second)
    assert np.array_equal(first,target)
    assert all(a['fitness'] >= b['fitness'] for a,b in zip(history,history[1:]))


def test_adapter_requires_measured_layer_and_resets():
    detector=TabuPIO(dict(variant='length_only',length_layer='dns'))
    with pytest.raises(ValueError): detector.detect({'dns':{'qname':'tunnel.com'}})
    event={'dns':{'qname':'x.tunnel.com'},'tabu_pio':{'packet_length':101,'length_layer':'dns'}}
    detector.detect(event)
    assert detector.alarms == {'tunnel.com'}
    assert detector.event_predictions[0]['prediction'] == 1
    detector.reset()
    assert detector.alarms == set() and detector.event_predictions == []


def test_splits_reject_group_leakage(tmp_path):
    paths={}
    for i,name in enumerate(['train','validation','test']):
        paths[name]=tmp_path/f'{name}.csv'
        pd.DataFrame(dict(label=[0,1],group=['same','also_same'],a=[i*2,i*2+1])).to_csv(paths[name],index=False)
    with pytest.raises(ValueError,match='overlapping groups'):
        load_splits(paths,['a'],'group',None)



