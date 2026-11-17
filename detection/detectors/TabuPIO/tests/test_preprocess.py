import numpy as np
import pandas as pd
import pytest

from detection.detectors.TabuPIO.preprocess import FEATURES, dns_records, encode_and_scale


def records():
    frame = pd.DataFrame([{feature: 0 for feature in FEATURES} for _ in range(3)])
    frame.insert(0, 'id', [1, 2, 3])
    frame['proto'], frame['service'], frame['state'] = 'udp', 'dns', 'INT'
    frame['attack_cat'], frame['label'] = 'Normal', 0
    return frame


def test_preserve_distinct_record_ids_and_only_filter_dns():
    frame = records()
    frame.loc[2, 'service'] = 'http'
    selected, removed = dns_records(pd.concat([frame, frame.iloc[:1]]))
    assert removed == 1
    assert selected.id.tolist() == [1, 2]
    assert selected.drop(columns='id').duplicated().sum() == 1


def test_reject_inconsistent_labels():
    frame = records()
    frame.loc[0, 'label'] = 1
    with pytest.raises(ValueError, match='disagrees'):
        dns_records(frame)


def test_scaling_uses_training_extrema_without_clipping_test():
    train, test = records(), records()
    train['dur'] = [0., 5., 10.]
    test['dur'] = [20., 30., 40.]
    encoded, scaled, mappings, scaler = encode_and_scale(train, test)
    assert len(FEATURES) == 42
    assert 'id' not in encoded['train'] and 'label' not in encoded['train']
    assert scaled['train'].dur.tolist() == [0., .5, 1.]
    assert scaled['test'].dur.tolist() == [2., 3., 4.]
    assert np.isfinite(scaled['train']).all().all()
    assert mappings['service'] == {'dns': 0}


def test_reject_unseen_categories_without_learning_from_test():
    train, test = records(), records()
    test.loc[0, 'state'] = 'NEW'
    with pytest.raises(ValueError, match='unseen state'):
        encode_and_scale(train, test)
