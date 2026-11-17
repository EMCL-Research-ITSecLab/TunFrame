"""Verify the committed classifier without requiring the original dataset."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from detection.detectors.TabuPIO.detector import TabuPIO

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts/unsw-table6'


def test_published_artifact_hash_schema_scaling_and_predictions():
    fixture = json.loads(Path(__file__).with_name('unsw_artifact_cases.json').read_text())
    manifest = json.loads((ARTIFACT / 'manifest.json').read_text())
    preprocessing = json.loads((ARTIFACT / 'preprocessing.json').read_text())
    path = ARTIFACT / 'model-0.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['model_sha256'] == manifest['model_sha256']
    model = joblib.load(path)['model']
    assert model.feature_names == fixture['features'] == manifest['features']
    assert (np.flatnonzero(model.selected) + 1).tolist() == manifest['selected_indices_one_based']
    assert model.nu == manifest['selected_nu'] == .001
    np.testing.assert_allclose(model.scaler.data_min_, np.array(preprocessing['normalization']['minimum'])[model.selected])
    np.testing.assert_allclose(model.scaler.data_max_, np.array(preprocessing['normalization']['maximum'])[model.selected])
    cases = fixture['cases']
    prediction = model.predict([c['features'] for c in cases], variant='classifier_only')
    assert prediction.tolist() == [c['prediction'] for c in cases]
    # Include both correct decisions and the actual FP/FN cases in the release.
    assert {(c['label'], c['prediction']) for c in cases} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_artifact_adapter_uses_enriched_flow_features():
    fixture = json.loads(Path(__file__).with_name('unsw_artifact_cases.json').read_text())
    detector = TabuPIO(dict(model=ARTIFACT / 'model-0.joblib', variant='classifier_only'))
    for i, case in enumerate(fixture['cases']):
        domain = f'case{i}.com'
        detector.detect({'dns': {'qname': 'query.' + domain},
                         'tabu_pio': {'features': dict(zip(fixture['features'], case['features']))}})
        assert (domain in detector.alarms) == bool(case['prediction'])
    with pytest.raises(ValueError, match='length layer'):
        TabuPIO(dict(model=ARTIFACT / 'model-0.joblib', variant='combined', length_layer='dns'))
