# UNSW Table 6 classifier artifact

Selected trained model: `model-0.joblib`, nu=0.001, 15 published features.
The scaler is embedded. Supply **unscaled numeric flow features** in
`manifest.json` order; call `model.predict(X, variant="classifier_only")`.
Load the dictionary with joblib and retrieve its `model` member.

This is a published-subset classifier reconstruction. Optimizer discovery and
the packet-length hybrid remain unvalidated. This flow-feature model is not
installed as the packet collector adapter’s default model.

`manifest.json` records versions, hashes and assumptions; `preprocessing.json`
records category mappings and scaling. `results.json` retains every nu
candidate including unsuccessful ones. The original test predictions were
recomputed from this artifact and matched before committing.

Load only trusted joblib files. Training used scikit-learn 1.8.0; see the
manifest for the complete recorded package versions.

Verify the committed artifact without downloading the dataset:

```sh
python -m pytest detection/detectors/TabuPIO/tests/test_published_artifact.py -q
```

The tests check SHA-256, Table 6 feature indices, normalization bounds and
representative saved predictions, including real false positives and false
negatives. They also exercise the framework adapter with explicitly supplied
flow features. The full 47,294-row held-out test was recomputed before release:
TP=39,635, FP=3, TN=7,490, FN=166.

Example from the repository root:

```python
import joblib
from detection.detectors.TabuPIO.preprocess import FEATURES

artifact = joblib.load('detection/detectors/TabuPIO/artifacts/unsw-table6/model-0.joblib')
model = artifact['model']
# numeric_records is the unscaled, categorically encoded flow-feature DataFrame.
prediction = model.predict(numeric_records[FEATURES].to_numpy(), variant='classifier_only')
```

The artifact manifest records the assumptions behind the reproduction. The
paper's full optimizer and packet-length hybrid are not certified by these
classifier results. No flow byte count is substituted for packet length.
