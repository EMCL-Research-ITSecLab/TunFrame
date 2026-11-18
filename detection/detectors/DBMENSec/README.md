# DBM-ENSec

Reconstruction of [Li et al., Future Internet 2025](https://doi.org/10.3390/fi17070319).
The private dataset, feature extractor and trained weights are unavailable.
This is an experimental reconstruction, not an exact reproduction or a verified
match to the reported 99.92% private / 98.52% public accuracy.

```sh
pip install -r detection/detectors/DBMENSec/requirements.txt
python -m detection.detectors.DBMENSec.train features.csv
```

CSV input: the 23 numeric `FEATURE_NAMES` in `preprocess.py`, plus `label`
(0 benign, 1 tunnel). Optional `group` keeps captures/domains together across
both outer folds and inner validation splits. Default evaluation uses the
paper's five folds; outputs are CSVs and PDF/PNG figures under `results/`.
Final weights go under `model/`; neither directory is tracked.

Pipeline: train-fitted 0–255 scaling → padded 5×5 pixels → pretrained Bernoulli
DBM → residual projection → 64/128 Conv1D → max pooling → two attention layers
→ flattened features → LightGBM/XGBoost/RF/CatBoost hard votes.

Unspecified architecture choices are documented in `model.py`; `8×8` is
interpreted as two 8-unit hidden layers. Word extraction, injected volume and
several packet-field aggregations cannot be reconstructed exactly from the
paper. Table 2's duplicated row-11 name is corrected to
`Consecutive_Consonants_Ratio`. No feature values are invented.

Default voting follows Equation 5 (squared F1) and Section 4.3 (threshold 0.1).
`--weight-power 1 --threshold 0.01` selects the conflicting Algorithm 1 settings.
Weights use inner-validation weighted F1, avoiding Algorithm 1's test-label
leakage. Tree settings, residual connection, training limits and DBM inference
settings are declared reconstruction choices, not author-supplied parameters.

Framework selection: `DBMENSec`. Events must carry all features in the top-level
`dbm_ensec_features` mapping; ordinary collector events deliberately fail rather
than produce invalid benchmark scores. Positive packets raise domain alarms;
framework domain metrics differ from the paper's packet metrics. Override the
model location with `DBMENSEC_MODEL_DIR`.
