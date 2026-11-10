# Tabu-PIO / OC-SVM detector

Detector-local implementation of Electronics 12(6), 1467 (2023). The supplied
paper is in `reference/electronics-12-01467.pdf`. The original framework and
other detectors are unchanged; there is no separate detector paper.

## Reproduce the UNSW preprocessing

From the repository root:

```sh
python -m detection.detectors.TabuPIO.preprocess \
  --archive OneDrive_2026-10-27.zip \
  --output detection/detectors/TabuPIO/results/unsw-paper-preprocessing
```

The supplied archive includes the original labeled flow CSVs, feature
metadata, ground truth and published training/testing partitions. The module
reads the published partitions directly from the ZIP and verifies their CRCs.
It selects `service == dns` and preserves their row order and record IDs.

**The paper’s Table 3 reverses the official filenames:**

| Paper role | Official source file | DNS rows | Attack | Normal |
|---|---|---:|---:|---:|
| Training | UNSW_NB15_testing-set.csv | 21,367 | 18,299 | 3,068 |
| Testing | UNSW_NB15_training-set.csv | 47,294 | 39,801 | 7,493 |

All individual attack-category counts match Table 3. The training subset’s
first IDs `[407, 1041, 1275, 2136, 2427]` and final IDs match Appendix Figure A5.
These are flow-feature records, not the packet-level netrack feature schema.
The original UNSW release distinguishes these partitions from the four raw
flow CSVs: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Preprocessing excludes `id`, `attack_cat` and `label` from the 42 model
features, retaining their source order. The existing binary label is checked
against `attack_cat`: Normal is 0; all attack categories are 1, following the
paper. This mapping is not independent evidence that every attack is a DNS
tunnel. Categories use sorted integer mappings, fitted on paper training.
Equation 1 min–max scaling is fitted on all paper-training records and applied
to testing without refitting or clipping. Constant training features become 0.

Outputs include original DNS rows (`*-records.csv`), encoded features and
labels (`*-numeric.csv`), scaled features and labels (`*-normalized.csv`),
`features.txt`, and `preprocessing.json`. The manifest records source hashes,
category mappings, scaling bounds, class counts, duplicates, and the Table 6
Tabu/OC-SVM indices mapped to feature names. Outputs require an empty directory.

### What is verified and what remains ambiguous

The subset membership/counts and feature schema are verified against Table 3
and Figure A5. Exact-row duplicate removal changes neither subset. Removing
duplicate feature/label vectors would instead remove 15,276 training rows and
36,250 test rows, contradicting Table 3. There are also 3,770 test rows whose
feature vector and binary label appear in training. These are reported and
preserved; no new random split or cross-split deduplication is introduced.

The paper does not fully specify categorical mappings, normalization fit scope,
or what constitutes a redundant record. Our mappings are consistent with the
visible Figure A5 example, but that does not prove the full original encoding.
The manifest therefore records `exact_preprocessing_verified: false`.

UNSW flow byte counts cannot be substituted for individual packet lengths.
The provided PCAP archive lacks its ZIP directory, but all ten stored capture
entries pass their data-descriptor CRC checks. Their offsets, SHA-256 hashes
and capture start times are recorded in the local results inventory. They are
a subset of the February capture; the separate raw `_1.csv` begins in January.
No arbitrary row-to-packet length join or zero packet length is fabricated.

## Detector and training

The default detector implements the combined decision rule: lengths in
[100,115], [145,185] or [220,340] immediately flag tunneling; otherwise the
selected-feature RBF OC-SVM decides. The length shortcut bypasses feature
validation and classifier inference. Positive records add their registered
domain to the original framework’s alarm set.

The framework’s original no-argument constructor loads the model specified in
`config.py`, normally `model/model-0.joblib`. A trained combined model is
required; missing artifacts fail explicitly. There are no new framework
configuration keys. Only load trusted joblib artifacts in a matching environment.

Train the classifier from the reproduced two-way split:

```sh
python -m detection.detectors.TabuPIO.train \
  --paper-data detection/detectors/TabuPIO/results/unsw-paper-preprocessing \
  --output detection/detectors/TabuPIO/results/unsw-paper-model
```

This mode fixes the 15 feature indices published for Tabu/OC-SVM in Table 6.
It reads the **unscaled numeric** CSVs, fits min–max bounds on all 21,367
paper-training rows, and fits OC-SVM on their 3,068 benign rows. It fits all four
published nu candidates and selects by Equation 9 on training predictions.
This resubstitution selection rule, benign-only OC-SVM fit and RBF `gamma=scale`
are explicit assumptions; the paper does not fully specify them. The selected
artifact is saved before test data is opened. All candidates are subsequently
evaluated on the untouched 47,294-row test subset, with predictions retained.
Saved-model predictions are checked against the in-memory model.

`model-0.joblib` contains the selected model, scaler and feature mask. Separate
candidate artifacts, `predictions.csv`, `results.json` and `manifest.json`
retain parameters, counts, assumptions, hashes and software versions. This is
one deterministic classifier fit per nu, not 20 stochastic optimizer runs.
The existing three-way interface remains available for exploratory optimizer
search; it is not used by this published-subset mode.

The completed run selected nu=0.001: TP=39,635, FP=3, TN=7,490, FN=166;
accuracy=99.6427%, recall=99.5829%, FPR=0.0400%, F1=99.7873%.
These results reproduce a classifier using the published subset; they do not
validate the optimizer’s discovery of that subset or the hybrid length layer.
The UNSW model requires flow features and must be called with
`variant='classifier_only'` and unscaled numeric inputs in the saved order.
It is deliberately not installed as the packet detector’s default artifact:
the collector’s packet fields cannot supply these flow features.

The optimizer follows binary Equations 7–8, which conflict with Algorithm 1’s
continuous references. Sigmoid transfer, one-bit Tabu neighbours, FIFO tenure
10 and aspiration are explicit assumptions. The continuous compass factor
0.09 is unused in that binary interpretation. The separate netrack experiment
has its own unresolved 17-versus-18-feature discrepancy. Training artifacts
remain unvalidated; thresholds are not adjusted to obtain an expected score.

Run detector-local tests with:

```sh
python -m pytest detection/detectors/TabuPIO/tests -q
```
