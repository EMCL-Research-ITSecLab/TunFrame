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
