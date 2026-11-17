# Tabu-PIO / OC-SVM

Reconstruction of [Electronics 12(6), 1467](https://doi.org/10.3390/electronics12061467).
The trained UNSW classifier and usage instructions are in
[artifacts/unsw-table6](artifacts/unsw-table6/README.md).

From the repository root, preprocess the supplied CSV archive and train:

```sh
python -m detection.detectors.TabuPIO.preprocess \
  --archive OneDrive_2026-10-27.zip \
  --output detection/detectors/TabuPIO/results/unsw-paper-preprocessing

python -m detection.detectors.TabuPIO.train \
  --paper-data detection/detectors/TabuPIO/results/unsw-paper-preprocessing \
  --output detection/detectors/TabuPIO/results/unsw-paper-model
```

Use fresh output directories. Detailed outputs stay under ignored `results/`.

The paper reverses the official split filenames: its training subset contains
21,367 DNS records from the official testing file; testing contains 47,294
from the official training file. Training uses the published 15-feature subset,
min–max scaling and a benign-only RBF OC-SVM. Training fitness selects nu=0.001.
Category encoding and tuning details include documented reconstruction choices.

The saved model accepts unscaled numeric **flow features** in classifier-only
mode. It is not the default packet-collector model configured in `config.py`.
Optimizer discovery and the hybrid packet-length layer remain unverified.
