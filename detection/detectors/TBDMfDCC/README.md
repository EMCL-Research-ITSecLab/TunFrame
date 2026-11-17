# TBDMfDCC

Adapter for [Transformer-Based Detection Method for DNS Covert
Channel](https://doi.org/10.22323/1.434.0011). The default uses the supplied
DistilBERT checkpoint unchanged. `author_training.py` preserves their original
DNS training script; it is not imported by the framework.

```sh
pip install -r detection/detectors/TBDMfDCC/requirements.txt
python -m detection.detectors.TBDMfDCC.model --archive TBDMfDCC.zip
```

The extraction command requires an empty `model/` directory. Inference runs
offline on CPU with the supplied WordPiece vocabulary, complete DNS names,
attention masks and argmax (Normal=0, Covert=1). As in their test code, inputs
use 256 tokens; explicit truncation bounds long names. The adapter restores
the terminal DNS dot present in the authors’ examples.
DNS payload labels are parsed as DNS names, including `/` and `@`, when
recording registered-domain alarms. The original framework's shared parsing
and validation may still exclude such observations before the adapter sees them.

The supplied checkpoint agrees with only 18/32 inspected saved DNS predictions.
Its configuration retains sentiment labels and the source script never saves
fine-tuned weights. This discrepancy is documented, not silently corrected.
The default still evaluates the original supplied checkpoint.

Optional improved variant (separate weights, original untouched):

```sh
python -m detection.detectors.TBDMfDCC.train --archive TBDMfDCC.zip
```

This uses the authors’ default 2,000-row dataset and stratified 60/20/20 split
(seed 10), but fixes the schedule to cover two epochs, uses 64 tokens in every
phase, and saves the fine-tuned model under `model-improved/`. AdamW uses 2e-5;
batch size 16 bounds CPU memory. Training seed 10 is explicit. Set `VARIANT`
in `config.py` to `improved` to benchmark it; default is `authors`.

Both variants are compared on identical held-out rows in ignored
`results/comparison.csv` and `comparison.txt`. Use `--dataset Dataset.csv`
with a fresh `--output` for the larger release (73,919 rows versus 73,965
reported in the paper). The small run validates integration, not the paper’s
full-dataset result. Local weights and generated outputs are ignored by Git.

The local 400-row comparison gave authors: TP=0, FP=92, TN=208, FN=100;
improved: TP=100, FP=1, TN=299, FN=0. The source split has two DNS names
shared by training and testing. These small-dataset results do not establish
generalization to new domains or reproduce the paper's full experiment.
