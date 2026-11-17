# High-accuracy Char-CNN

Reconstruction of the final model in [Chen & Li, CIS 2020](https://doi.org/10.1109/CIS52066.2020.00086).
Six convolutions use widths 7,7,3,3,3,3; pooling follows layers 1,2,6.
Inputs are padded/truncated to 1,014 characters. Dense stages use batch
normalization and dropout 0.5; the output is sigmoid.

```sh
pip install -r detection/detectors/HighAccuracyCharCNN/requirements.txt
python -m detection.detectors.HighAccuracyCharCNN.train \
  --data path/to/public/datasets --per-class 1000 --per-capture 300 --epochs 3
```

Alternatively use `--csv` with headerless `domain,label` rows. The proxy budget
caps each class before the 90/10 stratified split; subsequent training balancing
never resamples validation. `--scope paper_data --per-class 0` requires the
paper's Table I class counts. Local weights and
reports stay under ignored `model/` and `results/`. `config.py` selects the
model and threshold; the original framework discovers it normally.

The private 55,251-row ISP/tool dataset and weights are unavailable. Our proxy
run classified all 200 validation queries correctly, but shares 57 exact query
strings with training. This is integration validation, not reproduction of
the paper’s .9954 precision and .9957 recall.

Unspecified choices: printable ASCII preserving case, 256 filters, 1,024 dense
units, Adam .001 and training seed/epochs. Widths follow the small Char-CNN
cited by the paper. The LSTM/GRU language-model comparisons are not implemented
or claimed. Saved-model probabilities and query-only adapter behavior agree.
