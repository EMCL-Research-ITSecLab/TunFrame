# Real-traffic DNS tunneling

Reconstruction of [Lambion et al., IEEE Big Data 2020](https://doi.org/10.1109/BigData50022.2020.9378418).
Groups use registered domain + client IP + UTC day. Curation removes `.arpa`,
the longest common character suffix, singleton and empty-prefix groups.

Models: parallel CNN kernels 2/3/4/5 with 256 filters each and two 1,024-node
hidden layers; 100-tree entropy RF; RF+CNN using all 12 Table II features.
Plain RF excludes the CNN feature. CNN decisions use strict majority voting.

```sh
pip install -r detection/detectors/RealTrafficDNS/requirements.txt
python -m detection.detectors.RealTrafficDNS.train \
  --data path/to/public/datasets --per-capture 1000 --max-prefixes 100 --epochs 3
```

For analyst-labelled data use `--csv` with
`qname,client_ip,timestamp,label,category`. Categories must be
`normal_resolved`, `non_tunnel`, `dns_tunnel` when selecting `--scope paper_data`.
Public traffic is explicitly proxy data and cannot replace the private
Infoblox analyst labels, particularly benign non-tunnel examples.

`config.py` selects `rf`, `cnn` or `rf_cnn`. Default threshold is 0.5; the
paper’s separately tuned live RF threshold is 0.8. Groups finalize at day
boundaries and experiment cutoff; the first and final day are conservatively
marked partial. Partial-day decisions and exclusions are
recorded. Local weights and reports are ignored by Git.

Alphabet, entropy/repetition formulas, pooling, embedding width, optimizer and
training details include documented choices. The split is SLD-disjoint for
both classes; the paper explicitly specifies this for malicious SLDs only.
RF+CNN uses CNN probabilities fitted on training groups; neither model fits
held-out data. Saved scores are verified after reload. Proxy results do not
reproduce Table III or the paper’s real-traffic false-positive rates.
