# Encoding and behavioral feature fusion

Reconstruction of [Tu et al., Computers & Security 132 (2023),
103357](https://doi.org/10.1016/j.cose.2023.103357).

DNS query/response pairs feed a character attention encoder (two outputs) and
six statistics into XGBoost. Pairing uses client/server endpoints, protocol,
transaction ID and questions; unmatched responses and expired queries never
become benign decisions. The adapter records pair coverage in benchmark output.

```sh
pip install -r detection/detectors/EncodingBehaviorFusion/requirements.txt
python -m detection.detectors.EncodingBehaviorFusion.train \
  --data path/to/authors-release/datasets --max-pairs 1000
```

Use the [paper authors’ release](https://github.com/Utah1A/DNS-tunnelling-detection-by-fusing-encoding-feature-and-behavioral-feature),
revision `5d71ddeddf84f8a7c7a024ce40a92891235bc3ce`. The linked Chen LSTM
repository supplies its malicious traffic; the authors’ release adds benign
PCAPs. Its current Git tree has no generalization dataset despite the README.

Algorithm 1 preprocessing, Table 3's six features and Table 4's XGBoost values
are implemented. The diagram’s four-dimensional inputs and eight heads use
separate four-dimensional Q/K/V projections. The paper does not specify enough
encoder details to reproduce its training exactly: embedding, masked pooling,
feed-forward width, loss, optimizer, padding, epoch count and split seed are
documented choices. Its RNN prose conflicts with its attention-only diagram;
this implementation follows the diagram and is not an author implementation.

The six statistics are DNS query/response lengths minus the 12-byte header,
mean answer TTL, answer/question count ratio, delay in seconds and DNS-message
upload/download ratio. These size conventions and zero TTL for empty answers
are assumptions. TCP stream reassembly and lossless collector answer text are
not guaranteed; unsupported observations are counted as exclusions.

Training balances classes, uses the paper’s 180:20:100 split proportions,
supervises the encoder first, then fits XGBoost on its two outputs plus six
statistics. Exact duplicate capture files are skipped. Default sampling caps
each capture at 5,000 pairs; `--max-pairs 0` reads all pairs. This is a seeded
pair split, not independent-capture generalization. No author weights or exact
split IDs were released. Local weights and detailed coverage/predictions stay
in ignored `model/` and `results/`; `config.py` selects the benchmark model.

The local 10-epoch run used 8,000 balanced pairs; held-out counts were
TP=1,318, FP=4, TN=1,329, FN=16 (99.25% accuracy). Saved-model reload and
collector/PCAP parity checks passed. There are 323 distinct encoded contents
shared by training and testing; these results do not verify the paper's scores.
