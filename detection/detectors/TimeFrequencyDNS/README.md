# Time-frequency DNS tunnel detection

Reconstruction of [Liu et al., ICC 2023](https://doi.org/10.1109/ICC45041.2023.10279472).
Traffic is grouped by client/server IP, deliberately ignoring changing ports.
300/600/1,200-second windows combine Table II statistics with FFT magnitudes
of mean DNS packet sizes in 10-second bins, then classify with an SVM.

```sh
pip install -r detection/detectors/TimeFrequencyDNS/requirements.txt
python -m detection.detectors.TimeFrequencyDNS.train \
  --data path/to/public/datasets --dictionary /usr/share/dict/words
```

The local run uses proxy PCAPs with unchanged timestamps, up to one hour per
capture and 200 sampled windows per capture/window size. `--seconds 0` and
`--max-windows 0` remove these limits. Both time-only and combined-feature
models are saved; `config.py` chooses the model/window and feature variant.
Weights include the dictionary, so inference needs no system word-list lookup.
Proxy PCAPs come from [Tu et al.'s separate dataset release](https://github.com/Utah1A/DNS-tunnelling-detection-by-fusing-encoding-feature-and-behavioral-feature),
revision `5d71ddeddf84f8a7c7a024ce40a92891235bc3ce`.

The paper’s private month-long low-throughput dataset and enterprise test day
are unavailable. Public tunnel PCAPs do not reproduce its mixed-background
exfiltration every 20–40 minutes. Random 70/30 window splitting follows the
reported experiment but can share packets through overlapping windows;
`--split capture` provides a separate capture-holdout check.

Unspecified choices include 10-second classification stride, lexical means,
greedy dictionary words, DNS-message sizes, byte upload/download ratio,
zero values for unavailable ratios/RTT, empty bins, full FFT including DC,
and training-only scaling with RBF SVM C=1. The 11 feature descriptions expand
to 13 scalars because Q2 and Q4 each describe two quantities.

The adapter emits only full windows, never pads incomplete observation periods,
and advances on observed timestamps. A short benchmark can therefore produce
no decisions; quiet-window completion requires an explicit `advance()` call.
Native alarms concern IP pairs/windows; the legacy API projects them onto
observed query domains, which is not domain attribution. Late, malformed and
capacity-dropped packets are counted. TCP stream/fragment reassembly is absent.
Detailed outputs stay in ignored `results/`; no exact paper reproduction is claimed.
