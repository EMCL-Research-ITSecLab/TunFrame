"""Reproduce the paper's UNSW DNS subsets from the original published splits.

Run as a detector-local module. Input is the supplied CSV ZIP, not the packet
feature schema from the separate netrack experiment. No resampling is performed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURES = '''dur proto service state spkts dpkts sbytes dbytes rate sttl dttl
sload dload sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt
synack ackdat smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl
ct_dst_ltm ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login ct_ftp_cmd
ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports'''.split()
CATEGORICAL = ('proto', 'service', 'state')
# Table 3 and Figure A5 identify the paper's roles, opposite the source filenames.
SOURCE_SPLITS = {'train': 'testing', 'test': 'training'}
EXPECTED = {
    'train': {'DoS': 40, 'Exploits': 68, 'Fuzzers': 17, 'Generic': 18162,
              'Reconnaissance': 12, 'Normal': 3068},
    'test': {'DoS': 107, 'Exploits': 185, 'Fuzzers': 358, 'Generic': 39116,
             'Reconnaissance': 35, 'Normal': 7493},
}
TABU_OCSVM_INDICES = [1, 2, 4, 6, 8, 10, 14, 15, 17, 19, 22, 26, 29, 30, 37]


def dns_records(frame):
    required = ['id', *FEATURES, 'attack_cat', 'label']
    if list(frame.columns) != required:
        raise ValueError('Expected the published 45-column UNSW split schema and order')
    dns = frame.loc[frame.service == 'dns'].copy()
    if dns.empty or dns.isna().any().any():
        raise ValueError('Empty DNS subset or missing values')
    expected_label = (dns.attack_cat != 'Normal').astype(int)
    if not dns.label.isin([0, 1]).all() or not dns.label.eq(expected_label).all():
        raise ValueError('Binary label disagrees with attack_cat')
    # Keep distinct record IDs. Feature-vector deduplication destroys Table 3.
    duplicates = int(dns.duplicated().sum())
    dns = dns.drop_duplicates().copy()
    if dns.id.duplicated().any():
        raise ValueError('Conflicting records share the same source ID')
    return dns, duplicates


def encode_and_scale(train, test):
    """Freeze category mappings and Eq. 1 scaling on paper-training records."""
    encoded = {'train': train[FEATURES].copy(), 'test': test[FEATURES].copy()}
    mappings = {}
    for column in CATEGORICAL:
        mappings[column] = {value: i for i, value in enumerate(sorted(train[column].unique()))}
        for split, frame in encoded.items():
            unknown = set(frame[column]) - set(mappings[column])
            if unknown:
                raise ValueError(f'{split}: unseen {column} values: {sorted(unknown)}')
            frame[column] = frame[column].map(mappings[column])
    for frame in encoded.values():
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Non-finite or non-numeric model features')
    scaler = MinMaxScaler().fit(encoded['train'])
    normalized = {split: pd.DataFrame(scaler.transform(frame), columns=FEATURES, index=frame.index)
                  for split, frame in encoded.items()}
    return encoded, normalized, mappings, scaler


def prepare(archive, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; existing outputs are not overwritten')
    frames, sources, audits = {}, {}, {}
    with zipfile.ZipFile(archive) as bundle:
        for role, source_role in SOURCE_SPLITS.items():
            filename = f'UNSW_NB15_{source_role}-set.csv'
            candidates = [n for n in bundle.namelist() if Path(n).name == filename]
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one {filename}')
            data = bundle.read(candidates[0])  # zipfile verifies the member CRC.
            original = pd.read_csv(io.BytesIO(data), encoding='utf-8-sig')
            dns, removed = dns_records(original)
            counts = {k: int(v) for k, v in dns.attack_cat.value_counts().items()}
            if counts != EXPECTED[role]:
                raise ValueError(f'{role}: DNS class counts differ from Table 3: {counts}')
            frames[role] = dns
            sources[role] = dict(member=candidates[0], sha256=hashlib.sha256(data).hexdigest(),
                                 original_rows=len(original), paper_role=role)
            audits[role] = dict(rows=len(dns), classes=counts, exact_duplicate_rows_removed=removed,
                duplicate_feature_label_rows=int(dns.duplicated([*FEATURES, 'label']).sum()),
                first_ids=dns.id.head().tolist(), last_ids=dns.id.tail().tolist())
    # Additional independent correspondence with the image in Appendix A5.
    if frames['train'].id.head().tolist() != [407, 1041, 1275, 2136, 2427]:
        raise ValueError('Paper-training row order does not match Figure A5')
    encoded, normalized, mappings, scaler = encode_and_scale(frames['train'], frames['test'])
    output.mkdir(parents=True, exist_ok=True)
    for role, frame in frames.items():
        frame.to_csv(output/f'{role}-records.csv', index=False)
        for name, values in [('numeric', encoded[role]), ('normalized', normalized[role])]:
            values = values.copy()
            values['label'] = frame.label
            values.to_csv(output/f'{role}-{name}.csv', index=False)
    train_hashes = set(pd.util.hash_pandas_object(frames['train'][FEATURES+['label']], index=False))
    test_hashes = pd.util.hash_pandas_object(frames['test'][FEATURES+['label']], index=False)
    manifest = dict(schema_version=1, dataset='UNSW-NB15', status='table3_and_figure_A5_matched',
        exact_preprocessing_verified=False, sources=sources, subsets=audits, features=FEATURES,
        feature_count=len(FEATURES), categorical_mappings=mappings,
        normalization=dict(formula='(x - min) / (max - min)', fit_scope='paper_train_all_classes',
                           constant_feature_policy='scale denominator 1; training values become 0',
                           clip=False, minimum=scaler.data_min_.tolist(), maximum=scaler.data_max_.tolist()),
        test_rows_matching_train_feature_and_label=int(test_hashes.isin(train_hashes).sum()),
        table6_tabu_ocsvm=dict(one_based_indices=TABU_OCSVM_INDICES,
                              names=[FEATURES[i-1] for i in TABU_OCSVM_INDICES]),
        unresolved=['Paper does not specify the full categorical mapping or normalization fit scope.',
                    'Exact-row deduplication preserves Table 3; feature-vector deduplication does not.',
                    'UNSW attack label 1 follows the paper; it does not prove DNS exfiltration.',
                    'Flow byte counts are not individual packet lengths; no hybrid lengths are fabricated.',
                    'These normalized files must not be normalized again by the legacy training module.',
                    'The separate three-way validation training interface is not a reproduction of these published splits.'])
    (output/'preprocessing.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'features.txt').write_text(','.join(FEATURES)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.archive, args.output)
    print(json.dumps(report['subsets'], indent=2))
