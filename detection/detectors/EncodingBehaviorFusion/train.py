"""Train a documented reconstruction on the paper authors' released PCAPs."""
import argparse
import csv
import hashlib
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from scapy.all import PcapReader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch

from detection.detectors.EncodingBehaviorFusion.features import Pairer, from_packet, STATISTICS
from detection.detectors.EncodingBehaviorFusion.model import AttentionEncoder, FusionModel, classifier, tensors


def extract(root, limit):
    records, coverage, hashes = [], [], set()
    for label, folder in [(0, 'base_benign_dataset'), (1, 'base_malicious_dataset')]:
        files = sorted(p for p in (root/folder).rglob('*') if p.suffix.lower() in ('.pcap', '.pcapng'))
        if not files:
            raise ValueError(f'No PCAPs under {root/folder}')
        for path in files:
            with path.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest in hashes:
                coverage.append([str(path.relative_to(root)), label, digest, 0, 0, 0, 'duplicate_capture'])
                continue
            hashes.add(digest)
            pairer = Pairer()
            read = excluded = pairs = 0
            with PcapReader(str(path)) as reader:
                for packet in reader:
                    read += 1
                    try:
                        observation = from_packet(packet)
                        pair = pairer.add(observation) if observation else None
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError):
                        excluded += 1
                        continue
                    if pair:
                        pair.update(label=label, source=str(path.relative_to(root)))
                        records.append(pair)
                        pairs += 1
                        if limit and pairs >= limit:
                            break
            coverage.append([str(path.relative_to(root)), label, digest, read, excluded, pairs,
                             'prefix_limit' if limit and pairs >= limit else 'capture_complete'])
            print(f'{path.name}: {pairs} pairs, {excluded} exclusions', flush=True)
    return records, coverage


def chunks(indices, size=32):
    for offset in range(0, len(indices), size):
        yield indices[offset:offset+size]


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory')
    if args.epochs < 1 or args.max_pairs < 0 or args.length < 1:
        raise ValueError('Invalid epochs, pair limit or sequence length')
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    start = perf_counter()
    records, coverage = extract(args.data, args.max_pairs)
    rng = np.random.default_rng(args.seed)
    classes = [[i for i,r in enumerate(records) if r['label']==label] for label in (0,1)]
    count = min(map(len, classes))
    if count < 30:
        raise ValueError('Too few complete pairs for both classes')
    # Paper Table 5 balances classes and uses 180K/20K/100K per class.
    chosen = np.concatenate([rng.choice(indices, count, replace=False) for indices in classes])
    rng.shuffle(chosen)
    selected = [records[i] for i in chosen]
    y = np.asarray([r['label'] for r in selected])
    fitting, test = train_test_split(np.arange(len(selected)), test_size=1/3, random_state=args.seed, stratify=y)
    training, validation = train_test_split(fitting, test_size=.1, random_state=args.seed, stratify=y[fitting])
    encoder = AttentionEncoder(args.length)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()
    tokens, lengths = tensors([r['content'] for r in selected], args.length)
    labels = torch.tensor(y)
    for epoch in range(args.epochs):
        encoder.train()
        losses = []
        for indices in chunks(rng.permutation(training)):
            optimizer.zero_grad()
            loss = loss_function(encoder(tokens[indices], lengths[indices]), labels[indices])
            loss.backward();optimizer.step();losses.append(loss.item())
        encoder.eval()
        with torch.inference_mode():
            prediction = torch.cat([encoder(tokens[i], lengths[i]).argmax(-1) for i in chunks(validation)]).numpy()
        print(f'Epoch {epoch+1}: loss={np.mean(losses):.5f}, validation accuracy={(prediction==y[validation]).mean():.4f}', flush=True)
    fusion = FusionModel(encoder, classifier(args.seed))
    def features(indices):
        return np.concatenate([fusion.features([selected[i] for i in batch]) for batch in chunks(indices)])
    fusion.classifier.fit(features(training), y[training])
    validation_predictions = fusion.classifier.predict(features(validation))
    predicted = fusion.classifier.predict(features(test))
    fusion.save(args.output)
    reloaded = FusionModel.load(args.output)
    restored = np.concatenate([reloaded.predict([selected[i] for i in batch]) for batch in chunks(test)])
    assert np.array_equal(predicted, restored)
    results = args.output.parent/'results'
    results.mkdir(parents=True, exist_ok=True)
    with (results/'coverage.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['capture','label','sha256','packets_read','excluded','pairs','scope']);writer.writerows(coverage)
    with (results/'predictions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row','source','qname','label','prediction'])
        writer.writerows((i,selected[i]['source'],selected[i]['qname'],int(y[i]),int(p)) for i,p in zip(test,predicted))
    tn, fp, fn, tp = confusion_matrix(y[test], predicted, labels=[0,1]).ravel()
    overlap = len(set(selected[i]['content'] for i in training) & set(selected[i]['content'] for i in test))
    summary = [f'Paper-based reconstruction; seed={args.seed}; epochs={args.epochs}; max_pairs_per_capture={args.max_pairs}',
        f'Balanced pairs: {len(selected)}; train/validation/test={len(training)}/{len(validation)}/{len(test)}',
        f'Features: two learned encoder outputs + {", ".join(STATISTICS)}',
        f'Test TP={tp}, FP={fp}, TN={tn}, FN={fn}; accuracy={(predicted==y[test]).mean():.6f}',
        f'Fusion validation accuracy={(validation_predictions==y[validation]).mean():.6f}',
        f'Train/test distinct content overlap={overlap}; source-capture-independent evaluation=False',
        f'Truncated contents={sum(len(r["content"])>args.length for r in selected)}; max_length={args.length}',
        f'Saved model predictions verified; elapsed_seconds={perf_counter()-start:.1f}',
        'Unspecified paper details reconstructed: alphabet order/unknown handling, padding/truncation,',
        'per-head dimension=4, masked mean pooling, feed-forward width=32, supervised cross-entropy,',
        'Adam 1e-3, batch=32, epoch count and split seed.',
        'DNS lengths minus 12-byte header for payload; DNS lengths for upload/download ratio;',
        'TTL=mean answer-section TTL (0 without answers), ratio=answer RR count/question count.',
        'TCP stream reassembly not implemented; incomplete TCP DNS excluded.',
        'No published pretrained model or exact split IDs available; scores do not certify exact reproduction.']
    (results/'training.txt').write_text('\n'.join(summary)+'\n')
    print('\n'.join(summary), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Authors release datasets/ directory')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'model')
    parser.add_argument('--max-pairs', type=int, default=5000, help='Per capture prefix; 0 means all pairs')
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)
    train(parser.parse_args())
