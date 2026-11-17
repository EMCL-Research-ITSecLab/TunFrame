"""Train the paper's final Char-CNN; supplied PCAPs are explicitly proxy data."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scapy.all import PcapReader,DNS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch

from detection.detectors.HighAccuracyCharCNN.model import CharCNN,encode


def read_data(directory,limit):
    names,labels=[],[]
    coverage=[]
    for label,folder in [(0,'base_benign_dataset'),(1,'base_malicious_dataset')]:
        paths=sorted(p for p in (directory/folder).rglob('*') if p.suffix.lower() in ('.pcap','.pcapng'))
        for path in paths:
            found=0
            with PcapReader(str(path)) as reader:
                for packet in reader:
                    if DNS not in packet or packet[DNS].qr or not packet[DNS].qd:
                        continue
                    name=packet[DNS].qd[0].qname.decode('latin1').rstrip('.')
                    if name:
                        names.append(name);labels.append(label);found+=1
                    if limit and found>=limit:break
            coverage.append((str(path.relative_to(directory)),label,found))
    if set(labels)!={0,1}:raise ValueError('Need both benign and tunnel examples')
    return names,np.asarray(labels),coverage


def train(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    if args.csv:
        with args.csv.open() as stream:rows=list(csv.reader(stream))
        if any(len(r)!=2 or r[1].strip() not in ('0','1') for r in rows):raise ValueError('Expected domain,label CSV without header')
        names=[r[0] for r in rows];labels=np.asarray([int(r[1]) for r in rows]);coverage=[]
    else:
        if args.scope!='public_proxy':raise ValueError('Public PCAPs cannot be identified as the private paper dataset')
        names,labels,coverage=read_data(args.data,args.per_capture)
    if args.scope=='paper_data' and (len(labels)!=55251 or int((labels==1).sum())!=6000):
        raise ValueError('Paper Table I requires 6,000 tunnel and 49,251 normal records')
    if args.scope=='paper_data' and args.per_class:
        raise ValueError('Use --per-class 0 to preserve the original paper cohort')
    # Optional cap is for bounded proxy runs; not the paper's 55,251 private rows.
    selected=[]
    for label in [0,1]:
        indices=np.flatnonzero(labels==label)
        selected.extend(rng.choice(indices,min(len(indices),args.per_class),replace=False) if args.per_class else indices)
    selected=np.asarray(selected)
    training,validation=train_test_split(selected,test_size=.1,random_state=args.seed,stratify=labels[selected])
    # Resampling is confined to training, never performed before splitting.
    classes=[training[labels[training]==label] for label in [0,1]]
    size=max(map(len,classes))
    sampled=np.concatenate([rng.choice(c,size,replace=len(c)<size) for c in classes])
    model=CharCNN();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss_function=torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train();losses=[]
        order=rng.permutation(sampled)
        for start in range(0,len(order),16):
            indices=order[start:start+16]
            if len(indices)<2:continue  # Batch normalization requires two training rows.
            optimizer.zero_grad()
            loss=loss_function(model(encode([names[i] for i in indices])),torch.tensor(labels[indices],dtype=torch.float32))
            loss.backward();optimizer.step();losses.append(loss.item())
        print(f'Epoch {epoch+1}: loss={np.mean(losses):.5f}',flush=True)
    probabilities=np.asarray(model.probabilities([names[i] for i in validation]))
    predicted=(probabilities>=.5).astype(int)
    model.save(args.output/'charcnn.pt')
    restored=CharCNN.load(args.output/'charcnn.pt')
    np.testing.assert_allclose(restored.probabilities([names[i] for i in validation]),probabilities,rtol=0,atol=0)
    results=args.output.parent/'results';results.mkdir(parents=True,exist_ok=True)
    tn,fp,fn,tp=confusion_matrix(labels[validation],predicted,labels=[0,1]).ravel()
    with (results/'predictions.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['source_row','name','label','probability'])
        writer.writerows((i,names[i],int(labels[i]),float(p)) for i,p in zip(validation,probabilities))
    with (results/'coverage.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['capture','label','queries']);writer.writerows(coverage)
    summary=[f'Paper-based Char-CNN reconstruction; data scope={args.scope}; seed={args.seed}; epochs={args.epochs}',
        f'Initial proxy-cohort cap per class={args.per_class}; subsequent balancing uses training rows only.',
        f'Train/validation={len(training)}/{len(validation)}; resampled training={len(sampled)}',
        f'Validation TP={tp}, FP={fp}, TN={tn}, FN={fn}; accuracy={(predicted==labels[validation]).mean():.6f}',
        f'Exact query overlap={len(set(names[i] for i in training)&set(names[i] for i in validation))}',
        'Paper Table III Char-CNN precision=.9954, recall=.9957; proxy scores are not directly comparable.',
        'Exact private dataset, alphabet, channel/dense widths, optimizer, epochs and sampling recipe unavailable.',
        'Choices: printable ASCII, preserve case, 256 conv channels, 1024 dense nodes, Adam .001, seed above.',
        'Implemented final Char-CNN only; recurrent language-model comparisons are not claimed.',
        'Saved checkpoint predictions verified exactly.']
    (results/'training.txt').write_text('\n'.join(summary)+'\n');print('\n'.join(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path,help='Public proxy dataset directory containing base_*_dataset folders')
    source.add_argument('--csv',type=Path,help='domain,label rows; no header')
    parser.add_argument('--scope',choices=['public_proxy','paper_data'],default='public_proxy')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'model')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--per-class',type=int,default=2000,help='0 uses every available row')
    parser.add_argument('--per-capture',type=int,default=500)
    train(parser.parse_args())
