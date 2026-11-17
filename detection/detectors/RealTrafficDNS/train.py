"""Reconstruct the group CNN/RF models; private paper data is not substituted silently."""
import argparse
from collections import defaultdict,Counter
import csv
import hashlib
from pathlib import Path

import numpy as np
from scapy.all import PcapReader,DNS,IP,IPv6
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import torch

from detection.detectors.RealTrafficDNS.features import group_key,curate,statistics
from detection.detectors.RealTrafficDNS.model import PrefixCNN,RealTrafficModel,encode


def proxy_records(directory,limit):
    rows=[];seen=set()
    for label,folder in [(0,'base_benign_dataset'),(1,'base_malicious_dataset')]:
        for path in sorted((directory/folder).rglob('*')):
            if path.suffix.lower() not in ('.pcap','.pcapng'):continue
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest in seen:continue
            seen.add(digest);count=0
            with PcapReader(str(path)) as reader:
                for packet in reader:
                    if DNS not in packet or packet[DNS].qr or not packet[DNS].qd:continue
                    network=packet[IP] if IP in packet else packet[IPv6] if IPv6 in packet else None
                    if network is None:continue
                    name=packet[DNS].qd[0].qname.decode('latin1').rstrip('.')
                    rows.append(dict(qname=name,client_ip=network.src,timestamp=float(packet.time),label=label,
                                     category='benign_proxy' if label==0 else 'tunnel_proxy'))
                    count+=1
                    if limit and count>=limit:break
    return rows


def grouped(rows):
    values=defaultdict(list);labels=defaultdict(set);categories=defaultdict(set)
    for row in rows:
        key=group_key(row['qname'],row['client_ip'],row['timestamp'])
        if key is None:continue
        label=int(row['label'])
        if label not in (0,1):raise ValueError('Labels must be binary')
        values[key].append(row['qname']);labels[key].add(label);categories[key].add(row['category'])
    groups=[];dropped=Counter()
    for key,names in values.items():
        if len(labels[key])!=1 or len(categories[key])!=1:
            dropped['ambiguous_group']+=1;continue
        group=curate(names)
        if group is None:dropped['curation']+=1;continue
        group.update(key=key,label=next(iter(labels[key])),category=next(iter(categories[key])))
        groups.append(group)
    return groups,dict(dropped)


def split_groups(groups,rng,scope):
    train=[];test=[]
    # SLD-disjoint per-class holdout. The original paper specifies this for
    # malicious SLDs; extending it to benign domains is an explicit choice.
    domains_by_label=[set(g['key'][0] for g in groups if g['label']==label) for label in (0,1)]
    if domains_by_label[0]&domains_by_label[1]:raise ValueError('SLD has conflicting class labels')
    for label,domains in enumerate(domains_by_label):
        domains=sorted(domains)
        if len(domains)<2:raise ValueError('Need at least two SLDs per class for a disjoint holdout')
        chosen=set(rng.choice(domains,max(1,int(np.ceil(.1*len(domains)))),replace=False))
        for group in groups:
            if group['label']==label:(test if group['key'][0] in chosen else train).append(group)
    positive=[g for g in train if g['label']==1]
    if scope=='paper_data':
        negative=[]
        for category,size in [('normal_resolved',(len(positive)+1)//2),('non_tunnel',len(positive)//2)]:
            available=[g for g in train if g['category']==category]
            if len(available)<size:raise ValueError(f'Insufficient {category} groups for paper balancing')
            negative.extend(available[i] for i in rng.choice(len(available),size,replace=False))
    else:
        negative=[g for g in train if g['label']==0]
        size=min(len(positive),len(negative))
        positive=[positive[i] for i in rng.choice(len(positive),size,replace=False)]
        negative=[negative[i] for i in rng.choice(len(negative),size,replace=False)]
    return positive+negative,test


def train(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use an empty model directory')
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    if args.data:
        if args.scope!='public_proxy':raise ValueError('Public substitute traffic cannot be labelled paper_data')
        rows=proxy_records(args.data,args.per_capture)
    else:
        with args.csv.open() as stream:rows=list(csv.DictReader(stream))
    if args.scope=='paper_data':
        if set(r['category'] for r in rows)!={'normal_resolved','non_tunnel','dns_tunnel'}:
            raise ValueError('Paper scope requires all three analyst-labelled categories')
        if any(int(r['label'])!=int(r['category']=='dns_tunnel') for r in rows):
            raise ValueError('Labels disagree with analyst categories')
    groups,dropped=grouped(rows)
    training,test=split_groups(groups,rng,args.scope)
    prefixes=[];labels=[]
    for group in training:
        selected=group['prefixes']
        if args.max_prefixes and len(selected)>args.max_prefixes:
            selected=[selected[i] for i in rng.choice(len(selected),args.max_prefixes,replace=False)]
        prefixes.extend(selected);labels.extend([group['label']]*len(selected))
    cnn=PrefixCNN();optimizer=torch.optim.Adam(cnn.parameters(),lr=.001)
    loss_function=torch.nn.BCEWithLogitsLoss();labels=np.asarray(labels)
    for epoch in range(args.epochs):
        cnn.train();losses=[]
        order=rng.permutation(len(prefixes))
        for start in range(0,len(order),64):
            indices=order[start:start+64];optimizer.zero_grad()
            loss=loss_function(cnn(encode([prefixes[i] for i in indices])),torch.tensor(labels[indices],dtype=torch.float32))
            loss.backward();optimizer.step();losses.append(loss.item())
        print(f'Epoch {epoch+1}: CNN loss={np.mean(losses):.5f}',flush=True)
    # CNN is trained only on training groups. RF+CNN uses its training scores;
    # the paper does not describe out-of-fold stacking or release its code.
    x=np.asarray([statistics(g,cnn.probabilities(g['prefixes'])) for g in training])
    y=np.asarray([g['label'] for g in training])
    rf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=args.seed,n_jobs=4)
    combined=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=args.seed,n_jobs=4)
    rf.fit(np.delete(x,7,axis=1),y);combined.fit(x,y)
    model=RealTrafficModel(cnn,rf,combined);model.save(args.output)
    restored=RealTrafficModel.load(args.output)
    results=args.output.parent/'results';results.mkdir(parents=True,exist_ok=True)
    predictions=[]
    for group in test:
        scores=model.scores(group);saved=restored.scores(group)
        np.testing.assert_allclose(list(scores.values()),list(saved.values()),rtol=0,atol=0)
        predictions.append(dict(sld=group['key'][0],client=group['key'][1],day=group['key'][2],
            category=group['category'],label=group['label'],size=len(group['prefixes']),**scores))
    with (results/'predictions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    summary=[f'Paper-based reconstruction; scope={args.scope}; seed={args.seed}; epochs={args.epochs}',
        f'Queries={len(rows)}; curated groups={len(groups)}; dropped={dropped}',
        f'Train/test groups={len(training)}/{len(test)}; CNN training prefixes={len(prefixes)}',
        f'Groups unused after training balancing={len(groups)-len(training)-len(test)}',
        f'Max CNN training prefixes per group={args.max_prefixes}; SLD-disjoint split; held-out SLD overlap=0',
        'Threshold=.5 (strict majority for CNN); RF real-day threshold=.8 is a separate paper experiment.']
    truth=[g['label'] for g in test]
    for variant in ['rf','cnn','rf_cnn']:
        predicted=[int(p[variant]>.5 if variant=='cnn' else p[variant]>=.5) for p in predictions]
        tn,fp,fn,tp=confusion_matrix(truth,predicted,labels=[0,1]).ravel()
        summary.append(f'{variant}: TP={tp}, FP={fp}, TN={tn}, FN={fn}; accuracy={np.mean(np.asarray(predicted)==truth):.6f}')
    summary += ['Private Infoblox analyst labels and trained weights unavailable; proxy scores do not reproduce Table III.',
        'Unspecified choices: ASCII alphabet, case preservation, length=253, embedding=32, max pooling,',
        'ReLU dense activation, Adam .001, epoch count, and extending SLD holdout to benign domains.',
        'Normalized entropy=H/log2(length); repeat ratio=(length-unique)/unique; exact formulas not given.',
        'Final-day stream groups are partial at benchmark cutoff; group majority ties are benign.',
        'All saved model scores verified after reload.']
    (results/'training.txt').write_text('\n'.join(summary)+'\n');print('\n'.join(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path,help='Proxy base_*_dataset PCAP folders')
    source.add_argument('--csv',type=Path,help='qname,client_ip,timestamp,label,category CSV')
    parser.add_argument('--scope',choices=['public_proxy','paper_data'],default='public_proxy')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'model')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--per-capture',type=int,default=500)
    parser.add_argument('--max-prefixes',type=int,default=0,help='Per-group training cap; 0 keeps all')
    train(parser.parse_args())
