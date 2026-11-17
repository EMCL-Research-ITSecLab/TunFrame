"""Time-only/time-frequency ablations with native packet timestamps."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import dpkt
import numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from detection.detectors.TimeFrequencyDNS.features import from_wire,window_features,Lexicon,TIME_FEATURES
from detection.detectors.TimeFrequencyDNS.model import TimeFrequencyModel


def read_capture(path,seconds):
    flows=defaultdict(list);start=end=None;excluded=late=0
    with path.open('rb') as stream:
        try:reader=dpkt.pcap.Reader(stream)
        except ValueError:
            stream.seek(0);reader=dpkt.pcapng.Reader(stream)
        linktype=reader.datalink()
        for stamp,raw in reader:
            stamp=float(stamp)
            if start is None:start=stamp
            if seconds and stamp>start+seconds:
                end=start+seconds;break
            if end is not None and stamp<end:
                late+=1;continue
            end=stamp
            try:packet=from_wire(stamp,raw,linktype)
            except (dpkt.UnpackError,ValueError,IndexError,UnicodeError):
                excluded+=1;continue
            if packet:flows[packet.flow].append(packet)
    return flows,start,end,excluded,late


def extract(args,lexicon):
    datasets={size:[] for size in [300,600,1200]};coverage=[];seen=set()
    rng=np.random.default_rng(args.seed)
    for label,folder in [(0,'base_benign_dataset'),(1,'base_malicious_dataset')]:
        files=sorted(p for p in (args.data/folder).rglob('*') if p.suffix.lower() in ('.pcap','.pcapng'))
        if not files:raise ValueError(f'No captures under {args.data/folder}')
        for path in files:
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest in seen:
                coverage.append([str(path.relative_to(args.data)),label,digest,0,0,0,0,0,0,'duplicate_capture']);continue
            seen.add(digest)
            flows,origin,horizon,excluded,late=read_capture(path,args.seconds)
            counts=[]
            for size in datasets:
                available=[]
                if origin is not None and horizon-origin>=size:
                    for flow,packets in flows.items():
                        stamps=[p.timestamp for p in packets]
                        for end in np.arange(origin+size,horizon+1e-7,args.step):
                            left=bisect_left(stamps,end-size);right=bisect_left(stamps,end)
                            if right>left:available.append((flow,float(end),left,right))
                # Sample windows before feature calculation, never alter timestamps.
                if args.max_windows and len(available)>args.max_windows:
                    available=[available[i] for i in sorted(rng.choice(len(available),args.max_windows,replace=False))]
                for flow,end,left,right in available:
                    features=window_features(flows[flow][left:right],end-size,end,10.,lexicon)
                    datasets[size].append(dict(features=features,label=label,source=str(path.relative_to(args.data)),
                                               flow=flow,start=end-size,end=end))
                counts.append(len(available))
            duration=horizon-origin if origin is not None else 0
            coverage.append([str(path.relative_to(args.data)),label,digest,duration,
                             excluded,late,*counts,'prefix_limit' if args.seconds and duration>=args.seconds else 'capture_complete'])
            print(path.name,'300/600/1200s windows:',counts,flush=True)
    return datasets,coverage


def train(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh model directory')
    if args.step<=0 or args.seconds<0 or args.max_windows<0:raise ValueError('Invalid sampling settings')
    lexicon=Lexicon(args.dictionary.read_text(errors='strict').splitlines())
    datasets,coverage=extract(args,lexicon)
    args.output.mkdir(parents=True,exist_ok=True)
    results=args.output.parent/'results';results.mkdir(parents=True,exist_ok=True)
    rows=[];prediction_rows=[]
    for size,records in datasets.items():
        labels=np.asarray([r['label'] for r in records]);counts=np.bincount(labels,minlength=2)
        if min(counts)<2:
            print(f'{size}s skipped: insufficient full native windows for both classes',flush=True);continue
        if args.split=='window':
            train,test=train_test_split(np.arange(len(records)),test_size=.3,random_state=args.seed,stratify=labels)
        else:
            sources=[r['source'] for r in records]
            train,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=args.seed).split(records,labels,sources))
            if len(set(labels[train]))!=2 or len(set(labels[test]))!=2:
                raise ValueError(f'{size}s capture holdout lacks both classes; no metric fabricated')
        x=np.asarray([r['features'] for r in records]);models={}
        for variant in ['time','time_frequency']:
            values=x[:,:len(TIME_FEATURES)] if variant=='time' else x
            pipeline=make_pipeline(StandardScaler(),SVC(kernel='rbf',C=1.,gamma='scale'))
            pipeline.fit(values[train],labels[train]);models[variant]=pipeline
            scores=pipeline.decision_function(values[test]);predicted=(scores>=0).astype(int)
            tn,fp,fn,tp=confusion_matrix(labels[test],predicted,labels=[0,1]).ravel()
            auc=float(roc_auc_score(labels[test],scores))
            rows.append([size,variant,len(train),len(test),int(tp),int(fp),int(tn),int(fn),auc,args.split])
            for i,score,prediction in zip(test,scores,predicted):
                r=records[i];prediction_rows.append([size,variant,r['source'],*r['flow'],r['start'],r['end'],int(labels[i]),int(prediction),float(score)])
            print(f'{size}s {variant}: AUC={auc:.5f}, TP={tp}, FP={fp}, TN={tn}, FN={fn}',flush=True)
        artifact=dict(schema_version=1,window=size,fine=10.,step=args.step,words=lexicon.words,
                      time_features=TIME_FEATURES,models=models,scope='public_proxy_reconstruction',
                      svm_parameters=dict(kernel='rbf',C=1.,gamma='scale',scaling='train_only_StandardScaler'))
        model=TimeFrequencyModel(artifact);path=args.output/f'window-{size}.joblib';model.save(path)
        restored=TimeFrequencyModel.load(path)
        for variant in models:
            for i in test[:20]:assert restored.score(x[i],variant)==model.score(x[i],variant)
    if not rows:raise ValueError('No model trained: captures are too short; do not retime packets to fabricate windows')
    with (results/'metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['window_seconds','features','train','validation','tp','fp','tn','fn','auc','split']);writer.writerows(rows)
    with (results/'predictions.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['window_seconds','features','capture','client','server','start','end','label','prediction','margin']);writer.writerows(prediction_rows)
    with (results/'coverage.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['capture','label','sha256','observed_seconds','excluded','late','windows_300','windows_600','windows_1200','scope']);writer.writerows(coverage)
    summary=[f'Public proxy reconstruction; original month-long enterprise datasets are unavailable.',
        f'Native timestamp prefix={args.seconds}s (0=all), stride={args.step}s, fine bins=10s, max_windows/capture={args.max_windows}',
        f'Split={args.split}, 70/30; seed={args.seed}; saved model scores verified after reload.',
        'No timestamp compression, expansion or fabricated low-throughput labels.',
        '11 Table II descriptors expand to 13 scalar features: Q2 and Q4 each contain two quantities.',
        'Choices: per-query lexical means, special characters exclude dots, greedy longest English dictionary words (minimum length 2),',
        'mean DNS message size, upload/download byte ratio, request/response packet-count ratio, mean matched in-window RTT.',
        'No response/no RTT uses 0; empty fine bins use 0; FFT uses all unnormalized magnitudes including DC.',
        'Word dictionary: '+str(args.dictionary)+'; normalized word-list SHA256='+hashlib.sha256('\n'.join(lexicon.words).encode()).hexdigest(),
        'SVM kernel/C/gamma, scaling, stride, aggregation and lexicon are not fully specified in the paper.',
        'Random window split can share packets across overlapping train/validation windows; capture holdout is optional.',
        'Private Dataset2 and low-throughput mixed-background exfiltration experiment are not reproduced.',
        'Native decision unit is client/server IP pair/window; legacy domain alarms are projections.',
        'TCP stream/fragment reassembly not implemented; partial or unsupported packets are excluded.']
    (results/'training.txt').write_text('\n'.join(summary)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Public base_*_dataset PCAP folders (proxy only)')
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'model')
    parser.add_argument('--step',type=float,default=10.)
    parser.add_argument('--seconds',type=float,default=3600.)
    parser.add_argument('--max-windows',type=int,default=200)
    parser.add_argument('--split',choices=['window','capture'],default='window')
    parser.add_argument('--seed',type=int,default=10)
    train(parser.parse_args())
