"""Train and evaluate the unvalidated binary Tabu-PIO reconstruction.

Explicit train/validation/test CSVs prevent test-driven tuning. No artifact
produced here is automatically admitted as a reproduction of the published model.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from detection.detectors.TabuPIO.model import (ASSUMPTIONS, HybridModel, NU_CANDIDATES, LENGTH_RANGES,
    SearchSettings, classification_metrics, fitness, select_features)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_splits(paths, features, group_column, length_column):
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for name, frame in frames.items():
        required = set(features) | {'label', group_column}
        if length_column:
            required.add(length_column)
        if not required <= set(frame.columns):
            raise ValueError(f'{name}: missing columns {required - set(frame.columns)}')
        if frame.empty or frame[group_column].isna().any():
            raise ValueError(f'{name}: empty data or missing group identifiers')
        if not set(frame.label.unique()) <= {0, 1} or frame.label.nunique() != 2:
            raise ValueError(f'{name}: require both binary classes (1=tunnel)')
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: non-finite features')
    # A qname grouping is weaker than capture/session grouping, but is explicit.
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlap = set(frames[a][group_column]) & set(frames[b][group_column])
        if overlap:
            raise ValueError(f'{a}/{b}: {len(overlap)} overlapping groups')
        hashes_a = set(pd.util.hash_pandas_object(frames[a][features], index=False))
        hashes_b = set(pd.util.hash_pandas_object(frames[b][features], index=False))
        if hashes_a & hashes_b:
            raise ValueError(f'{a}/{b}: duplicate feature vectors; deduplicate before splitting')
    return frames


def run(args):
    if args.paper_data:
        if args.runs not in (None, 1):
            raise ValueError('Published-subset OC-SVM fitting is deterministic; use --runs 1 or omit it')
        return run_paper(args.paper_data, args.output)
    if args.runs is None:
        args.runs = 20
    if not all([args.train, args.validation, args.test, args.features]):
        raise ValueError('Provide --paper-data or all three splits and --features')
    features = [v.strip() for v in args.features.split(',') if v.strip()]
    if not features or len(features) != len(set(features)):
        raise ValueError('Provide unique, ordered feature names')
    if set(features) & {'label', args.group_column, args.length_column}:
        raise ValueError('Labels, grouping and length metadata cannot be model features')
    if args.runs < 1 or (args.length_column is None) != (args.length_layer == 'unresolved'):
        raise ValueError('Provide positive runs and both length column/layer, or neither')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a fresh output directory; existing evidence is never overwritten')
    paths = dict(train=args.train, validation=args.validation, test=args.test)
    frames = load_splits(paths, features, args.group_column, args.length_column)
    args.output.mkdir(parents=True, exist_ok=True)
    train, validation, test = (frames[s] for s in paths)
    x_benign = train.loc[train.label == 0, features].to_numpy(float)
    x_validation = validation[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_validation, y_test = validation.label.to_numpy(), test.label.to_numpy()
    lengths = test[args.length_column].to_numpy(float) if args.length_column else None
    variants = ['classifier_only'] if lengths is None else ['length_only', 'classifier_only', 'combined']
    manifest = dict(schema_version=1, experiment=args.name,
        status='reconstruction_unvalidated', admission=False, assumptions=ASSUMPTIONS,
        purpose=args.purpose, features=features, length_layer=args.length_layer, length_ranges=LENGTH_RANGES,
        group_column=args.group_column, evaluation_unit='record',
        seeds=list(range(args.seed, args.seed + args.runs)),
        split_counts={name: len(frame) for name, frame in frames.items()},
        input_sha256={name: sha256(path) for name, path in paths.items()},
        software={n: importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib','pandas']},
        python=platform.python_version(), hardware=platform.machine(),
        unavailable_variants=[] if lengths is not None else ['length_only','combined'],
        inference_timing='one batch, excluding file IO; includes scaling and shortcut routing',
        search_settings=dict(population=args.population, iterations=args.iterations,
                             local_iterations=args.local_iterations, tabu_tenure=10),
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[3])): sha256(p)
                       for p in [Path(__file__).resolve(), Path(__file__).resolve().parent/'model.py']})
    write_json(args.output/'manifest.json', manifest)
    results, runtime_rows = [], []
    with (args.output/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed','variant','row','group','label','prediction','packet_length'])
        writer.writeheader()
        for seed in manifest['seeds']:
            start = perf_counter()
            scaler = MinMaxScaler().fit(x_benign)
            scaled_train, scaled_validation = scaler.transform(x_benign), scaler.transform(x_validation)
            choices = {}
            def evaluate(mask):
                candidates = []
                for nu in NU_CANDIDATES:
                    classifier = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
                    classifier.fit(scaled_train[:, mask])
                    prediction = (classifier.predict(scaled_validation[:, mask]) == -1).astype(int)
                    score = fitness(int(mask.sum()), len(mask), classification_metrics(y_validation, prediction))
                    candidates.append((score, nu))
                choices[tuple(mask)] = min(candidates)
                return choices[tuple(mask)][0]
            settings = SearchSettings(args.population, args.iterations, args.local_iterations, seed=seed)
            mask, history = select_features(evaluate, len(features), settings)
            nu = choices[tuple(mask)][1]
            model = HybridModel(features, mask, nu, args.length_layer).fit(x_benign)
            training_seconds = perf_counter() - start
            artifact = dict(schema_version=1, model=model, seed=seed, status=manifest['status'],
                            admission=False, assumptions=ASSUMPTIONS, input_sha256=manifest['input_sha256'])
            model_path = args.output / f'model-{seed}.joblib'
            joblib.dump(artifact, model_path)
            write_json(args.output/f'search-{seed}.json', history)
            for variant in variants:
                start = perf_counter()
                prediction = model.predict(x_test, lengths, variant)
                elapsed = perf_counter() - start
                metrics = classification_metrics(y_test, prediction)
                results.append(dict(detector='Tabu-PIO / OC-SVM', variant=variant, seed=seed,
                    scenario=args.name, unit='record', status=manifest['status'], purpose=args.purpose,
                    training_seconds=training_seconds, inference_seconds=elapsed,
                    us_per_record=1e6*elapsed/len(test), selected_features=int(mask.sum()), nu=nu,
                    model_sha256=sha256(model_path), **metrics))
                for i, predicted in enumerate(prediction):
                    writer.writerow(dict(seed=seed, variant=variant, row=i,
                        group=test.iloc[i][args.group_column], label=int(y_test[i]),
                        prediction=int(predicted), packet_length='' if lengths is None else lengths[i]))
                for size in sorted({min(len(test), n) for n in (100, 500, 1000, 5000, len(test))}):
                    for repeat in range(3):
                        start = perf_counter()
                        model.predict(x_test[:size], None if lengths is None else lengths[:size], variant)
                        runtime_rows.append(dict(seed=seed, variant=variant, size=size, repeat=repeat,
                                                 seconds=perf_counter()-start))
            print(f'Seed {seed}: {mask.sum()} features, nu={nu:g}, training={training_seconds:.2f}s', flush=True)
    write_json(args.output/'results.json', dict(schema_version=1, manifest=manifest, results=results))
    pd.DataFrame(runtime_rows).to_csv(args.output/'runtime.csv', index=False)


def run_paper(directory, output):
    """Fit Table 6's published feature subset; do not rerun ambiguous selection.

    Nu selection uses training resubstitution fitness, an explicit assumption.
    All candidates are fixed before the held-out paper test set is read.
    """
    from detection.detectors.TabuPIO.preprocess import FEATURES, EXPECTED, TABU_OCSVM_INDICES
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    prep_path=directory/'preprocessing.json'
    prep=json.loads(prep_path.read_text())
    if prep['features'] != FEATURES or prep['feature_count'] != 42:
        raise ValueError('Expected the reproduced 42-feature UNSW schema')
    train_path=directory/'train-numeric.csv'
    test_path=directory/'test-numeric.csv'
    train=pd.read_csv(train_path)
    def check(frame, role):
        if list(frame.columns) != FEATURES+['label'] or not frame.label.isin([0,1]).all():
            raise ValueError(f'{role}: invalid schema or labels')
        if len(frame)!=sum(EXPECTED[role].values()) or int((frame.label==0).sum())!=EXPECTED[role]['Normal']:
            raise ValueError(f'{role}: counts differ from Table 3')
        if not np.isfinite(frame[FEATURES].to_numpy(float)).all():
            raise ValueError(f'{role}: non-finite features')
    check(train,'train')
    x=train[FEATURES].to_numpy(float);y=train.label.to_numpy()
    selected=np.zeros(len(FEATURES),dtype=bool)
    selected[np.array(TABU_OCSVM_INDICES)-1]=True
    output.mkdir(parents=True,exist_ok=True)
    assumptions=[
        'Published Table 6 feature subset is fixed; this run does not reproduce optimizer discovery.',
        'OC-SVM fits benign training rows; min-max bounds fit all paper-training rows.',
        'RBF gamma=scale is a library default not specified by the paper.',
        'Nu selected by Equation 9 on training resubstitution, not test; paper tuning protocol unspecified.',
        'Original split duplicates preserved. No validation split is invented.',
        'Classifier-only: no verified per-packet length correspondence for UNSW flow rows.']
    manifest=dict(schema_version=1,status='published_subset_classifier_reconstruction',admission=False,
        assumptions=assumptions,features=FEATURES,selected_indices_one_based=TABU_OCSVM_INDICES,
        selected_names=[f for f,s in zip(FEATURES,selected) if s],nu_candidates=list(NU_CANDIDATES),
        feature_selection='fixed_published_table6',length_layer='unresolved',
        normalization='fit all paper-training rows once; inference accepts unscaled numeric features',
        fit_rows=int((y==0).sum()),training_rows=len(train),
        input_sha256={'train':sha256(train_path),'preprocessing':sha256(prep_path)},
        software={n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','pandas','joblib']},
        source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('model.py')]})
    write_json(output/'manifest.json',manifest)
    candidates=[]
    for nu in NU_CANDIDATES:
        start=perf_counter()
        model=HybridModel(FEATURES,selected,nu,'unresolved').fit(x[y==0],scaler_reference=x)
        metrics=classification_metrics(y,model.predict(x,variant='classifier_only'))
        score=fitness(int(selected.sum()),len(selected),metrics)
        candidates.append((score,nu,model,metrics,perf_counter()-start))
        print(f'nu={nu}: training fitness={score:.6f}',flush=True)
    chosen=min(candidates,key=lambda c:(c[0],c[1]))
    if not np.isfinite(chosen[0]):raise ValueError('No finite training-fitness candidate')
    manifest['selected_nu']=chosen[1]
    # Persist the selection before opening the test dataset.
    write_json(output/'manifest.json',manifest)
    artifact=dict(schema_version=1,model=chosen[2],status=manifest['status'],admission=False,
                  assumptions=assumptions,manifest=manifest.copy(),input_sha256=manifest['input_sha256'])
    joblib.dump(artifact,output/'model-0.joblib')
    test=pd.read_csv(test_path);check(test,'test')
    xt=test[FEATURES].to_numpy(float);yt=test.label.to_numpy()
    results=[];predictions=pd.DataFrame({'row':np.arange(len(test)),'label':yt})
    for score,nu,model,training_metrics,seconds in candidates:
        predicted=model.predict(xt,variant='classifier_only')
        predictions[f'nu_{nu}']=predicted
        results.append(dict(nu=nu,selected=nu==chosen[1],training_seconds=seconds,
            training_fitness=score if np.isfinite(score) else None,
            training_metrics=training_metrics,test_metrics=classification_metrics(yt,predicted)))
        joblib.dump(dict(artifact,model=model),output/f'model-nu-{nu}.joblib')
    manifest['input_sha256']['test']=sha256(test_path)
    manifest['test_rows']=len(test)
    manifest['model_sha256']=sha256(output/'model-0.joblib')
    restored=joblib.load(output/'model-0.joblib')['model']
    if not np.array_equal(restored.predict(xt,variant='classifier_only'),predictions[f'nu_{chosen[1]}']):
        raise AssertionError('Saved model predictions differ')
    predictions.to_csv(output/'predictions.csv',index=False)
    write_json(output/'manifest.json',manifest)
    write_json(output/'results.json',dict(manifest=manifest,candidates=results))
    print(f'Saved selected nu={chosen[1]} model to {output / "model-0.joblib"}',flush=True)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for split in ['train','validation','test']:
        p.add_argument('--'+split, type=Path)
    p.add_argument('--features', help='Ordered comma-separated feature names')
    p.add_argument('--paper-data',type=Path,help='Reproduced UNSW preprocessing directory; fit the Table 6 subset')
    p.add_argument('--group-column', default='group')
    p.add_argument('--length-column')
    p.add_argument('--length-layer', choices=['dns','ip','frame','unresolved'], default='unresolved')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', default='tabu-pio-reconstruction')
    p.add_argument('--purpose', choices=['exploratory','smoke'], default='exploratory')
    p.add_argument('--runs', type=int, help='Search repetitions (default 20); published-subset mode uses one deterministic fit per nu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--population', type=int, default=128)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--local-iterations', type=int, default=10)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
