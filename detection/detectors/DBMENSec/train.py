"""Train a documented DBM-ENSec reconstruction from explicit Table 2 features."""

import argparse
import copy
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, train_test_split,
)
from torch import nn

from .model import Encoder, decision_scores, extract
from .preprocess import FEATURE_NAMES, fit_scaler, grayscale, load_csv

HERE = Path(__file__).resolve().parent


def inner_split(labels, groups, seed):
    if groups is None:
        return train_test_split(np.arange(len(labels)), test_size=0.2,
                                stratify=labels, random_state=seed)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(labels)), labels, groups))


def require_classes(labels, name):
    if set(labels) != {0, 1}:
        raise ValueError(f"{name} needs both classes; provide more independent groups/samples")


def fit(matrix, labels, groups, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, validation = inner_split(labels, groups, seed)
    require_classes(labels[train], "Inner training split")
    require_classes(labels[validation], "Inner validation split")
    scaler = fit_scaler(matrix[train])
    images = torch.tensor(grayscale(matrix, scaler).astype(np.float32) / 255)
    targets = torch.tensor(labels, dtype=torch.float32)
    encoder = Encoder()
    encoder.dbm.pretrain(images[train].flatten(1), args.pretrain_epochs, args.batch_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate,
                                 weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stalled, best_state, history = float("inf"), 0, None, []
    for epoch in range(args.epochs):
        encoder.train()
        loss_sum = 0.0
        for index in torch.tensor(train)[torch.randperm(len(train))].split(args.batch_size):
            optimizer.zero_grad()
            loss = criterion(encoder(images[index]), targets[index])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
        encoder.eval()
        with torch.no_grad():
            val_sum = 0.0
            for index in torch.tensor(validation).split(args.batch_size):
                val_sum += float(criterion(encoder(images[index]), targets[index])) * len(index)
            val_loss = val_sum / len(validation)
        history.append((epoch + 1, loss_sum / len(train), val_loss))
        if not np.isfinite(history[-1][1:]).all():
            raise ValueError("Nonfinite training loss; refusing to export a broken model")
        print(f"  epoch {epoch + 1}: loss={history[-1][1]:.5f} val={val_loss:.5f}", flush=True)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, stalled = val_loss, 0
            best_state = copy.deepcopy(encoder.state_dict())
        else:
            stalled += 1
            if stalled >= 5:
                break
    encoder.load_state_dict(best_state)
    features = extract(encoder, images, args.batch_size)
    # Tree settings are declared reconstruction defaults, not published settings.
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    classifiers = [
        LGBMClassifier(n_estimators=args.trees, max_depth=6, num_leaves=31,
                       random_state=seed, n_jobs=args.threads, verbosity=-1),
        XGBClassifier(n_estimators=args.trees, max_depth=6, learning_rate=0.1,
                      random_state=seed, n_jobs=args.threads, tree_method="hist"),
        RandomForestClassifier(n_estimators=args.trees, random_state=seed,
                                n_jobs=args.threads),
        CatBoostClassifier(iterations=args.trees, depth=6, learning_rate=0.1,
                           random_seed=seed, thread_count=args.threads,
                           verbose=False, allow_writing_files=False),
    ]
    weights = []
    for classifier in classifiers:
        classifier.fit(features[train], labels[train])
        prediction = classifier.predict(features[validation]).reshape(-1)
        weights.append(f1_score(labels[validation], prediction, average="weighted",
                                zero_division=0))
    # Fail on undefined voting rather than silently replacing all-zero weights.
    decision_scores(np.zeros((1, 4)), weights, args.weight_power)
    bundle = dict(
        feature_names=FEATURE_NAMES, scaler=scaler, classifiers=classifiers,
        f1_weights=weights, threshold=args.threshold, weight_power=args.weight_power,
        settings=vars(args), seed=seed, train_samples=len(train),
        validation_samples=len(validation),
        status="reconstruction; author feature extractor and weights unavailable",
    )
    return encoder, bundle, history


def predict(encoder, bundle, matrix):
    images = torch.tensor(grayscale(matrix, bundle["scaler"]).astype(np.float32) / 255)
    features = extract(encoder, images)
    votes = np.column_stack([classifier.predict(features).reshape(-1)
                             for classifier in bundle["classifiers"]])
    probabilities = np.column_stack([classifier.predict_proba(features)[:, 1]
                                     for classifier in bundle["classifiers"]])
    scores = decision_scores(votes, bundle["f1_weights"], bundle["weight_power"])
    return scores, votes, probabilities


def metrics(labels, prediction, scores):
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(labels, prediction),
                precision=precision_score(labels, prediction, zero_division=0),
                recall=recall_score(labels, prediction, zero_division=0),
                f1=f1_score(labels, prediction, zero_division=0),
                fpr=fp / (fp + tn), auc=roc_auc_score(labels, scores),
                ap=average_precision_score(labels, scores),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(labels, scores, prediction, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), layout="constrained")
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[0].plot(fpr, tpr, label=f"AUC {roc_auc_score(labels, scores):.3f}")
        axes[0].plot([0, 1], [0, 1], ":", color="0.6")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                    xlim=(0, 1), ylim=(0, 1.02))
        axes[0].legend(loc="lower right", frameon=False)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[1].plot(recall, precision,
                     label=f"AP {average_precision_score(labels, scores):.3f}")
        axes[1].axhline(labels.mean(), linestyle=":", color="0.6")
        axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        axes[1].legend(loc="lower left", frameon=False)
        counts = confusion_matrix(labels, prediction, labels=[0, 1])
        axes[2].imshow(counts / counts.sum(axis=1, keepdims=True),
                       vmin=0, vmax=1, cmap="Blues")
        for (row, column), value in np.ndenumerate(counts):
            fraction = value / counts[row].sum()
            axes[2].text(column, row, f"{value}\n{fraction:.1%}", ha="center", va="center",
                         color="white" if fraction > 0.5 else "black")
        axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Tunnel"],
                    yticklabels=["Benign", "Tunnel"], xlabel="Predicted", ylabel="Actual")
        for extension in ("pdf", "png"):
            figure.savefig(output / f"cross-validation.{extension}", dpi=200)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="23 numeric Table 2 columns plus label; optional group")
    parser.add_argument("--output", type=Path, default=HERE / "model")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Section 4.3: 0.1; Algorithm 1 instead specifies 0.01")
    parser.add_argument("--weight-power", type=int, choices=[1, 2], default=2,
                        help="Equation 5: squared F1; Algorithm 1: linear F1")
    args = parser.parse_args()
    if min(args.epochs, args.pretrain_epochs, args.batch_size, args.trees, args.threads) < 1:
        parser.error("Epochs, batch size, trees and threads must be positive")
    if args.folds < 2 or not 0 < args.threshold <= 1 or args.learning_rate <= 0 or args.l2 < 0:
        parser.error("Require folds>=2, 0<threshold<=1, learning-rate>0 and l2>=0")
    torch.set_num_threads(args.threads)
    matrix, labels, groups = load_csv(args.csv)
    splitter = (StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
                if groups is None else
                StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed))
    print(f"{len(labels)} samples; split unit: {'row' if groups is None else 'group'}", flush=True)
    args.results.mkdir(parents=True, exist_ok=True)
    scores, assignments = np.empty(len(labels)), np.empty(len(labels), dtype=int)
    rows = []
    names = ("LightGBM", "XGBoost", "RandomForest", "CatBoost")
    for fold, (train, test) in enumerate(splitter.split(matrix, labels, groups), 1):
        require_classes(labels[test], f"Outer fold {fold}")
        print(f"Fold {fold}/{args.folds}: train={len(train)} test={len(test)}", flush=True)
        encoder, bundle, _ = fit(matrix[train], labels[train],
                                 None if groups is None else groups[train], args, args.seed + fold)
        fold_scores, votes, probabilities = predict(encoder, bundle, matrix[test])
        scores[test], assignments[test] = fold_scores, fold
        for column, name in enumerate(names):
            rows.append(dict(fold=fold, model=name,
                             **metrics(labels[test], votes[:, column], probabilities[:, column])))
        rows.append(dict(fold=fold, model="DBMENSec", **metrics(
            labels[test], fold_scores >= args.threshold, fold_scores)))
        print(rows[-1], flush=True)
    prediction = (scores >= args.threshold).astype(int)
    write_csv(args.results / "fold-metrics.csv", rows)
    write_csv(args.results / "predictions.csv", [dict(
        row=index, fold=assignments[index], label=label, score=score,
        prediction=predicted) for index, (label, score, predicted)
        in enumerate(zip(labels, scores, prediction))])
    ensemble_rows = [row for row in rows if row["model"] == "DBMENSec"]
    summary = []
    for key in metrics(labels, prediction, scores):
        values = [row[key] for row in ensemble_rows]
        summary.append(dict(metric=key, mean=np.mean(values), std=np.std(values, ddof=1)))
    write_csv(args.results / "cross-validation.csv", summary)
    plot_results(labels, scores, prediction, args.results)
    print("Final fit with an internal validation split; CV scores above remain held out.", flush=True)
    encoder, bundle, history = fit(matrix, labels, groups, args, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.output / "encoder.pt")
    joblib.dump(bundle, args.output / "ensemble.joblib")
    write_csv(args.results / "training.csv", [dict(epoch=e, loss=t, validation_loss=v)
                                              for e, t, v in history])
    print(f"Saved reconstruction to {args.output}", flush=True)


if __name__ == "__main__":
    main()
