def true_positive_rate(tp, fn):
    if tp == 0:
        return 0
    return tp / (tp + fn)


def false_positive_rate(fp, tn):
    if fp == 0:
        return 0
    return fp / (fp + tn)


def classification_metrics(labels, predictions):
    """Binary metrics with explicit undefined values; 1 is the positive class."""
    labels, predictions = list(labels), list(predictions)
    if len(labels) != len(predictions):
        raise ValueError('Labels and predictions must be aligned')
    if any(value not in (0, 1) for value in labels + predictions):
        raise ValueError('Labels and predictions must be binary')
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    for actual, predicted in zip(labels, predictions):
        counts[{(1,1):'tp', (0,1):'fp', (0,0):'tn', (1,0):'fn'}[(actual,predicted)]] += 1
    return metrics_from_counts(**counts)


def metrics_from_counts(tp, fp, tn, fn):
    """Summarize one well-defined evaluation population, never mix units."""
    if any(int(v) != v or v < 0 for v in (tp, fp, tn, fn)):
        raise ValueError('Confusion counts must be non-negative integers')
    tp, fp, tn, fn = map(int, (tp,fp,tn,fn))
    def ratio(a,b): return a/b if b else None
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,n=tp+fp+tn+fn,
                accuracy=ratio(tp+tn,tp+fp+tn+fn), precision=ratio(tp,tp+fp),
                recall=ratio(tp,tp+fn), fpr=ratio(fp,fp+tn),
                f1=ratio(2*tp,2*tp+fp+fn))
