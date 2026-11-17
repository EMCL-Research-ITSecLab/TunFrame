"""Paper curation steps 1–5 and Table II grouped-prefix features."""
from collections import Counter
from datetime import datetime,timezone
import math
from os.path import commonprefix
from urllib.parse import SplitResult
import tldextract

_suffixes=tldextract.TLDExtract(suffix_list_urls=(),cache_dir=None)
FEATURES=('mean_length','mean_unique_characters','mean_label_count','mean_gini',
          'mean_classification_error','mean_digits','mean_repeat_ratio','mean_cnn_probability',
          'mean_normalized_entropy','concatenated_normalized_entropy','unique_prefixes','group_size')


def domain(name):
    return _suffixes.extract_urllib(SplitResult('',name.rstrip('.'),'','','')).top_domain_under_public_suffix.lower()


def group_key(name,client,timestamp):
    name=name.rstrip('.')
    if name.lower().endswith('.arpa'):return None
    sld=domain(name)
    if not sld:return None
    if isinstance(timestamp,str):
        stamp=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
        if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
    else:stamp=datetime.fromtimestamp(float(timestamp),timezone.utc)
    return sld,str(client),stamp.astimezone(timezone.utc).date().isoformat()


def curate(names):
    names=[n.rstrip('.') for n in names]
    if len(names)<2:return None
    suffix=commonprefix([n[::-1] for n in names])[::-1]
    retained=[(name[:-len(suffix)] if suffix else name,name.count('.')+1) for name in names]
    retained=[row for row in retained if row[0]]
    if len(retained)<2:return None
    return dict(prefixes=[r[0] for r in retained],label_counts=[r[1] for r in retained],suffix=suffix)


def normalized_entropy(text):
    if len(text)<=1:return 0.
    probabilities=[count/len(text) for count in Counter(text).values()]
    return -sum(p*math.log2(p) for p in probabilities)/math.log2(len(text))


def statistics(group,probabilities=None):
    rows=[]
    for prefix,labels in zip(group['prefixes'],group['label_counts']):
        counts=Counter(prefix);n=len(prefix);unique=len(counts)
        p=[v/n for v in counts.values()]
        rows.append([n,unique,labels,1-sum(v*v for v in p),1-max(p),
                     sum(c.isascii() and c.isdigit() for c in prefix),(n-unique)/unique,
                     normalized_entropy(prefix)])
    mean=[sum(r[i] for r in rows)/len(rows) for i in range(8)]
    # Keep Table II feature 7 reserved for the CNN; plain RF drops this column.
    score=sum(probabilities)/len(probabilities) if probabilities is not None else 0.
    return mean[:7]+[score,mean[7],normalized_entropy(''.join(group['prefixes'])),
                     len(set(group['prefixes'])),len(group['prefixes'])]
