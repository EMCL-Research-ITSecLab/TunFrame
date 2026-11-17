"""Figure 4 attention encoder (reconstructed) and Table 4 XGBoost fusion."""
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from xgboost import XGBClassifier

from detection.detectors.EncodingBehaviorFusion.features import encode


class AttentionEncoder(nn.Module):
    def __init__(self, length=256):
        super().__init__()
        self.length = length
        self.embedding = nn.Embedding(39, 4)
        # Eight separate four-dimensional Q/K/V projections: four input
        # dimensions cannot directly use torch MultiheadAttention with 8 heads.
        self.qkv = nn.Linear(4, 8*4*3)
        self.feed_forward = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.LayerNorm(32))
        self.output = nn.Linear(32, 2)
        position = torch.arange(length).float().unsqueeze(1)
        divisor = torch.exp(torch.arange(0, 4, 2).float()*(-math.log(10000.)/4))
        pe = torch.zeros(length, 4)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position*divisor), torch.cos(position*divisor)
        self.register_buffer('position', pe)

    def forward(self, tokens, lengths):
        value = self.embedding(tokens) + self.position
        batch, size, _ = value.shape
        q, k, v = self.qkv(value).reshape(batch, size, 3, 8, 4).permute(2, 0, 3, 1, 4)
        mask = torch.arange(size, device=tokens.device)[None, :] < lengths[:, None]
        scores = (q @ k.transpose(-1, -2))/2.
        scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, size, 32)
        context = self.feed_forward(context)
        pooled = (context*mask[:, :, None]).sum(1)/lengths[:, None].clamp(min=1)
        return self.output(pooled)


def tensors(contents, length):
    rows = [encode(text, length) for text in contents]
    return torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows])


class FusionModel:
    def __init__(self, encoder, classifier):
        self.encoder = encoder.eval()
        self.classifier = classifier

    def features(self, records):
        if not records:
            return np.empty((0, 8), dtype=np.float32)
        tokens, lengths = tensors([r['content'] for r in records], self.encoder.length)
        with torch.inference_mode():
            encoded = self.encoder(tokens, lengths).cpu().numpy()
        return np.concatenate([encoded, np.asarray([r['statistics'] for r in records], dtype=np.float32)], axis=1)

    def predict(self, records):
        return self.classifier.predict(self.features(records)) if records else np.array([], dtype=int)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(dict(length=self.encoder.length, state=self.encoder.state_dict()), directory/'encoder.pt')
        self.classifier.save_model(directory/'classifier.ubj')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        state = torch.load(directory/'encoder.pt', map_location='cpu', weights_only=True)
        encoder = AttentionEncoder(state['length'])
        encoder.load_state_dict(state['state'], strict=True)
        classifier = XGBClassifier()
        classifier.load_model(directory/'classifier.ubj')
        return cls(encoder, classifier)


def classifier(seed):
    return XGBClassifier(max_depth=4, n_estimators=150, min_child_weight=6,
        subsample=1., colsample_bytree=1., gamma=0., reg_alpha=0., reg_lambda=1.,
        learning_rate=.03, objective='binary:logistic', eval_metric='logloss',
        random_state=seed, n_jobs=4)
