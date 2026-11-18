"""Explicit reconstruction of the underspecified DBM/CNN in Li et al. (2025).

8x8 is interpreted as two hidden layers of eight units. This is a genuine
undirected Bernoulli DBM with mean-field inference, not Algorithm 1's unexplained
'Dual-BranchModule'. Pretraining: greedy RBM CD-1 followed by joint persistent
contrastive divergence. Residual projection and CNN placement are assumptions:
add a projection of 25 pixels to the 8-unit top hidden layer, then Conv1D.
Attention uses one scaled-dot-product head twice, with learned Q/K/V projections.
These choices are reproducible, but cannot establish author-code equivalence.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .preprocess import FEATURE_NAMES, grayscale


class BoltzmannMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(25, 8) * 0.01)
        self.w2 = nn.Parameter(torch.randn(8, 8) * 0.01)
        self.bv = nn.Parameter(torch.zeros(25))
        self.b1 = nn.Parameter(torch.zeros(8))
        self.b2 = nn.Parameter(torch.zeros(8))

    def forward(self, visible):
        h1 = torch.sigmoid(visible @ self.w1 + self.b1)
        h2 = torch.sigmoid(h1 @ self.w2 + self.b2)
        for _ in range(5):
            h1 = torch.sigmoid(visible @ self.w1 + h2 @ self.w2.T + self.b1)
            h2 = torch.sigmoid(h1 @ self.w2 + self.b2)
        return h1, h2

    @torch.no_grad()
    def pretrain(self, visible, epochs, batch_size, learning_rate=0.01):
        # Greedy layerwise RBM initialization, before joint DBM training.
        data = visible
        for weight, vb, hb in ((self.w1, self.bv, self.b1),
                               (self.w2, self.b1, self.b2)):
            for _ in range(epochs):
                for index in torch.randperm(len(data)).split(batch_size):
                    v0 = data[index]
                    p0 = torch.sigmoid(v0 @ weight + hb)
                    vn = torch.sigmoid(torch.bernoulli(p0) @ weight.T + vb)
                    pn = torch.sigmoid(vn @ weight + hb)
                    rate = learning_rate / len(index)
                    weight.add_(rate * (v0.T @ p0 - vn.T @ pn))
                    vb.add_(learning_rate * (v0 - vn).mean(0))
                    hb.add_(learning_rate * (p0 - pn).mean(0))
            data = torch.sigmoid(data @ weight + hb)

        # Persistent negative chain of fixed batch size; positive phase uses
        # clamped-visible mean-field expectations for the undirected DBM.
        negative_v = torch.bernoulli(torch.full((batch_size, 25), 0.5))
        negative_h2 = torch.bernoulli(torch.full((batch_size, 8), 0.5))
        for _ in range(epochs):
            for index in torch.randperm(len(visible)).split(batch_size):
                positive_v = visible[index]
                positive_h1, positive_h2 = self(positive_v)
                for _ in range(5):
                    negative_h1 = torch.bernoulli(torch.sigmoid(
                        negative_v @ self.w1 + negative_h2 @ self.w2.T + self.b1))
                    negative_v = torch.bernoulli(torch.sigmoid(
                        negative_h1 @ self.w1.T + self.bv))
                    negative_h2 = torch.bernoulli(torch.sigmoid(
                        negative_h1 @ self.w2 + self.b2))
                self.w1.add_(learning_rate * (
                    positive_v.T @ positive_h1 / len(index)
                    - negative_v.T @ negative_h1 / batch_size))
                self.w2.add_(learning_rate * (
                    positive_h1.T @ positive_h2 / len(index)
                    - negative_h1.T @ negative_h2 / batch_size))
                for bias, positive, negative in (
                    (self.bv, positive_v, negative_v),
                    (self.b1, positive_h1, negative_h1),
                    (self.b2, positive_h2, negative_h2),
                ):
                    bias.add_(learning_rate * (positive.mean(0) - negative.mean(0)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbm = BoltzmannMachine()
        self.residual = nn.Linear(25, 8)
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.dropout = nn.Dropout(0.4)
        self.attention1 = nn.MultiheadAttention(128, 1, batch_first=True)
        self.attention2 = nn.MultiheadAttention(128, 1, batch_first=True)
        self.head = nn.Linear(4 * 128, 1)

    def extract(self, images):
        visible = images.flatten(1)
        _, hidden = self.dbm(visible)
        hidden = F.relu(hidden + self.residual(visible)).unsqueeze(1)
        hidden = self.dropout(F.relu(self.conv1(hidden)))
        hidden = self.dropout(F.relu(self.conv2(hidden)))
        hidden = F.max_pool1d(hidden, 2).transpose(1, 2)
        hidden, _ = self.attention1(hidden, hidden, hidden, need_weights=False)
        hidden, _ = self.attention2(hidden, hidden, hidden, need_weights=False)
        return hidden.flatten(1)

    def forward(self, images):
        return self.head(self.extract(images)).squeeze(1)


@torch.no_grad()
def extract(model, images, batch_size=256):
    model.eval()
    return np.concatenate([
        model.extract(batch).numpy() for batch in images.split(batch_size)
    ])


def decision_scores(predictions, f1_weights, power=2):
    """Equation 5: F1-squared weighted hard votes, not probability averaging.

    Algorithm 1 instead prints linear F1 weights. power=1 exposes that variant.
    """
    weights = np.asarray(f1_weights, dtype=np.float64)
    if (weights.shape != (4,) or not np.isfinite(weights).all()
            or (weights < 0).any() or (weights > 1).any() or weights.sum() <= 0):
        raise ValueError("Four nonnegative F1 weights with positive sum required")
    if power not in (1, 2):
        raise ValueError("Weight power must select Equation 5 (2) or Algorithm 1 (1)")
    weights = weights ** power
    votes = np.asarray(predictions)
    if votes.ndim != 2 or votes.shape[1] != 4 or not np.isin(votes, [0, 1]).all():
        raise ValueError("Expected N x 4 binary classifier decisions")
    return votes @ weights / weights.sum()


class Predictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.bundle = joblib.load(directory / "ensemble.joblib")
        if tuple(self.bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Model feature schema differs from Table 2 contract")
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(directory / "encoder.pt",
                                                map_location="cpu", weights_only=True))
        self.encoder.eval()

    def scores(self, matrix):
        pixels = grayscale(matrix, self.bundle["scaler"])
        images = torch.from_numpy(pixels.astype(np.float32) / 255)
        features = extract(self.encoder, images)
        votes = np.column_stack([model.predict(features).reshape(-1)
                                 for model in self.bundle["classifiers"]])
        return decision_scores(votes, self.bundle["f1_weights"], self.bundle["weight_power"])

    def predict(self, matrix):
        return (self.scores(matrix) >= self.bundle["threshold"]).astype(np.int64)
