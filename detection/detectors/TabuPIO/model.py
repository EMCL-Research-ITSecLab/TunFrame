"""Paper-specified inference and an explicitly unvalidated optimizer reconstruction.

See this detector's README.md for unresolved algorithm/feature semantics.
The reconstruction is executable, but is never admitted as an exact reproduction.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from detection.detectors.TabuPIO.metrics import classification_metrics

LENGTH_RANGES = ((100, 115), (145, 185), (220, 340))
NU_CANDIDATES = (0.074, 0.001, 0.01, 0.1)


def length_alarm(lengths):
    lengths = np.asarray(lengths, dtype=float)
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("Packet lengths must be finite non-negative bytes")
    return np.logical_or.reduce([(lengths >= lo) & (lengths <= hi)
                                 for lo, hi in LENGTH_RANGES])



def fitness(selected, total, metrics):
    """Equation 9; zero TPR has infinite cost, not an arbitrary epsilon."""
    if not selected or not metrics['recall'] or metrics['fpr'] is None:
        return float('inf')
    return .04 * selected / total + .48 * metrics['fpr'] + .48 / metrics['recall']


@dataclass(frozen=True)
class SearchSettings:
    population: int = 128
    iterations: int = 200
    local_iterations: int = 10
    # The paper does not specify tabu tenure or the transfer function.
    tabu_tenure: int = 10
    seed: int = 0


def select_features(evaluate, n_features, settings):
    """Binary Eq. 7–8 interpretation; assumptions are recorded in every artifact.

    Sigmoid transfer, single-bit tabu neighbours, FIFO tenure and aspiration are
    reconstruction choices. Algorithm 1 instead cites continuous Eq. 2–6; these
    conflicting paths must not be conflated. R=0.09 is consequently not used by
    this explicitly named binary variant.
    """
    if n_features < 1 or settings.population < 2 or settings.iterations < 1:
        raise ValueError("Positive dimensions/iterations and population >= 2 required")
    if settings.local_iterations < 0 or settings.tabu_tenure < 1:
        raise ValueError("Invalid local search settings")
    rng = np.random.default_rng(settings.seed)
    population = rng.integers(0, 2, (settings.population, n_features)).astype(bool)
    population[0] = True
    cache = {}
    def score(mask):
        key = tuple(mask)
        if key not in cache:
            cache[key] = evaluate(mask) if mask.any() else float('inf')
        return cache[key]
    best = min(population, key=score).copy()
    history = []
    tabu = []
    def move(pigeons):
        denominator = np.linalg.norm(pigeons, axis=1) * np.linalg.norm(best)
        cosine = np.divide(pigeons.astype(float) @ best.astype(float), denominator,
                           out=np.zeros(len(pigeons)), where=denominator != 0)
        keep = rng.random(pigeons.shape) < (1 / (1 + np.exp(-cosine)))[:, None]
        return np.where(keep, pigeons, best)
    for iteration in range(settings.iterations):
        population = move(population)
        candidate = min(population, key=score)
        if score(candidate) < score(best):
            best = candidate.copy()
        current = best.copy()
        for _ in range(settings.local_iterations):
            neighbours = []
            for bit in range(n_features):
                neighbour = current.copy()
                neighbour[bit] = ~neighbour[bit]
                if neighbour.any() and (bit not in tabu or score(neighbour) < score(best)):
                    neighbours.append((score(neighbour), bit, neighbour))
            if not neighbours:
                break
            _, bit, current = min(neighbours, key=lambda item: (item[0], item[1]))
            tabu.append(bit)
            tabu = tabu[-settings.tabu_tenure:]
            if score(current) < score(best):
                best = current.copy()
        population[0] = best
        history.append(dict(iteration=iteration + 1, fitness=score(best),
                            selected=int(best.sum()), evaluated=len(cache)))
    # Modified landmark: retain the best half, apply the same binary update.
    while len(population) > 1:
        population = np.array(sorted(population, key=score)[:max(1, len(population)//2)])
        population = move(population)
        candidate = min(population, key=score)
        if score(candidate) < score(best):
            best = candidate.copy()
    if not np.isfinite(score(best)):
        raise ValueError("No finite-fitness feature subset; inspect validation labels")
    return best, history


class HybridModel:
    """Frozen selected-feature OC-SVM with the paper's length override."""
    def __init__(self, feature_names, selected, nu, length_layer):
        self.feature_names = list(feature_names)
        self.selected = np.asarray(selected, dtype=bool)
        self.nu = float(nu)
        self.length_layer = length_layer
        if len(self.selected) != len(self.feature_names) or not self.selected.any():
            raise ValueError("Selected mask must match the feature schema")
        if length_layer not in ('dns', 'ip', 'frame', 'unresolved'):
            raise ValueError("Unknown length layer")
        self.scaler = MinMaxScaler()
        self.classifier = OneClassSVM(kernel='rbf', gamma='scale', nu=self.nu)

    def _matrix(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("Missing, non-finite, or mismatched model features")
        selected = x[:, self.selected]
        if not np.isfinite(selected).all():
            raise ValueError('Missing or non-finite selected model features')
        return selected

    def fit(self, x_benign, scaler_reference=None):
        # Benign-only OC-SVM training is an explicit reconstruction choice.
        reference = x_benign if scaler_reference is None else scaler_reference
        self.scaler.fit(self._matrix(reference))
        self.classifier.fit(self.scaler.transform(self._matrix(x_benign)))
        return self

    def predict(self, x, lengths=None, variant='combined'):
        if variant not in ('length_only', 'classifier_only', 'combined'):
            raise ValueError("Unknown detector variant")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError('Feature matrix must match the model schema')
        if variant != 'classifier_only':
            if self.length_layer == 'unresolved' or lengths is None:
                raise ValueError("Length-based modes require an explicit length layer and values")
            alarm = length_alarm(lengths)
            if alarm.shape != (len(x),):
                raise ValueError("Lengths must align with feature rows")
            if variant == 'length_only':
                return alarm.astype(int)
        else:
            alarm = np.zeros(len(x), dtype=bool)
        # Actually bypass inference on shortcut positives, including in timing runs.
        remaining = ~alarm
        if remaining.any():
            alarm[remaining] = self.classifier.predict(self.scaler.transform(self._matrix(x[remaining]))) == -1
        return alarm.astype(int)


ASSUMPTIONS = [
    'Binary update follows prose Eq. 7–8, not continuous Algorithm 1 Eq. 2–6.',
    'Sigmoid transfer, one-bit neighbourhood, FIFO tabu tenure=10, aspiration on global improvement.',
    'OC-SVM and scaler fit benign training rows; mixed validation labels score fitness and nu.',
    'Map/compass factor 0.09 is unused in the binary interpretation.',
    'Feature schema and packet-length layer require independent verification against the paper.',
]
