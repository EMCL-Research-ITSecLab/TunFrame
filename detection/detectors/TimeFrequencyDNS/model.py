"""Scaled SVM time-only and time-frequency ablations."""
import joblib
from detection.detectors.TimeFrequencyDNS.features import Lexicon,TIME_FEATURES


class TimeFrequencyModel:
    def __init__(self,artifact):
        if artifact['time_features']!=TIME_FEATURES:raise ValueError('Feature order differs from training')
        self.window=artifact['window'];self.fine=artifact['fine'];self.step=artifact['step']
        self.lexicon=Lexicon(artifact['words']);self.models=artifact['models'];self.artifact=artifact

    def score(self,features,variant='time_frequency'):
        if variant not in self.models:raise ValueError('Unknown feature variant')
        values=features[:len(TIME_FEATURES)] if variant=='time' else features
        return float(self.models[variant].decision_function([values])[0])

    def save(self,path):joblib.dump(self.artifact,path)

    @classmethod
    def load(cls,path):return cls(joblib.load(path))
