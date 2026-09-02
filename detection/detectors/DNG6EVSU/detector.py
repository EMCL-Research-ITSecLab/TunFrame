from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import get_registered_domain, get_fqdn
from detection.detectors.DNG6EVSU.train import load_model
from detection.detectors.DNG6EVSU.matchgram import MatchGram

import os
import pickle

# OptiTuneD: An Optimized Framework for Zero-Day DNS Tunnel Detection Using N-Grams

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model/matchgram_Dataset-2_n1_3000domains.pkl")

class OptiTuneD(Detector):

    def __init__(self):
        super().__init__()
        self.model = load_model(MODEL_PATH)

    def detect(self, logline: dict):
        domain = get_registered_domain(logline)
        if not domain:
            return
        if domain in self.alarms:
            return
        if not self.model.check_domain(get_fqdn(logline)):
            self.alarms.add(domain)

