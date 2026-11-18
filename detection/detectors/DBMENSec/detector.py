"""Framework bridge for explicitly supplied Table 2 packet features."""

import os
from pathlib import Path
from urllib.parse import SplitResult

import tldextract

from detection.detector_base.detector_base import Detector
from .model import Predictor
from .preprocess import feature_matrix


class DBMENSecDetector(Detector):
    def __init__(self):
        super().__init__()
        directory = Path(os.environ.get("DBMENSEC_MODEL_DIR",
                         str(Path(__file__).resolve().parent / "model")))
        if not (directory / "ensemble.joblib").is_file():
            raise FileNotFoundError(
                "DBM-ENSec has no author weights. Train explicit Table 2 features "
                "with python -m detection.detectors.DBMENSec.train first.")
        self.model = Predictor(directory)
        self.domains = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
        self.packet_decisions = 0
        self.packet_positives = 0

    def reset(self):
        super().reset()
        self.packet_decisions = self.packet_positives = 0

    def detect(self, logline):
        features = logline.get("dbm_ensec_features")
        if not isinstance(features, dict):
            raise ValueError(
                "DBM-ENSec requires dbm_ensec_features with all 23 Table 2 values. "
                "The ordinary collector cannot reproduce these features; "
                "unavailable fields must not be replaced with zeros.")
        name = logline.get("dns", {}).get("qname", "").rstrip(".")
        domain = self.domains.extract_urllib(
            SplitResult("", name, "", "", "")).top_domain_under_public_suffix
        if not domain:
            raise ValueError("DBM-ENSec packet needs a registered qname for framework alarms")
        prediction = int(self.model.predict(feature_matrix([features]))[0])
        self.packet_decisions += 1
        self.packet_positives += prediction
        if prediction:
            self.alarms.add(domain)

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        result.update(native_unit="packet", native_packet_decisions=self.packet_decisions,
                      native_packet_positives=self.packet_positives,
                      alarm_projection="positive packet -> registered domain")
        return result
