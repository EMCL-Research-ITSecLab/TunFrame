"""Live query-response adapter; incomplete pairs never become benign decisions."""
from detection.detector_base.detector_base import Detector
from detection.detectors.EncodingBehaviorFusion.config import MODEL_DIR, PAIR_TIMEOUT, MAX_PENDING
from detection.detectors.EncodingBehaviorFusion.features import Pairer, from_collector, registered_domain
from detection.detectors.EncodingBehaviorFusion.model import FusionModel


class EncodingBehaviorFusion(Detector):
    def __init__(self):
        super().__init__()
        self.model = FusionModel.load(MODEL_DIR)
        self.pairer = Pairer(PAIR_TIMEOUT, MAX_PENDING)
        self.excluded = 0

    def reset(self):
        super().reset()
        self.pairer = Pairer(PAIR_TIMEOUT, MAX_PENDING)
        self.excluded = 0

    def detect(self, logline):
        try:
            pair = self.pairer.add(from_collector(logline))
        except (KeyError, ValueError, TypeError, OverflowError):
            self.excluded += 1
            return
        if pair and self.model.predict([pair])[0] == 1:
            domain = registered_domain(pair['qname'])
            if domain:
                self.alarms.add(domain)

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        result['pair_statistics'] = dict(self.pairer.counts, pending=len(self.pairer.pending), excluded=self.excluded)
        result['decision_unit'] = 'dns_query_response_pair'
        result['reproduction_status'] = 'paper_based_reconstruction'
        return result
