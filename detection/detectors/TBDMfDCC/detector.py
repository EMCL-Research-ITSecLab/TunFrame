"""Framework adapter for the authors' DistilBERT DNS classifier."""
from detection.detector_base.detector_base import Detector
from urllib.parse import SplitResult
import tldextract
from detection.detectors.TBDMfDCC.config import (MODEL_DIR, MAX_LENGTH, VARIANT,
                                               IMPROVED_MODEL_DIR, IMPROVED_MAX_LENGTH)
from detection.detectors.TBDMfDCC.model import DNSClassifier

_suffixes = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


class TBDMfDCC(Detector):
    def __init__(self, variant=None):
        super().__init__()
        self.variant = VARIANT if variant is None else variant
        if self.variant == 'authors':
            self.classifier = DNSClassifier(MODEL_DIR, MAX_LENGTH)
        elif self.variant == 'improved':
            self.classifier = DNSClassifier(IMPROVED_MODEL_DIR, IMPROVED_MAX_LENGTH)
        else:
            raise ValueError('TBDMfDCC variant must be authors or improved')

    def detect(self, logline):
        name = logline.get('dns', {}).get('qname')
        if not isinstance(name, str) or not name:
            return
        # Payload labels may contain '/' or '@'; a DNS name is not a URL.
        parsed = _suffixes.extract_urllib(SplitResult('', name.rstrip('.'), '', '', ''))
        domain = parsed.top_domain_under_public_suffix.lower()
        if not domain or domain in self.alarms:
            return
        # The authors' DNS datasets contain absolute names with a trailing dot.
        if self.classifier.predict([name.rstrip('.') + '.'])[0] == 1:
            self.alarms.add(domain)

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        result['model_variant'] = self.variant
        result['token_limit'] = self.classifier.max_length
        return result
