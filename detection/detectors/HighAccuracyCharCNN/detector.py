from urllib.parse import SplitResult
import tldextract

from detection.detector_base.detector_base import Detector
from detection.detectors.HighAccuracyCharCNN.config import MODEL_PATH, THRESHOLD
from detection.detectors.HighAccuracyCharCNN.model import CharCNN

_suffixes=tldextract.TLDExtract(suffix_list_urls=(),cache_dir=None)


class HighAccuracyCharCNN(Detector):
    def __init__(self):
        super().__init__()
        self.model=CharCNN.load(MODEL_PATH)

    def detect(self,logline):
        dns=logline.get('dns',{})
        name=dns.get('qname')
        if dns.get('flags',{}).get('qr') or not isinstance(name,str) or not name:
            return
        domain=_suffixes.extract_urllib(SplitResult('',name.rstrip('.'),'','','')).top_domain_under_public_suffix.lower()
        if domain and domain not in self.alarms and self.model.probabilities([name])[0]>=THRESHOLD:
            self.alarms.add(domain)
