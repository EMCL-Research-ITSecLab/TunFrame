from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import get_registered_domain, extract_subdomain_string

# DNS-tunneling-detection Method by Monitoring DNS Subdomain Length for General Usage

class HLVZLLAL(Detector):

    def __init__(self):
        super().__init__()

    def detect(self, logline: dict):
        domain = get_registered_domain(logline)
        if not domain:
            return
        if domain in self.alarms:
            return
        if len(extract_subdomain_string(logline)) > 180:
            self.alarms.add(domain)

