from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import get_registered_domain, extract_subdomain_string

# DNS tunneling Detection Using Elasticsearch

class BLVZG5P6(Detector):

    def __init__(self):
        super().__init__()
        self.domain_tracker = {}

    def detect(self, logline: dict):
        domain = get_registered_domain(logline)
        if not domain:
            return
        if domain in self.alarms:
            return
        if domain not in self.domain_tracker:
            self.domain_tracker[domain] = set()

        subdomain = extract_subdomain_string(logline)

        if not subdomain:
            return
        
        self.domain_tracker[domain].add(subdomain)

        if len(self.domain_tracker[domain]) > 300:
            self.alarms.add(domain)

