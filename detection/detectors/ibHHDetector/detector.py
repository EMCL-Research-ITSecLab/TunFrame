import sys
import os
import tldextract
import pandas as pd
import tldextract
import datetime
from pathlib import Path


from detection.detectors.ibHHDetector.config import (
    pt_path,
    global_allowlist_path,
    detections_path,
    wt_dataset_path,
    detection_threshold_path,
    k,
    time_window,
)
from detection.detectors.ibHHDetector.ibHH.InformationBasedHeavyHitter import InformationBasedHeavyHitter     
from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import extract_timestamp, get_registered_domain, get_fqdn


class ibHHDetector(Detector):

    def __init__(self):
        super().__init__()
        self.current_window = 0
        self.ibhh = None
        self.extractor = tldextract.TLDExtract()
        with open(f"{Path(__file__).resolve().parent}/detection_threshold.txt", "r") as f:
            self.detection_threshold = int(f.read())

    def detect(self, logline: dict):
        
        if get_registered_domain(logline) in self.alarms:
            return
        timestamp = extract_timestamp(logline)
        extracted = self.extractor(get_fqdn(logline))
        domain = extracted.top_domain_under_public_suffix.lower()
        subdomain = extracted.subdomain
        if timestamp > self.current_window + time_window:
            self.ibhh = InformationBasedHeavyHitter(k=k)
            self.current_window = timestamp
        self.ibhh.add_pair(subdomain, domain)
        count = self.ibhh.count_domain_information(domain)
        if count > self.detection_threshold:
            self.alarms.add(domain)

