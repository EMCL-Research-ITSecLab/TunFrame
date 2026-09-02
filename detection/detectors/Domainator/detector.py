#!/usr/bin/env python3

import sys
import os
import joblib
import numpy as np
import tldextract
from typing import Set


from detection.detector_base.detector_base import Detector
from detection.detector_base.feature_extraction import get_registered_domain, extract_subdomain_string
from detection.detectors.Domainator.config import (
    MODEL_PATH,
    DETECTION_THRESHOLD,
    FEATURE_NAMES,
    VERBOSE
)
from detection.detectors.Domainator.aggregator import DomainWindowManager



class DomainatorDetector(Detector):
    """
    Real-time DNS Tunneling Detector based on Domainator methodology.
    
    Detection Pipeline:
    1. Group DNS queries by registered domain
    2. Extract subdomains and build sliding windows (size=10, step=10)
    3. Compute 7 string-similarity metrics per window
    4. Classify window using Random Forest (malicious vs. legitimate)
    5. Add domain to alarms if classified as malicious
    
    Paper Reference:
        Petrov et al. 2025, Section 3.5 & 4.1
        Binary Classification F1-Score: 0.966
        False Positive Rate: 1.2%
    """
    
    def __init__(self):
        """Initialize Domainator Detector."""
        super().__init__()
        
        # Window manager (handles sliding windows per domain)
        self.window_manager = DomainWindowManager()
        
        # TLD extractor
        self.tld_extractor = tldextract.TLDExtract()
        
        # Load trained Random Forest model
        self.model = self._load_model()
        
        if VERBOSE:
            print("[Domainator] Initialized")
            print(f"[Domainator] Model: {MODEL_PATH}")
            print(f"[Domainator] Detection Threshold: {DETECTION_THRESHOLD}")
    
    def _load_model(self):
        """Load trained Random Forest model from pickle file."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                f"Please train the model first using train_classifier.py"
            )
        
        try:
            model = joblib.load(MODEL_PATH)
            if VERBOSE:
                print("[Domainator] ✓ Model loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    
    def detect(self, logline: dict) -> None:
        """
        Process a single DNS query logline.
        
        Args:
            logline: Dictionary containing DNS query metadata
                     Expected keys: 'dns.qname' (FQDN)
        
        Detection Logic:
            1. Extract registered domain and subdomain
            2. Skip if domain already in alarms
            3. Add query to sliding window buffer
            4. If window ready: extract features and classify
            5. If malicious: add domain to alarms
        """
        
        domain = get_registered_domain(logline)
        if not domain:
            return
        
        if domain in self.alarms:
            return
        
        subdomain = extract_subdomain_string(logline)
        if not subdomain:
            return
        
        window_ready, features = self.window_manager.add_query(domain, subdomain)
        
        if window_ready:
            self._classify_window(domain, features)
    
    def _classify_window(self, domain: str, features: np.ndarray) -> None:
        """
        Classify a feature vector using Random Forest.
        
        Args:
            domain: Registered domain being classified
            features: NumPy array of shape (7,) with feature values
        """
        
        features_2d = features.reshape(1, -1)
        prediction = self.model.predict(features_2d)[0]
        probabilities = self.model.predict_proba(features_2d)[0]
        malicious_prob = probabilities[1]
        
        if VERBOSE:
            print(f"[Domainator] Domain: {domain} | "
                  f"Pred: {prediction} | "
                  f"P(malicious): {malicious_prob:.3f}")
        
        if prediction == 1 and malicious_prob >= DETECTION_THRESHOLD:
            self.alarms.add(domain)
    
