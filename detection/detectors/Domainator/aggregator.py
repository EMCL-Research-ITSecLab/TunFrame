#!/usr/bin/env python3

import sys
import os
import numpy as np
from collections import defaultdict, deque
from itertools import combinations
from typing import List, Tuple, Dict

import Levenshtein
import jellyfish


from detection.detectors.Domainator.config import WINDOW_SIZE, STEP_SIZE, MIN_WINDOW_SIZE
from detection.detector_base.feature_extraction import extract_subdomain_string


class DomainWindowBuffer:
    """
    Manages sliding windows per domain (registered domain).
    Stores subdomains and triggers feature extraction when window is full.
    """
    
    def __init__(self, domain: str, window_size: int = WINDOW_SIZE, 
                 step_size: int = STEP_SIZE, min_size: int = MIN_WINDOW_SIZE):
        self.domain = domain
        self.window_size = window_size
        self.step_size = step_size
        self.min_size = min_size
        
        self.subdomain_buffer = deque(maxlen=window_size)
        self.query_count = 0
        
    def add_subdomain(self, subdomain: str) -> bool:
        """
        Add a subdomain to the buffer.
        Returns True if window is ready for feature extraction.
        """
        self.subdomain_buffer.append(subdomain)
        self.query_count += 1
        
        if len(self.subdomain_buffer) >= self.min_size:
            if self.query_count % self.step_size == 0:
                return True
        
        return False
    
    def get_window(self) -> List[str]:
        """Return current window as list of subdomains."""
        return list(self.subdomain_buffer)
    
    def has_minimum_queries(self) -> bool:
        """Check if buffer has minimum required queries."""
        return len(self.subdomain_buffer) >= self.min_size



def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein Distance: 1 - (distance / max_length)
    Range: [0, 1] where 1 = identical
    """
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    distance = Levenshtein.distance(s1, s2)
    
    return 1.0 - (distance / max_len) if max_len > 0 else 0.0


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro Similarity (range: [0, 1])"""
    if not s1 or not s2:
        return 0.0
    return jellyfish.jaro_similarity(s1, s2)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler Similarity (range: [0, 1])"""
    if not s1 or not s2:
        return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)


def longest_common_substring(s1: str, s2: str) -> int:
    """
    Length of Longest Common Substring (consecutive chars).
    Dynamic programming approach.
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
    
    return max_len


def longest_common_subsequence_ratio(s1: str, s2: str) -> float:
    """
    LCS Ratio (normalized by max length).
    Range: [0, 1]
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0.0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    return lcs_len / max(m, n)


def compute_7_metrics(s1: str, s2: str) -> List[float]:
    """
    Compute all 7 string-similarity metrics (Paper Section 3.5).
    
    Returns:
        List of 7 floats: [lev_norm, jaro, jaro_winkler, lcs_ratio, 
                           lcs_substr_norm, jaro_rev, jaro_winkler_rev]
    """
    if not s1 or not s2:
        return [0.0] * 7
    
    max_len = max(len(s1), len(s2))
    
    # 1. Normalized Levenshtein
    lev_norm = normalized_levenshtein(s1, s2)
    
    # 2. Jaro Similarity
    jaro = jaro_similarity(s1, s2)
    
    # 3. Jaro-Winkler Similarity
    jaro_winkler = jaro_winkler_similarity(s1, s2)
    
    # 4. LCS Ratio
    lcs_ratio = longest_common_subsequence_ratio(s1, s2)
    
    # 5. Longest Common Substring (normalized)
    lcs_substr = longest_common_substring(s1, s2)
    lcs_substr_norm = lcs_substr / max_len if max_len > 0 else 0.0
    
    # 6. Jaro Similarity (reversed strings)
    jaro_rev = jaro_similarity(s1[::-1], s2[::-1])
    
    # 7. Jaro-Winkler Similarity (reversed strings)
    jaro_winkler_rev = jaro_winkler_similarity(s1[::-1], s2[::-1])
    
    return [lev_norm, jaro, jaro_winkler, lcs_ratio, 
            lcs_substr_norm, jaro_rev, jaro_winkler_rev]


def compute_window_features(subdomains: List[str]) -> np.ndarray:
    """
    Compute MEAN of 7 metrics over all subdomain pairs in window.
    
    Args:
        subdomains: List of subdomain strings
    
    Returns:
        NumPy array of shape (7,) with mean feature values
    """
    if len(subdomains) < 2:
        return np.zeros(7)
    
    # Build all pairwise combinations
    pairs = list(combinations(subdomains, 2))
    
    if not pairs:
        return np.zeros(7)
    
    all_metrics = []
    for s1, s2 in pairs:
        metrics = compute_7_metrics(s1, s2)
        all_metrics.append(metrics)
    
    mean_metrics = np.mean(all_metrics, axis=0)
    
    return mean_metrics


class DomainWindowManager:
    """
    Manages sliding windows for all domains.
    Tracks buffers per registered domain.
    """
    
    def __init__(self):
        # Dictionary: domain -> DomainWindowBuffer
        self.buffers: Dict[str, DomainWindowBuffer] = {}
    
    def add_query(self, domain: str, subdomain: str) -> Tuple[bool, np.ndarray]:
        """
        Add a DNS query to the appropriate domain buffer.
        
        Args:
            domain: Registered domain (e.g., "tunnel.com")
            subdomain: Subdomain part (e.g., "abc123")
        
        Returns:
            Tuple[bool, np.ndarray]:
                - bool: True if window is ready for classification
                - np.ndarray: Feature vector (7 floats) if ready, else empty array
        """
        if domain not in self.buffers:
            self.buffers[domain] = DomainWindowBuffer(domain)
        
        buffer = self.buffers[domain]
        
        window_ready = buffer.add_subdomain(subdomain)
        
        if window_ready:
            window_subdomains = buffer.get_window()
            features = compute_window_features(window_subdomains)
            return True, features
        
        return False, np.array([])