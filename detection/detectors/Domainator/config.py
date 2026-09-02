#!/usr/bin/env python3

import os
from pathlib import Path

# Path to trained Random Forest model (sklearn pkl file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "domainator_model.pkl")


# Window size: Number of DNS queries per window
WINDOW_SIZE = 10

# Step size: Non-overlapping windows
STEP_SIZE = 10

# Minimum queries required to compute features
MIN_WINDOW_SIZE = 3


FEATURE_NAMES = [
    'levenshtein_norm',
    'jaro',
    'jaro_winkler',
    'lcs_ratio',
    'lcs_substr_norm',
    'jaro_reversed',
    'jaro_winkler_reversed'
]

DETECTION_THRESHOLD = 0.5
VERBOSE = False