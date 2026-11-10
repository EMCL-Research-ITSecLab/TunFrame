"""Detector-local settings; the original framework needs no configuration changes."""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'model-0.joblib'
# The paper calls this DNS packet length. Verify the capture-layer convention
# before claiming exact reproduction of its reported experimental numbers.
LENGTH_LAYER = 'dns'
