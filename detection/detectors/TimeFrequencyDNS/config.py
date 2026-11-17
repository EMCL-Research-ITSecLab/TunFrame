from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'window-300.joblib'
VARIANT = 'time_frequency'  # time or time_frequency
MAX_PACKETS = 1000000
