from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / 'model'
VARIANT = 'rf_cnn'  # rf, cnn, or rf_cnn
THRESHOLD = 0.5  # The paper's separately tuned live RF threshold is 0.8.
MAX_GROUPS = 10000
MAX_QUERIES_PER_GROUP = 100000
