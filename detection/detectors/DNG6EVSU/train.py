import pickle
import csv
from datetime import datetime
from typing import Tuple, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from detection.detectors.DNG6EVSU.matchgram import MatchGram

BASE_DIR = os.path.dirname(__file__)


DATASET_PATH = "detection/detectors/DNG6EVSU/Dataset-2/training.csv"
N_GRAM_SIZE = 1

# --- Constants ---
# Change MODEL_DIR to be relative to the script, not working directory
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure directory exists

# --- Helper Functions ---
def load_domains_from_csv(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load domains from CSV with format: `label,domain` (no headers).

    Args:
        filepath: Path to CSV file (e.g., "./Dataset-2/training.csv")

    Returns:
        Tuple of (benign_domains, tunneled_domains)
    """
    benign = []
    tunneled = []
    with open(filepath, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            label, domain = row[0].strip(), row[1].strip().lower()
            if label == '1':
                tunneled.append(domain)
            elif label == '0':
                benign.append(domain)
    return benign, tunneled

def load_domains_from_txt(file_path: str) -> List[str]:
    """Load domains from a text file (one domain per line)."""
    with open(file_path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def generate_model_name(dataset_path: str, n: int, num_domains: int) -> str:
    """
    Generate a descriptive model filename.

    Example: matchgram_Dataset-2_n3_3966domains_20260804.pkl
    """
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    return f"matchgram_{dataset_name}_n{n}_{num_domains}domains.pkl"

def train_model(n: int, domains: List[str], verbose: bool = True) -> MatchGram:
    """
    Train a MatchGram model on the given domains.

    Args:
        n: N-gram size (e.g., 3 for trigrams)
        domains: List of benign domains for training
        verbose: If True, print training progress

    Returns:
        Trained MatchGram model
    """
    model = MatchGram(n)
    total = len(domains)

    if verbose:
        print(f"[INFO] Training MatchGram with n={n} on {total} domains...")
        print(f"[INFO] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for i, domain in enumerate(domains, 1):
        model.add_domain(domain)
        if verbose and i % 1000 == 0:
            print(f"[PROGRESS] Added {i}/{total} domains ({i/total:.1%})")

    if verbose:
        print(f"[SUCCESS] Training completed in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return model

def load_model(file_path: str) -> MatchGram:
    """Load a trained model from disk."""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(file_path: str, model: MatchGram) -> None:
    """Save a trained model to disk."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[SUCCESS] Model saved to: {file_path}")

if __name__ == "__main__":

    # Load data
    print("[INFO] Loading domains from CSV...")
    benign_domains, tunneled_domains = load_domains_from_csv(DATASET_PATH)
    print(f"[INFO] Loaded {len(benign_domains)} benign and {len(tunneled_domains)} tunneled domains")

    # Train model
    model = train_model(
        n=N_GRAM_SIZE,
        domains=benign_domains,
        verbose=True
    )

    # Save model with smart naming
    model_name = generate_model_name(
        dataset_path=DATASET_PATH,
        n=N_GRAM_SIZE,
        num_domains=len(benign_domains)
    )
    model_path = os.path.join(MODEL_DIR, model_name)
    save_model(model_path, model)