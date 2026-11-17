from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MAX_LENGTH = 256  # Authors' validation/test encoding (training uses 64).

# Keep the original model as default. Set VARIANT='improved' for a separately
# fine-tuned checkpoint; never replace the authors' original weights.
VARIANT = 'authors'
IMPROVED_MODEL_DIR = Path(__file__).resolve().parent / 'model-improved'
IMPROVED_MAX_LENGTH = 64
