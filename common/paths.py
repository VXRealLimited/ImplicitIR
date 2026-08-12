from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
SPLITS_DIR = DATA_DIR / "sft_splits"

OUTPUTS_DIR = REPO_ROOT / "outputs"
