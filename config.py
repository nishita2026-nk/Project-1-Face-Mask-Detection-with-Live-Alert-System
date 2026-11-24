
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset directories
DATASET_DIR = PROJECT_ROOT / "dataset"

WITH_MASK_DIR = DATASET_DIR / "with_mask"
WITHOUT_MASK_DIR = DATASET_DIR / "without_mask"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"

# Image configuration
IMAGE_SIZE = (128, 128)
IMAGE_CHANNELS = 3

# Class mapping
CLASS_MAPPING = {
    "without_mask": 0,
    "with_mask": 1
}

# Dataset split configuration
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Reproducibility
RANDOM_SEED = 42

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Model configuration
CONFIDENCE_THRESHOLD = 0.70

# Alert configuration
ALERT_COOLDOWN = 5

# Create required output directories
for directory in [
    RESULTS_DIR,
    MODELS_DIR,
    DATABASE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project configuration loaded successfully.")
    print("Project root:", PROJECT_ROOT)
    print("Dataset directory:", DATASET_DIR)
    print("Image size:", IMAGE_SIZE)
    print("Class mapping:", CLASS_MAPPING)
