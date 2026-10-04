"""Constants for the Jev BFCL benchmark."""

from pathlib import Path

DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DATASET_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
MODEL = "jev-latest"
MAX_WORKERS = 8
RUNS = 3
MAX_GAP = 2
THRESHOLD = 0.5
NONE = "none"
WORD_PATTERN = r"[-+]?\d[\d.]*(?:[eE][-+]?\d+)?|\w+|[^\w\s]"
RESULTS_DIR = Path("results")
