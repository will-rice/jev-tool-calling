"""Constants for the Jev BFCL benchmark."""

from pathlib import Path

from jevtools.models import Split, Stage

DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DATASET_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
MODEL = "jev-latest"
MAX_WORKERS = 8
SPLITS: dict[Stage, tuple[Split, ...]] = {
    "dev": ("live_simple",),
    "test": ("simple", "multiple", "irrelevance"),
}
RUNS: dict[Stage, int] = {"dev": 1, "test": 3}
MAX_WORDS = 95
MAX_GAP = 2
THRESHOLD = 0.5
LABEL_THRESHOLD = 0.7
NONE = "none"
WORD_PATTERN = r"[-+]?\d[\d.]*(?:[eE][-+]?\d+)?|\w+|[^\w\s]"
RESULTS_DIR = Path("results")
