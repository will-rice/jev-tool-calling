"""Constants for the Jev BFCL benchmark."""

from pathlib import Path

from jevtools.models import Split, Stage

DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DATASET_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
MODEL = "jev-latest"
MAX_WORKERS = 8
SPLITS: dict[Stage, tuple[Split, ...]] = {
    "dev": ("live_simple",),
    "test": (
        "simple",
        "multiple",
        "irrelevance",
        "live_multiple",
        "live_irrelevance",
        "live_relevance",
        "parallel",
        "parallel_multiple",
        "live_parallel",
        "live_parallel_multiple",
    ),
}
RUNS: dict[Stage, int] = {"dev": 1, "test": 3}
UNANSWERED: tuple[Split, ...] = ("irrelevance", "live_irrelevance", "live_relevance")
PARALLEL: tuple[Split, ...] = (
    "parallel",
    "parallel_multiple",
    "live_parallel",
    "live_parallel_multiple",
)
MAX_GAP = 2
THRESHOLD = 0.5
LABEL_THRESHOLD = 0.7
BRIDGE_THRESHOLD = 0.3
MAX_BRIDGE = 6
NONE = "none"
# A word is a number, a run of letters, or one punctuation mark. A number
# keeps its thousands separators, so "1,000" is one word that does not parse
# rather than a "1". A sign is part of a number only when no word, digit, or
# closing bracket precedes it, so "5-10" is two numbers and a hyphen.
WORD_PATTERN = (
    r"(?:(?<![\w)\]])[-+])?"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d[\d.]*(?:[eE][-+]?\d+)?)"
    r"|\w+|[^\w\s]"
)
MAX_OPTIONS = 255
CONTEXT_WORDS = 95
BATCH_CHARACTERS = 120_000
RESULTS_DIR = Path("results")
SPEC_DIR = Path("specs")
