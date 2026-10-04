"""Score the saved full runs and report each metric's mean and range."""

import logging
from typing import get_args

from jevtools.config import RUNS
from jevtools.metrics import evaluate, summarize
from jevtools.models import Prediction, Split
from jevtools.scripts.run import results_path


def main() -> None:
    """Log every metric as a percentage with its range across runs."""
    logging.basicConfig(level=logging.INFO)
    scores = [evaluate(run) for run in load_runs()]
    for name, (mean, low, high) in summarize(scores).items():
        logging.info("%s: %.1f (%.1f–%.1f)", name, 100 * mean, 100 * low, 100 * high)


def load_runs() -> list[list[Prediction]]:
    """Load each full run's predictions, all splits together."""
    return [
        [
            Prediction.model_validate_json(line)
            for split in get_args(Split)
            for line in results_path(split, run, None).read_text().splitlines()
        ]
        for run in range(1, RUNS + 1)
    ]


if __name__ == "__main__":
    main()
