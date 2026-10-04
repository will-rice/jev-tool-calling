"""Score the saved full runs and report each metric's mean and range."""

import argparse
import logging
from typing import get_args

from jevtools.config import RUNS, SPLITS
from jevtools.metrics import evaluate, summarize
from jevtools.models import Prediction, Stage
from jevtools.scripts.run import results_path


def main() -> None:
    """Log every metric as a percentage with its range across runs."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=get_args(Stage), help="Which runs to score.")
    scores = [evaluate(run) for run in load_runs(parser.parse_args().stage)]
    for name, (mean, low, high) in summarize(scores).items():
        logging.info("%s: %.1f (%.1f–%.1f)", name, 100 * mean, 100 * low, 100 * high)


def load_runs(stage: Stage) -> list[list[Prediction]]:
    """Load each full run's predictions for a stage, all its splits together."""
    return [
        [
            Prediction.model_validate_json(line)
            for split in SPLITS[stage]
            for line in results_path(split, run, None).read_text().splitlines()
        ]
        for run in range(1, RUNS[stage] + 1)
    ]


if __name__ == "__main__":
    main()
