"""Score the saved full runs and report each metric's mean and range."""

import argparse
import logging
from typing import get_args

from jevtools.config import RUNS, SPLITS
from jevtools.metrics import evaluate, summarize
from jevtools.models import Method, Prediction, Stage
from jevtools.scripts.run import results_path


def main() -> None:
    """Log every metric as a percentage with its range across runs."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=get_args(Stage), help="Which runs to score.")
    parser.add_argument("method", choices=get_args(Method), help="Which method.")
    args = parser.parse_args()
    scores = [evaluate(run) for run in load_runs(args.stage, args.method)]
    for name, (mean, low, high) in summarize(scores).items():
        logging.info("%s: %.1f (%.1f–%.1f)", name, 100 * mean, 100 * low, 100 * high)


def load_runs(stage: Stage, method: Method) -> list[list[Prediction]]:
    """Load each full run of a method on a stage, all its splits together."""
    return [
        [
            Prediction.model_validate_json(line)
            for split in SPLITS[stage]
            for line in results_path(split, method, run, None).read_text().splitlines()
        ]
        for run in range(1, RUNS[stage] + 1)
    ]


if __name__ == "__main__":
    main()
