"""Evaluate Jev zero-shot on BFCL tool calling."""

import argparse
import logging
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv
from tqdm.contrib.concurrent import thread_map
from typesafe_sdk import TypeSafeClient

from jevtools.config import MAX_WORKERS, RESULTS_DIR, RUNS
from jevtools.data import load_examples
from jevtools.jev import predict
from jevtools.metrics import evaluate, summarize
from jevtools.models import Split


def main() -> None:
    """Run every split several times, save each run, and log the metrics."""
    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx2", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N queries of each split."
    )
    args = parser.parse_args()
    splits = get_args(Split)
    paths = {
        (split, run): results_path(split, run, args.limit)
        for run in range(1, RUNS + 1)
        for split in splits
    }
    load_dotenv()

    examples = {split: load_examples(split)[: args.limit] for split in splits}
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = []
    with TypeSafeClient() as client:
        for run in range(1, RUNS + 1):
            predictions = []
            for split in splits:
                rows = thread_map(
                    lambda example: predict(client, example),
                    examples[split],
                    max_workers=MAX_WORKERS,
                    desc=f"{split} run {run}",
                )
                path = paths[split, run]
                path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
                logging.info("Wrote %d predictions to %s", len(rows), path)
                predictions.extend(rows)
            logging.info("Model versions: %s", sorted({p.model for p in predictions}))
            runs.append(
                evaluate(predictions)
                | {
                    "tool/input_tokens": sum(p.tool_input_tokens for p in predictions),
                    "argument/input_tokens": sum(
                        p.argument_input_tokens for p in predictions
                    ),
                }
            )

    for name, (mean, low, high) in summarize(runs).items():
        logging.info("%s: mean %.4f, range %.4f to %.4f", name, mean, low, high)


def results_path(split: Split, run: int, limit: int | None) -> Path:
    """Return where one run of a split saves its predictions.

    A limited run gets its own file so it cannot overwrite a full run.

    Raises:
        ValueError: If the limit is not positive.
    """
    stem = f"{split}-run{run}"
    if limit is None:
        return RESULTS_DIR / f"{stem}.jsonl"
    if limit < 1:
        raise ValueError(f"--limit must be positive, got {limit}")
    return RESULTS_DIR / f"{stem}-first{limit}.jsonl"


if __name__ == "__main__":
    main()
