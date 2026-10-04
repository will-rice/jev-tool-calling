"""Tests for loading saved runs."""

from pathlib import Path

import pytest

from jevtools.config import RUNS, SPLITS
from jevtools.models import Example, Prediction
from jevtools.scripts.report import load_runs
from jevtools.scripts.run import results_path


def test_load_runs_reads_every_split_of_each_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run is all three splits' predictions together."""
    monkeypatch.chdir(tmp_path)
    Path("results").mkdir()
    for run in range(1, RUNS["test"] + 1):
        for split in SPLITS["test"]:
            prediction = Prediction(
                example=Example(
                    id=f"{split}_0", split=split, query="q", functions=(), gold=None
                ),
                method="words",
                call=None,
                tool_probabilities={},
                choices={},
                nouls={},
                tool_input_tokens=1,
                argument_input_tokens=0,
                model="jev-1.13.0",
            )
            results_path(split, "words", run, None).write_text(
                prediction.model_dump_json() + "\n"
            )
    runs = load_runs("test", "words")
    assert len(runs) == RUNS["test"]
    assert [p.example.id for p in runs[0]] == [
        "simple_0",
        "multiple_0",
        "irrelevance_0",
    ]
