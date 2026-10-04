"""Tests for the run script's output path."""

from pathlib import Path

import pytest

from jevtools.scripts.run import results_path


def test_full_run_path_names_the_split_and_run() -> None:
    """Each split and repeat owns its own file."""
    assert results_path("multiple", "spec", 2, None) == Path(
        "results/multiple-spec-run2.jsonl"
    )


def test_limited_run_does_not_overwrite_the_full_results() -> None:
    """A smoke run writes to its own file."""
    assert results_path("simple", "words", 1, 20) == Path(
        "results/simple-words-run1-first20.jsonl"
    )


@pytest.mark.parametrize("limit", [0, -5])
def test_non_positive_limit_is_rejected(limit: int) -> None:
    """A limit that selects nothing, or drops from the end, is an error."""
    with pytest.raises(ValueError, match="positive"):
        results_path("simple", "words", 1, limit)
