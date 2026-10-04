"""Tests for the constants that shape the benchmark."""

import re

from jevtools.config import WORD_PATTERN


def test_word_pattern_splits_punctuation_from_values() -> None:
    """A value is never glued to the bracket, comma, or symbol beside it."""
    words = re.findall(WORD_PATTERN, "A(3,4) costs $1e-9 or -2.5 units.")
    assert words == [
        "A", "(", "3", ",", "4", ")", "costs", "$", "1e-9", "or", "-2.5", "units", ".",
    ]  # fmt: skip
