"""Tests for the constants that shape the benchmark."""

import re

from jevtools.config import WORD_PATTERN


def test_word_pattern_splits_punctuation_from_values() -> None:
    """A value is never glued to the bracket, comma, or symbol beside it."""
    words = re.findall(WORD_PATTERN, "A(3,4) costs $1e-9 or -2.5 units.")
    assert words == [
        "A", "(", "3", ",", "4", ")", "costs", "$", "1e-9", "or", "-2.5", "units", ".",
    ]  # fmt: skip


def test_word_pattern_keeps_a_thousands_number_whole() -> None:
    """A number with separators is one word, so it is never read as its first group."""
    assert re.findall(WORD_PATTERN, "pay 1,200,000 or (3,4)") == [
        "pay", "1,200,000", "or", "(", "3", ",", "4", ")",
    ]  # fmt: skip


def test_word_pattern_reads_a_hyphen_between_numbers_as_a_separator() -> None:
    """A sign belongs to a number only when nothing it could separate precedes it."""
    assert re.findall(WORD_PATTERN, "from 5-10 at x=-3") == [
        "from", "5", "-", "10", "at", "x", "=", "-3",
    ]  # fmt: skip
