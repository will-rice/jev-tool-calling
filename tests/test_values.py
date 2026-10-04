"""Tests for finding and assembling values in code."""

import pytest

from jevtools.values import complete_place, format_date, number_candidates


def test_number_candidates_offer_each_number_as_written() -> None:
    """Every number in the request is a candidate, in order, once."""
    assert number_candidates("sides 3, 4 and 3", "integer") == {"3": None, "4": None}


def test_number_candidates_normalise_common_written_forms() -> None:
    """Separators, magnitudes, percentages, and number words are also read."""
    candidates = number_candidates(
        "$1.2M for 200,000 units at 40%, in three lots", "float"
    )
    assert candidates == {
        "1.2": None,
        "1200000.0": "1.2M",
        "200000.0": "200,000",
        "40.0": None,
        "0.4": "40%",
        "3.0": "three",
    }


def test_number_candidates_do_not_read_a_spaced_letter_as_a_magnitude() -> None:
    """In "5 m" the m is a unit; in "5m" or "5 million" it is a magnitude."""
    assert number_candidates("5 m wide", "float") == {"5.0": None}
    assert "5000000.0" in number_candidates("5 million wide", "float")


def test_number_candidates_for_an_integer_leave_out_fractions() -> None:
    """An integer argument is only offered whole numbers."""
    assert number_candidates("2.5 kg for 12 people", "integer") == {"12": None}


def test_format_date_writes_the_parts_in_the_functions_format() -> None:
    """Code, not the model, puts the date in the wanted format."""
    assert format_date("March", "5", "2023", "%Y-%m-%d") == "2023-03-05"
    assert format_date("March", "5", "2023", "%m/%d/%Y") == "03/05/2023"


def test_format_date_needs_only_the_parts_the_format_uses() -> None:
    """A month-and-year format does not need a day."""
    assert format_date("March", "none", "2023", "%m-%Y") == "03-2023"
    assert format_date("March", "none", "none", "%m-%Y") is None
    assert format_date("none", "none", "2023", "%m-%Y") is None


def test_format_date_gives_nothing_for_a_missing_or_impossible_date() -> None:
    """A part the request does not give, or February 30, yields no date."""
    assert format_date("March", "none", "2023", "%Y-%m-%d") is None
    assert format_date("February", "30", "2023", "%Y-%m-%d") is None


@pytest.mark.parametrize(
    ("city", "state", "country", "place_format", "no_state", "place"),
    [
        ("Austin", "TX", "United States", "city, state_abbr", "city", "Austin, TX"),
        ("Austin", "TX", "none", "city, state_name", "city", "Austin, Texas"),
        (
            "Paris",
            "none",
            "France",
            "city, state_abbr",
            "city, country",
            "Paris, France",
        ),
        ("Paris", "none", "France", "city, state_abbr", "city", "Paris"),
        ("Paris", "none", "France", "city, country", "city, country", "Paris, France"),
        ("Austin", "TX", "United States", "city", "city", "Austin"),
        ("Austin, TX", "CA", "United States", "city, state_abbr", "city", "Austin, TX"),
    ],
)
def test_complete_place_follows_the_functions_format(
    city: str, state: str, country: str, place_format: str, no_state: str, place: str
) -> None:
    """A city is completed with its state or country as the function asks."""
    assert complete_place(city, state, country, place_format, no_state) == place
