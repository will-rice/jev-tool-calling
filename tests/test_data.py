"""Tests for data loading against the real dataset."""

from collections import Counter
from typing import get_args

import pytest

from jevtools.data import load_examples
from jevtools.models import Split


def test_splits_parse_to_their_known_sizes() -> None:
    """The pinned revision has 400, 200, and 240 queries."""
    assert len(load_examples("simple")) == 400
    assert len(load_examples("multiple")) == 200
    assert len(load_examples("irrelevance")) == 240


@pytest.mark.parametrize(
    ("split", "size"),
    [
        ("live_simple", 258),
        ("live_multiple", 1053),
        ("live_irrelevance", 882),
        ("live_relevance", 18),
    ],
)
def test_live_splits_keep_every_row(split: Split, size: int) -> None:
    """No row is left out, whatever its shape: a skipped row would not be scored."""
    assert len(load_examples(split)) == size


def test_earlier_messages_are_kept_as_context() -> None:
    """The query is the last user message; anything before it is context."""
    examples = load_examples("live_multiple")
    with_system = next(e for e in examples if e.context)
    assert with_system.context[0][0] == "system"
    assert with_system.query
    assert all(e.context == () for e in load_examples("simple"))


def test_number_enums_are_kept_as_text_options() -> None:
    """An enum of numbers is offered as its values' text and still a choice."""
    parameters = [
        parameter
        for example in load_examples("live_multiple")
        for function in example.functions
        for parameter in function.parameters
        if parameter.type == "integer" and parameter.enum
    ]
    assert parameters
    assert all(parameter.kind == "choice" for parameter in parameters)
    assert all(isinstance(value, str) for p in parameters for value in p.enum)


def test_only_answerable_splits_have_gold_calls() -> None:
    """Irrelevance and relevance splits are scored on whether a call was made."""
    assert all(e.gold is None for e in load_examples("live_irrelevance"))
    assert all(e.gold is None for e in load_examples("live_relevance"))
    assert all(e.gold is not None for e in load_examples("live_multiple"))


def test_simple_example_carries_its_query_function_and_parameters() -> None:
    """A row parses into the query, the offered function, and typed parameters."""
    example = load_examples("simple")[0]
    assert example.id == "simple_0"
    assert example.query == (
        "Find the area of a triangle with a base of 10 units and height of 5 units."
    )
    [function] = example.functions
    assert function.name == "calculate_triangle_area"
    assert [(p.name, p.type, p.required) for p in function.parameters] == [
        ("base", "integer", True),
        ("height", "integer", True),
        ("unit", "string", False),
    ]


def test_multiple_example_carries_accepted_values_and_omittable_arguments() -> None:
    """Gold keeps every accepted value, with "" marking an omittable argument."""
    example = load_examples("multiple")[0]
    assert example.gold is not None
    assert example.gold.name == "triangle_properties.get"
    assert example.gold.accepted["side1"] == [5]
    assert example.gold.accepted["get_area"] == ["", True]
    assert len(example.functions) > 1


def test_simple_gold_is_the_one_offered_function() -> None:
    """A gold key without the module prefix resolves to the offered function."""
    example = next(e for e in load_examples("simple") if e.id == "simple_363")
    assert example.gold is not None
    assert example.gold.name == "restaurant_search.find_closest"


def test_irrelevance_examples_have_no_gold_call() -> None:
    """No offered function fits an irrelevance query."""
    assert all(example.gold is None for example in load_examples("irrelevance"))


def test_parameter_kinds_across_all_splits() -> None:
    """Almost every parameter is open-valued; 37 cannot be filled at all."""
    kinds = Counter(
        parameter.kind
        for split in get_args(Split)[:3]
        for example in load_examples(split)
        for function in example.functions
        for parameter in function.parameters
    )
    assert kinds == {"words": 3016, "choice": 111, "flag": 137, "set": 74, None: 37}
