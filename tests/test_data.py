"""Tests for data loading against the real dataset."""

from collections import Counter

from jevtools.config import SPLITS
from jevtools.data import load_examples
from jevtools.jev import word_spans

DEV_SIZE = 233


def test_splits_parse_to_their_known_sizes() -> None:
    """The pinned revision has 400, 200, and 240 queries."""
    assert len(load_examples("simple")) == 400
    assert len(load_examples("multiple")) == 200
    assert len(load_examples("irrelevance")) == 240


def test_dev_split_keeps_only_rows_shaped_like_the_test_splits() -> None:
    """Rows with a system message, a non-string enum, or over 95 words are left out."""
    examples = load_examples("live_simple")
    assert len(examples) == DEV_SIZE
    assert all(example.gold is not None for example in examples)
    assert max(len(word_spans(example.query)) for example in examples) <= 95


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
        for split in SPLITS["test"]
        for example in load_examples(split)
        for function in example.functions
        for parameter in function.parameters
    )
    assert kinds == {"words": 3016, "choice": 111, "flag": 137, "set": 74, None: 37}
