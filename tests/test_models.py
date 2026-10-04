"""Tests for the data models."""

import pytest
from pydantic import ValidationError

from jevtools.models import Call, Example, Function, Gold, Parameter, Prediction

AREA = Function(
    name="triangle.area",
    description="Area of a triangle.",
    parameters=(
        Parameter(name="base", type="integer", description="The base.", required=True),
    ),
)


@pytest.mark.parametrize(
    ("parameter", "kind"),
    [
        (Parameter(name="p", type="boolean", description=None, required=True), "flag"),
        (
            Parameter(
                name="p", type="string", description=None, required=True, enum=("a",)
            ),
            "choice",
        ),
        (
            Parameter(
                name="p",
                type="array",
                description=None,
                required=True,
                enum=("a",),
                item_type="string",
            ),
            "set",
        ),
        (Parameter(name="p", type="float", description=None, required=True), "words"),
        (
            Parameter(
                name="p",
                type="array",
                description=None,
                required=True,
                item_type="integer",
            ),
            "words",
        ),
        (Parameter(name="p", type="dict", description=None, required=True), None),
        (
            Parameter(
                name="p",
                type="array",
                description=None,
                required=True,
                item_type="dict",
            ),
            None,
        ),
    ],
)
def test_kind_follows_the_schema(parameter: Parameter, kind: str | None) -> None:
    """How a parameter is asked is decided by its type and enum alone."""
    assert parameter.kind == kind


def test_value_type_of_an_array_is_its_item_type() -> None:
    """Array elements are coerced by the item type, scalars by their own."""
    array = Parameter(
        name="p", type="array", description=None, required=True, item_type="float"
    )
    assert array.value_type == "float"
    assert AREA.parameters[0].value_type == "integer"


def test_gold_must_name_an_offered_function() -> None:
    """An example whose answer calls a function it does not offer is invalid."""
    with pytest.raises(ValidationError, match="not offered"):
        Example(
            id="x",
            split="simple",
            query="q",
            functions=(AREA,),
            gold=(Gold(name="other", accepted={}),),
        )


def test_target_returns_the_gold_function_and_accepted_values() -> None:
    """Scoring reads the gold function and its accepted values together."""
    example = Example(
        id="x",
        split="simple",
        query="q",
        functions=(AREA,),
        gold=(Gold(name="triangle.area", accepted={"base": [10]}),),
    )
    assert example.target == (AREA, {"base": [10]})


def test_target_raises_when_there_is_no_gold_call() -> None:
    """An irrelevance example has nothing to score arguments against."""
    example = Example(
        id="x", split="irrelevance", query="q", functions=(AREA,), gold=()
    )
    with pytest.raises(ValueError, match="no gold call"):
        _ = example.target


def test_prediction_survives_a_json_round_trip() -> None:
    """Saved records reload unchanged, with value types intact."""
    prediction = Prediction(
        example=Example(
            id="x",
            split="simple",
            query="base 10",
            functions=(AREA,),
            gold=(
                Gold(name="triangle.area", accepted={"base": [10], "unit": ["", "cm"]}),
            ),
        ),
        method="spec",
        calls=(
            Call(
                name="triangle.area", arguments={"base": 10, "ratio": 2.5, "on": True}
            ),
        ),
        tool_probabilities={"triangle.area": 0.9, "none": 0.1},
        choices={"token_0": {"base": 0.2, "none": 0.8}},
        nouls={"flag.on": 0.7},
        tool_input_tokens=100,
        argument_input_tokens=200,
        model="jev-1.13.0",
    )
    reloaded = Prediction.model_validate_json(prediction.model_dump_json())
    assert reloaded == prediction
    assert type(reloaded.calls[0].arguments["base"]) is int
    assert type(reloaded.calls[0].arguments["on"]) is bool
