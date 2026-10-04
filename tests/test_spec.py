"""Tests for function specs."""

import pytest

from jevtools.config import SPLITS
from jevtools.data import load_examples
from jevtools.models import Function, Parameter, Stage
from jevtools.spec import (
    FunctionSpec,
    ParameterSpec,
    check,
    default_spec,
    load_specs,
)

WEATHER = Function(
    name="weather",
    description="Get the weather.",
    parameters=(
        Parameter(name="city", type="string", description="City.", required=True),
        Parameter(name="days", type="integer", description="Days.", required=False),
        Parameter(
            name="unit",
            type="string",
            description="Unit.",
            required=False,
            enum=("C", "F"),
        ),
        Parameter(name="hourly", type="boolean", description="Hourly.", required=True),
        Parameter(
            name="fields",
            type="array",
            description="Fields.",
            required=False,
            enum=("wind", "rain"),
            item_type="string",
        ),
        Parameter(name="extra", type="dict", description="Extra.", required=False),
    ),
)


def test_default_spec_asks_each_parameter_the_way_its_schema_implies() -> None:
    """With no authored spec, open values are text and closed sets are asked."""
    spec = default_spec(WEATHER)
    assert {name: entry.kind for name, entry in spec.parameters.items()} == {
        "city": "text",
        "days": "text",
        "unit": "options",
        "hourly": "flag",
        "fields": "set",
        "extra": "skip",
    }
    assert spec.parameters["unit"].options == {"C": None, "F": None}
    assert spec.parameters["unit"].stated is not None
    assert spec.parameters["hourly"].stated is None
    check(WEATHER, spec)


def test_check_rejects_a_spec_that_misses_a_parameter() -> None:
    """Every parameter needs an entry, or it would silently go unfilled."""
    spec = FunctionSpec(name="weather", parameters={})
    with pytest.raises(ValueError, match="city"):
        check(WEATHER, spec)


@pytest.mark.parametrize(
    ("name", "entry"),
    [
        ("city", ParameterSpec(kind="number", question="q")),
        ("days", ParameterSpec(kind="date", question="q", format="%Y")),
        ("hourly", ParameterSpec(kind="text", question="q")),
        ("extra", ParameterSpec(kind="text", question="q")),
        ("city", ParameterSpec(kind="options", question="q")),
        ("city", ParameterSpec(kind="date", question="q")),
        ("city", ParameterSpec(kind="place", question="q")),
    ],
)
def test_check_rejects_a_kind_that_does_not_fit_the_parameter(
    name: str, entry: ParameterSpec
) -> None:
    """A kind the parameter's type cannot take, or missing its fields, is an error."""
    spec = default_spec(WEATHER)
    broken = spec.model_copy(update={"parameters": {**spec.parameters, name: entry}})
    with pytest.raises(ValueError, match=name):
        check(WEATHER, broken)


def test_check_rejects_more_options_than_one_question_can_offer() -> None:
    """A Choice takes at most 255 options, and one is kept for none."""
    entry = ParameterSpec(
        kind="options", question="q", options={str(i): None for i in range(255)}
    )
    spec = default_spec(WEATHER)
    broken = spec.model_copy(update={"parameters": {**spec.parameters, "city": entry}})
    with pytest.raises(ValueError, match="254"):
        check(WEATHER, broken)


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_authored_specs_fit_every_offered_function(stage: Stage) -> None:
    """Each function a stage offers has an authored spec that fits its schema."""
    specs = load_specs(stage)
    functions = {
        function.key: function
        for split in SPLITS[stage]
        for example in load_examples(split)
        for function in example.functions
    }
    assert functions.keys() <= specs.keys()
    for key, function in functions.items():
        check(function, specs[key])


def test_function_key_changes_with_the_definition() -> None:
    """Two functions with one name but different parameters get different specs."""
    other = WEATHER.model_copy(update={"description": "Get tomorrow's weather."})
    assert WEATHER.key != other.key
    assert WEATHER.key == WEATHER.model_copy().key
