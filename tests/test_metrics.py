"""Tests for scoring."""

import pytest

from jevtools.data import load_examples
from jevtools.metrics import (
    accepts,
    evaluate,
    expressible,
    matches,
    reachable,
    summarize,
)
from jevtools.models import Call, Example, Function, Gold, Parameter, Prediction

CITY = Parameter(name="city", type="string", description="City.", required=True)
SPEED = Parameter(name="speed", type="float", description="Speed.", required=False)
STOPS = Parameter(
    name="stops", type="array", description="Stops.", required=False, item_type="string"
)
EXTRA = Parameter(name="extra", type="dict", description="Extra.", required=False)
ROUTE = Function(
    name="route", description="Plan a route.", parameters=(CITY, SPEED, STOPS, EXTRA)
)
EXAMPLE = Example(
    id="simple_x",
    split="simple",
    query="Drive to New York at 60 via Santa Barbara and Monterey",
    functions=(ROUTE,),
    gold=Gold(
        name="route",
        accepted={
            "city": ["New York", "NYC"],
            "speed": ["", 60.0],
            "stops": [["Santa Barbara", "Monterey"]],
        },
    ),
)
IRRELEVANT = Example(
    id="irrelevance_x", split="irrelevance", query="q", functions=(ROUTE,), gold=None
)
GOOD = Call(
    name="route",
    arguments={"city": "New York", "stops": ["Santa Barbara", "Monterey"]},
)


def predict(example: Example, call: Call | None) -> Prediction:
    """Wrap a call as a saved prediction."""
    return Prediction(
        example=example,
        method="words",
        call=call,
        tool_probabilities={},
        choices={},
        nouls={},
        tool_input_tokens=1,
        argument_input_tokens=1,
        model="jev-1.13.0",
    )


def test_accepts_compares_strings_the_way_bfcl_does() -> None:
    """Case, spaces, and light punctuation do not matter."""
    assert accepts("new-york", ["New York"])
    assert accepts("nyc", ["New York", "NYC"])
    assert not accepts("Newark", ["New York"])


def test_accepts_compares_lists_in_order() -> None:
    """A list matches an accepted list element by element."""
    assert accepts(["santa barbara", "Monterey"], [["Santa Barbara", "Monterey"]])
    assert not accepts(["Monterey", "Santa Barbara"], [["Santa Barbara", "Monterey"]])


def test_matches_a_call_that_omits_an_omittable_argument() -> None:
    """Leaving out an argument the gold marks optional is still right."""
    assert matches(EXAMPLE, GOOD)


def test_matches_rejects_no_call_and_the_wrong_function() -> None:
    """Abstaining or calling another function is wrong when a call is expected."""
    assert not matches(EXAMPLE, None)
    assert not matches(EXAMPLE, Call(name="other", arguments=GOOD.arguments))


def test_matches_rejects_a_missing_required_argument() -> None:
    """A required parameter must be present."""
    assert not matches(EXAMPLE, Call(name="route", arguments={"stops": ["a"]}))


def test_matches_rejects_a_missing_argument_the_gold_does_not_mark_optional() -> None:
    """An argument that is optional in the schema can still be needed."""
    assert not matches(EXAMPLE, Call(name="route", arguments={"city": "NYC"}))


def test_matches_rejects_an_argument_the_gold_does_not_list() -> None:
    """An argument with no accepted values is unexpected."""
    call = Call(name="route", arguments={**GOOD.arguments, "extra": "x"})
    assert not matches(EXAMPLE, call)


def test_matches_rejects_a_wrong_value() -> None:
    """Every value must be one of its accepted values."""
    call = Call(name="route", arguments={**GOOD.arguments, "speed": 70.0})
    assert not matches(EXAMPLE, call)


def test_matches_abstention_on_an_irrelevant_query() -> None:
    """With no gold call, the right answer is no call."""
    assert matches(IRRELEVANT, None)
    assert not matches(IRRELEVANT, GOOD)


def test_expressible_finds_a_value_in_the_query() -> None:
    """A string or number that appears in the query can be labelled."""
    assert expressible(EXAMPLE.query, CITY, ["New York"])
    assert expressible(EXAMPLE.query, SPEED, [60.0])
    assert not expressible(EXAMPLE.query, CITY, ["Boston"])


def test_expressible_needs_every_element_of_an_accepted_list() -> None:
    """An array is reachable only if each element is in the query."""
    assert expressible(EXAMPLE.query, STOPS, [["Santa Barbara", "Monterey"]])
    assert not expressible(EXAMPLE.query, STOPS, [["Santa Barbara", "Salinas"]])


def test_expressible_is_false_for_a_type_that_is_never_asked() -> None:
    """A dict argument cannot be filled by labelling words."""
    assert not expressible(EXAMPLE.query, EXTRA, [{"a": [1]}])


def test_reachable_requires_every_needed_argument() -> None:
    """An example counts toward the ceiling only if nothing is out of reach."""
    assert reachable(EXAMPLE)
    out_of_reach = EXAMPLE.model_copy(
        update={"gold": Gold(name="route", accepted={"city": ["Boston"]})}
    )
    assert not reachable(out_of_reach)


def test_ceiling_on_the_real_data() -> None:
    """With these words and no conversion, 305 and 159 queries are reachable."""
    assert sum(reachable(example) for example in load_examples("simple")) == 305
    assert sum(reachable(example) for example in load_examples("multiple")) == 159


def test_evaluate_reports_each_metric_per_split() -> None:
    """One right call, one call with a wrong value, and one right abstention."""
    wrong = Call(name="route", arguments={**GOOD.arguments, "city": "Boston"})
    metrics = evaluate(
        [predict(EXAMPLE, GOOD), predict(EXAMPLE, wrong), predict(IRRELEVANT, None)]
    )
    assert metrics == {
        "simple/call_accuracy": 0.5,
        "simple/tool_accuracy": 1.0,
        "simple/argument_accuracy": 0.75,
        "simple/ceiling": 1.0,
        "irrelevance/call_accuracy": 1.0,
    }


def test_evaluate_rejects_an_empty_run() -> None:
    """Scoring nothing is an error, not a score of zero."""
    with pytest.raises(ValueError, match="No predictions"):
        evaluate([])


def test_evaluate_rejects_a_split_where_no_tool_was_right() -> None:
    """Argument accuracy over no arguments is an error, not zero."""
    with pytest.raises(ValueError, match="No arguments to score"):
        evaluate([predict(EXAMPLE, None)])


def test_summarize_gives_mean_and_range_per_metric() -> None:
    """Runs are summarised by their mean, minimum, and maximum."""
    assert summarize([{"m": 0.2}, {"m": 0.4}]) == {"m": (pytest.approx(0.3), 0.2, 0.4)}
