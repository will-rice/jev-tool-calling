"""Tests for question building and answer decoding."""

import pytest
from typesafe_sdk import Choice, Noul, SystemOneResponse, Usage

from jevtools.jev import (
    argument_questions,
    coerce,
    decode_arguments,
    input_tokens,
    tool_question,
    word_spans,
    word_values,
)
from jevtools.models import Function, Parameter

AREA = Function(
    name="triangle.area",
    description="Area of a triangle.",
    parameters=(
        Parameter(name="base", type="integer", description="The base.", required=True),
        Parameter(
            name="unit",
            type="string",
            description="The unit.",
            required=False,
            enum=("cm", "in"),
        ),
        Parameter(name="round", type="boolean", description="Round it.", required=True),
        Parameter(
            name="tags",
            type="array",
            description="Tags.",
            required=False,
            enum=("a", "b"),
            item_type="string",
        ),
        Parameter(name="extra", type="dict", description="Extras.", required=False),
    ),
)
PING = Function(name="ping", description="Check the service.", parameters=())


def test_word_spans_give_each_words_offsets() -> None:
    """Offsets let a value be cut from the query exactly as written."""
    query = "base of 10cm."
    assert [query[start:end] for start, end in word_spans(query)] == [
        "base", "of", "10", "cm", ".",
    ]  # fmt: skip


def test_tool_question_offers_each_function_and_a_described_none() -> None:
    """Abstaining is an option with its own description."""
    question = tool_question((AREA, PING))
    assert question.criteria == {
        "triangle.area": "Area of a triangle.",
        "ping": "Check the service.",
        "none": "No offered function can answer the request.",
    }


def test_tool_question_rejects_a_function_named_none() -> None:
    """A function named none would collide with the abstain option."""
    clash = Function(name="none", description="Do nothing.", parameters=())
    with pytest.raises(ValueError, match="collides"):
        tool_question((clash,))


def test_argument_questions_cover_each_kind_once() -> None:
    """Words are labelled; closed sets are asked directly; dicts are skipped."""
    questions = argument_questions("base 10", AREA)
    assert list(questions) == [
        "token_0",
        "token_1",
        "choice.unit",
        "stated.unit",
        "flag.round",
        "member.tags.a",
        "member.tags.b",
    ]
    assert isinstance(questions["token_0"], Choice)
    assert isinstance(questions["choice.unit"], Choice)
    assert isinstance(questions["stated.unit"], Noul)
    assert isinstance(questions["flag.round"], Noul)
    assert isinstance(questions["member.tags.a"], Noul)


def test_token_question_offers_open_parameters_and_a_described_none() -> None:
    """Only open-valued parameters compete for a word, with their descriptions."""
    question = argument_questions("base 10", AREA)["token_1"]
    assert question.criteria == {
        "base": "The base.",
        "none": (
            "The word is not part of any argument's value. A word that only "
            "names or introduces an argument, such as a field name, a "
            "preposition, or punctuation around the value, is none."
        ),
    }


def test_token_question_identifies_a_word_by_the_words_around_it() -> None:
    """A repeated word is told apart by what comes before and after it."""
    questions = argument_questions("10 by 10", AREA)
    question = (
        "Which argument of `function` does `word` supply in `query`? "
        "Answer none if it supplies no argument."
    )
    assert questions["token_0"].instructions == {
        "function": "triangle.area",
        "words_before": "",
        "word": "10",
        "words_after": "by 10",
        "question": question,
    }
    assert questions["token_2"].instructions == {
        "function": "triangle.area",
        "words_before": "10 by",
        "word": "10",
        "words_after": "",
        "question": question,
    }


def test_choice_question_offers_exactly_the_enum_values() -> None:
    """Whatever the model picks is a value the function accepts."""
    question = argument_questions("base 10", AREA)["choice.unit"]
    assert question.criteria == {"cm": None, "in": None}
    assert question.instructions == {
        "function": "triangle.area",
        "argument": "unit",
        "description": "The unit.",
        "question": "Which value of `argument` does `query` ask for?",
    }


def test_required_closed_set_has_no_stated_question() -> None:
    """A required argument is always filled, so nothing asks if it is stated."""
    assert "stated.round" not in argument_questions("base 10", AREA)


def test_member_question_names_its_value() -> None:
    """Each member of a set is asked about separately."""
    question = argument_questions("base 10", AREA)["member.tags.b"]
    assert question.instructions == {
        "function": "triangle.area",
        "argument": "tags",
        "description": "Tags.",
        "value": "b",
        "question": "Does `query` ask for `value` in `argument`?",
    }


def test_function_without_fillable_parameters_has_no_questions() -> None:
    """Nothing is asked, so the caller must skip the request."""
    assert argument_questions("ping the service", PING) == {}


def test_function_without_open_parameters_has_no_token_questions() -> None:
    """Words are labelled only when some parameter can take them."""
    switch = Function(
        name="switch",
        description="Flip a switch.",
        parameters=(
            Parameter(name="on", type="boolean", description="On.", required=True),
        ),
    )
    assert list(argument_questions("turn it on", switch)) == ["flag.on"]


def test_open_parameter_named_none_is_rejected() -> None:
    """A parameter named none would collide with the no-argument option."""
    clash = Function(
        name="f",
        description="F.",
        parameters=(
            Parameter(name="none", type="string", description="N.", required=True),
        ),
    )
    with pytest.raises(ValueError, match="collides"):
        argument_questions("a b", clash)


ROUTE = Function(
    name="route",
    description="Plan a route.",
    parameters=(
        Parameter(name="city", type="string", description="City.", required=True),
        Parameter(
            name="stops",
            type="array",
            description="Stops.",
            required=False,
            item_type="string",
        ),
        Parameter(name="speed", type="float", description="Speed.", required=False),
    ),
)
AREA_NOULS = {
    "stated.unit": 0.9,
    "flag.round": 0.2,
    "member.tags.a": 0.8,
    "member.tags.b": 0.1,
}


def labels(*names: str) -> dict[str, dict[str, float]]:
    """Build certain token answers for a query, one label per word."""
    return {f"token_{index}": {name: 1.0} for index, name in enumerate(names)}


@pytest.mark.parametrize(
    ("text", "value_type", "value"),
    [
        ("10", "integer", 10),
        ("-3", "integer", -3),
        ("4.", "integer", 4),
        ("(5", "integer", 5),
        ("ten", "integer", None),
        ("200,000", "integer", None),
        ("2.5", "integer", None),
        ("2.5", "float", 2.5),
        ("5", "float", 5.0),
        ("1e-9", "float", 1e-9),
        ("$5", "float", 5.0),
        ("fast", "float", None),
        ("'Tech Inc'", "string", "Tech Inc"),
        ("New York,", "string", "New York"),
        ("?", "string", None),
    ],
)
def test_coerce_parses_a_value_or_gives_nothing(
    text: str, value_type: str, value: str | float | None
) -> None:
    """A value that is not literally of its type is dropped, never converted."""
    result = coerce(text, value_type)
    assert result == value
    assert type(result) is type(value)


def test_coerce_rejects_a_type_it_cannot_read_from_words() -> None:
    """Only strings and numbers are cut from the query."""
    with pytest.raises(ValueError, match="dict"):
        coerce("x", "dict")


def test_word_values_join_adjacent_words_as_written() -> None:
    """A value keeps the query's own spacing and punctuation."""
    query = "drive to New-York fast"
    values = word_values(
        query, ROUTE, labels("none", "none", "city", "city", "city", "none")
    )
    assert values == {"city": ["New-York"]}


def test_word_values_fill_short_gaps_inside_a_scalar_value() -> None:
    """Small unlabelled words inside a name stay in it."""
    query = "go to Stratford upon Avon now"
    values = word_values(
        query, ROUTE, labels("none", "none", "city", "none", "city", "none")
    )
    assert values == {"city": ["Stratford upon Avon"]}


def test_word_values_keep_array_elements_apart() -> None:
    """Gaps are not filled for an array, or its elements would merge."""
    query = "stop at Santa Barbara and Monterey"
    values = word_values(
        query, ROUTE, labels("none", "none", "stops", "stops", "none", "stops")
    )
    assert values == {"stops": ["Santa Barbara", "Monterey"]}


def test_decode_fills_each_kind_of_argument() -> None:
    """Words, a choice, a flag, and a set decode into typed values."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    assert decode_arguments("base 10", AREA, choices, AREA_NOULS) == {
        "base": 10,
        "unit": "cm",
        "round": False,
        "tags": ["a"],
    }


def test_decode_omits_an_optional_argument_the_query_does_not_state() -> None:
    """An unstated optional argument is left to the function's default."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    nouls = {**AREA_NOULS, "stated.unit": 0.1}
    assert "unit" not in decode_arguments("base 10", AREA, choices, nouls)


def test_decode_omits_an_empty_set() -> None:
    """A set with no member answered yes is not passed as an empty list."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    nouls = {**AREA_NOULS, "member.tags.a": 0.2}
    assert "tags" not in decode_arguments("base 10", AREA, choices, nouls)


def test_decode_omits_a_number_that_does_not_parse() -> None:
    """A word that is not a number never reaches a numeric argument."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    assert "base" not in decode_arguments("base ten", AREA, choices, AREA_NOULS)


def test_decode_omits_a_number_written_with_separators() -> None:
    """Nothing is converted, so 1,000 is left out rather than read as 1."""
    choices = {**labels("base", "base"), "choice.unit": {"cm": 1.0}}
    assert "base" not in decode_arguments("base 1,000", AREA, choices, AREA_NOULS)


def test_word_values_fill_a_gap_of_exactly_max_gap_words() -> None:
    """Two unlabelled words inside a name are filled; three are not."""
    query = "see The Lord of the Rings"
    filled = labels("none", "city", "city", "none", "none", "city")
    assert word_values(query, ROUTE, filled) == {"city": ["The Lord of the Rings"]}


def test_word_values_do_not_fill_a_gap_holding_another_label() -> None:
    """A gap is filled only if every word in it is unlabelled."""
    values = word_values("Paris 60 Rome", ROUTE, labels("city", "speed", "city"))
    assert values == {"city": ["Paris", "Rome"], "speed": ["60"]}


def test_word_values_count_a_label_at_exactly_the_threshold() -> None:
    """A label at 0.7 counts and one just under does not."""
    choices = {
        "token_0": {"city": 0.69, "none": 0.31},
        "token_1": {"city": 0.7, "none": 0.3},
    }
    assert word_values("to Paris", ROUTE, choices) == {"city": ["Paris"]}


def test_decode_uses_the_first_run_of_a_scalar() -> None:
    """A scalar takes one value even if two separate runs carry its label."""
    choices = labels("city", "none", "none", "none", "city")
    assert decode_arguments("Paris or maybe then Rome", ROUTE, choices, {}) == {
        "city": "Paris"
    }


def test_decode_builds_an_array_from_its_runs() -> None:
    """Each run of an array parameter is one element."""
    choices = labels("none", "none", "stops", "stops", "none", "stops")
    query = "stop at Santa Barbara and Monterey"
    assert decode_arguments(query, ROUTE, choices, {}) == {
        "stops": ["Santa Barbara", "Monterey"]
    }


def test_decode_omits_a_string_array_with_an_element_that_does_not_parse() -> None:
    """One unreadable element drops the whole array rather than shortening it."""
    choices = labels("none", "stops", "none", "stops")
    assert decode_arguments("via Paris and ?", ROUTE, choices, {}) == {}


def test_decode_ignores_a_label_the_model_is_not_confident_in() -> None:
    """A word labelled below the threshold is treated as no argument."""
    choices = {
        "token_0": {"city": 0.6, "none": 0.4},
        "token_1": {"city": 0.95, "none": 0.05},
    }
    assert decode_arguments("from Paris", ROUTE, choices, {}) == {"city": "Paris"}


def test_decode_reads_a_number_past_the_word_that_names_it() -> None:
    """A number is read word by word, so a labelled cue word does not hide it."""
    choices = {**labels("base", "none", "base"), "choice.unit": {"cm": 1.0}}
    arguments = decode_arguments("base of 10", AREA, choices, AREA_NOULS)
    assert arguments["base"] == 10


def test_decode_builds_a_number_array_from_each_word_that_parses() -> None:
    """Every labelled word that reads as a number is one element."""
    grid = Function(
        name="grid",
        description="Grid.",
        parameters=(
            Parameter(
                name="sizes",
                type="array",
                description="Sizes.",
                required=True,
                item_type="integer",
            ),
        ),
    )
    choices = labels("sizes", "sizes", "sizes", "none", "sizes")
    assert decode_arguments("sizes 3 4 and 5", grid, choices, {}) == {
        "sizes": [3, 4, 5]
    }


def test_decode_of_a_function_without_parameters_is_empty() -> None:
    """A function with nothing to fill decodes without any answers."""
    assert decode_arguments("ping the service", PING, {}, {}) == {}


def test_input_tokens_raises_when_the_response_has_no_usage() -> None:
    """Missing usage is an error, not zero tokens."""
    response = SystemOneResponse(model="jev-1.13.0", usage=Usage(), answers={})
    with pytest.raises(ValueError, match="no input token usage"):
        input_tokens(response)
