"""Tests for question building and answer decoding."""

import pytest
from pydantic import JsonValue
from typesafe_sdk import Choice, Noul, SystemOneResponse, Usage

from jevtools.jev import (
    NO_NUMBER,
    argument_questions,
    coerce,
    decode_arguments,
    input_tokens,
    question_batches,
    request_state,
    tool_question,
    word_runs,
    word_spans,
)
from jevtools.models import Example, Function, Parameter
from jevtools.spec import FunctionSpec, ParameterSpec, default_spec


def ask(query: str, function: Function) -> dict[str, Choice | Noul]:
    """Build the questions the schema alone implies."""
    return argument_questions(query, function, default_spec(function))


def decode(
    query: str,
    function: Function,
    choices: dict[str, dict[str, float]],
    nouls: dict[str, float],
) -> dict[str, JsonValue]:
    """Decode answers to the questions the schema alone implies."""
    return decode_arguments(query, function, default_spec(function), choices, nouls)


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
    questions = ask("base 10", AREA)
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
    question = ask("base 10", AREA)["token_1"]
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
    questions = ask("10 by 10", AREA)
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
    question = ask("base 10", AREA)["choice.unit"]
    assert question.criteria == {"cm": None, "in": None}
    assert question.instructions == {
        "function": "triangle.area",
        "argument": "unit",
        "description": "The unit.",
        "question": "Which value of `argument` does `query` ask for?",
    }


def test_required_closed_set_has_no_stated_question() -> None:
    """A required argument is always filled, so nothing asks if it is stated."""
    assert "stated.round" not in ask("base 10", AREA)


def test_member_question_names_its_value() -> None:
    """Each member of a set is asked about separately."""
    question = ask("base 10", AREA)["member.tags.b"]
    assert question.instructions == {
        "function": "triangle.area",
        "argument": "tags",
        "description": "Tags.",
        "value": "b",
        "question": "Does `query` ask for `value` in `argument`?",
    }


def test_function_without_fillable_parameters_has_no_questions() -> None:
    """Nothing is asked, so the caller must skip the request."""
    assert ask("ping the service", PING) == {}


def test_function_without_open_parameters_has_no_token_questions() -> None:
    """Words are labelled only when some parameter can take them."""
    switch = Function(
        name="switch",
        description="Flip a switch.",
        parameters=(
            Parameter(name="on", type="boolean", description="On.", required=True),
        ),
    )
    assert list(ask("turn it on", switch)) == ["flag.on"]


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
        ask("a b", clash)


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


def texts(
    query: str, function: Function, choices: dict[str, dict[str, float]]
) -> dict[str, list[str]]:
    """Cut each parameter's runs of words from the query."""
    spans = word_spans(query)
    return {
        name: [query[spans[first][0] : spans[last][1]] for first, last in runs]
        for name, runs in word_runs(query, function, choices).items()
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
    values = texts(query, ROUTE, labels("none", "none", "city", "city", "city", "none"))
    assert values == {"city": ["New-York"]}


def test_word_values_fill_short_gaps_inside_a_scalar_value() -> None:
    """Small unlabelled words inside a name stay in it."""
    query = "go to Stratford upon Avon now"
    values = texts(query, ROUTE, labels("none", "none", "city", "none", "city", "none"))
    assert values == {"city": ["Stratford upon Avon"]}


def test_word_values_keep_array_elements_apart() -> None:
    """Gaps are not filled for an array, or its elements would merge."""
    query = "stop at Santa Barbara and Monterey"
    values = texts(
        query, ROUTE, labels("none", "none", "stops", "stops", "none", "stops")
    )
    assert values == {"stops": ["Santa Barbara", "Monterey"]}


def test_decode_fills_each_kind_of_argument() -> None:
    """Words, a choice, a flag, and a set decode into typed values."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    assert decode("base 10", AREA, choices, AREA_NOULS) == {
        "base": 10,
        "unit": "cm",
        "round": False,
        "tags": ["a"],
    }


def test_decode_omits_an_optional_argument_the_query_does_not_state() -> None:
    """An unstated optional argument is left to the function's default."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    nouls = {**AREA_NOULS, "stated.unit": 0.1}
    assert "unit" not in decode("base 10", AREA, choices, nouls)


def test_decode_omits_an_empty_set() -> None:
    """A set with no member answered yes is not passed as an empty list."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    nouls = {**AREA_NOULS, "member.tags.a": 0.2}
    assert "tags" not in decode("base 10", AREA, choices, nouls)


def test_decode_omits_a_number_that_does_not_parse() -> None:
    """A word that is not a number never reaches a numeric argument."""
    choices = {**labels("none", "base"), "choice.unit": {"cm": 0.9, "in": 0.1}}
    assert "base" not in decode("base ten", AREA, choices, AREA_NOULS)


def test_decode_omits_a_number_written_with_separators() -> None:
    """Nothing is converted, so 1,000 is left out rather than read as 1."""
    choices = {**labels("base", "base"), "choice.unit": {"cm": 1.0}}
    assert "base" not in decode("base 1,000", AREA, choices, AREA_NOULS)


def test_word_values_fill_a_gap_of_exactly_max_gap_words() -> None:
    """Two unlabelled words inside a name are filled; three are not."""
    query = "see The Lord of the Rings"
    filled = labels("none", "city", "city", "none", "none", "city")
    assert texts(query, ROUTE, filled) == {"city": ["The Lord of the Rings"]}


def test_word_values_do_not_fill_a_gap_holding_another_label() -> None:
    """A gap is filled only if every word in it is unlabelled."""
    values = texts("Paris 60 Rome", ROUTE, labels("city", "speed", "city"))
    assert values == {"city": ["Paris", "Rome"], "speed": ["60"]}


def test_word_values_count_a_label_at_exactly_the_threshold() -> None:
    """A label at 0.7 counts and one just under does not."""
    choices = {
        "token_0": {"city": 0.69, "none": 0.31},
        "token_1": {"city": 0.7, "none": 0.3},
    }
    assert texts("to Paris", ROUTE, choices) == {"city": ["Paris"]}


def test_decode_uses_the_first_of_two_equally_strong_runs() -> None:
    """A scalar takes one value; a tie between runs goes to the earlier one."""
    choices = labels("city", "none", "none", "none", "city")
    assert decode("Paris or maybe then Rome", ROUTE, choices, {}) == {"city": "Paris"}


def test_decode_builds_an_array_from_its_runs() -> None:
    """Each run of an array parameter is one element."""
    choices = labels("none", "none", "stops", "stops", "none", "stops")
    query = "stop at Santa Barbara and Monterey"
    assert decode(query, ROUTE, choices, {}) == {"stops": ["Santa Barbara", "Monterey"]}


def test_decode_omits_a_string_array_with_an_element_that_does_not_parse() -> None:
    """One unreadable element drops the whole array rather than shortening it."""
    choices = labels("none", "stops", "none", "stops")
    assert decode("via Paris and ?", ROUTE, choices, {}) == {}


def test_decode_ignores_a_label_the_model_is_not_confident_in() -> None:
    """A word labelled below the threshold is treated as no argument."""
    choices = {
        "token_0": {"city": 0.6, "none": 0.4},
        "token_1": {"city": 0.95, "none": 0.05},
    }
    assert decode("from Paris", ROUTE, choices, {}) == {"city": "Paris"}


def test_decode_reads_a_number_past_the_word_that_names_it() -> None:
    """A number is read word by word, so a labelled cue word does not hide it."""
    choices = {**labels("base", "none", "base"), "choice.unit": {"cm": 1.0}}
    arguments = decode("base of 10", AREA, choices, AREA_NOULS)
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
    assert decode("sizes 3 4 and 5", grid, choices, {}) == {"sizes": [3, 4, 5]}


def test_decode_of_a_function_without_parameters_is_empty() -> None:
    """A function with nothing to fill decodes without any answers."""
    assert decode("ping the service", PING, {}, {}) == {}


def test_input_tokens_raises_when_the_response_has_no_usage() -> None:
    """Missing usage is an error, not zero tokens."""
    response = SystemOneResponse(model="jev-1.13.0", usage=Usage(), answers={})
    with pytest.raises(ValueError, match="no input token usage"):
        input_tokens(response)


def test_word_values_bridge_an_unsure_word_inside_a_scalar_string() -> None:
    """A word the model half-labels joins the confident words on either side."""
    choices = {
        "token_0": {"city": 0.9, "none": 0.1},
        **{f"token_{i}": {"city": 0.35, "none": 0.65} for i in (1, 2, 3)},
        "token_4": {"city": 0.9, "none": 0.1},
    }
    assert texts("my-bot-id", ROUTE, choices) == {"city": ["my-bot-id"]}


def test_word_values_do_not_bridge_a_word_below_the_bridge_threshold() -> None:
    """A long gap is bridged only if every word in it leans to the argument."""
    choices = {
        "token_0": {"city": 0.9, "none": 0.1},
        "token_1": {"city": 0.35, "none": 0.65},
        "token_2": {"city": 0.29, "none": 0.71},
        "token_3": {"city": 0.35, "none": 0.65},
        "token_4": {"city": 0.9, "none": 0.1},
    }
    assert texts("my-bot-id", ROUTE, choices) == {"city": ["my", "id"]}


def test_word_values_do_not_bridge_array_elements() -> None:
    """A half-labelled comma between two elements does not merge them."""
    choices = {
        "token_0": {"stops": 0.9, "none": 0.1},
        "token_1": {"stops": 0.45, "none": 0.55},
        "token_2": {"stops": 0.9, "none": 0.1},
    }
    assert texts("Paris, Rome", ROUTE, choices) == {"stops": ["Paris", "Rome"]}


def test_decode_takes_the_strongest_run_of_a_scalar_string() -> None:
    """Of two separate runs, the one with more probability behind it wins."""
    choices = {
        "token_0": {"city": 0.75, "none": 0.25},
        "token_1": {"city": 0.1, "none": 0.9},
        "token_2": {"city": 0.1, "none": 0.9},
        "token_3": {"city": 0.1, "none": 0.9},
        "token_4": {"city": 0.95, "none": 0.05},
        "token_5": {"city": 0.9, "none": 0.1},
    }
    query = "Yosemite which is in Mariposa County"
    assert decode(query, ROUTE, choices, {}) == {"city": "Mariposa County"}


TRIP = Function(
    name="trip",
    description="Plan a trip.",
    parameters=(
        Parameter(name="budget", type="integer", description="Budget.", required=True),
        Parameter(name="mode", type="string", description="Mode.", required=False),
        Parameter(name="day", type="string", description="Day.", required=True),
        Parameter(name="where", type="string", description="Where.", required=True),
        Parameter(name="level", type="integer", description="Level.", required=True),
    ),
)
TRIP_SPEC = FunctionSpec(
    name="trip",
    parameters={
        "budget": ParameterSpec(kind="number", question="How much can be spent?"),
        "mode": ParameterSpec(
            kind="options",
            question="How do they want to travel?",
            stated="Does the request say how to travel?",
            options={"transit": "bus, train, or other public transport", "car": None},
            open=True,
        ),
        "day": ParameterSpec(kind="date", question="When?", format="%Y-%m-%d"),
        "where": ParameterSpec(
            kind="place",
            question="Where to?",
            place_format="city, state_abbr",
            no_state="city, country",
        ),
        "level": ParameterSpec(
            kind="options", question="Which tier?", options={"1": "basic", "2": "full"}
        ),
    },
)
QUERY = "Austin by bus on March 5 2023 for $1,500"


def test_spec_questions_cover_each_kind() -> None:
    """A number is picked, options and date parts chosen, a place completed."""
    asked = argument_questions(QUERY, TRIP, TRIP_SPEC)
    assert [name for name in asked if not name.startswith("token_")] == [
        "number.budget",
        "choice.mode",
        "stated.mode",
        "date.day.month",
        "date.day.day",
        "date.day.year",
        "place.where.state",
        "place.where.country",
        "choice.level",
    ]
    assert set(asked["token_0"].criteria or {}) == {"mode", "where", "none"}


def test_number_question_offers_the_numbers_code_found() -> None:
    """The options are readings of the numbers in the query, plus none."""
    question = argument_questions(QUERY, TRIP, TRIP_SPEC)["number.budget"]
    assert question.criteria == {
        "5": None,
        "2023": None,
        "1500": "written as 1,500",
        "none": NO_NUMBER,
    }
    assert question.instructions == {
        "function": "trip",
        "argument": "budget",
        "description": "Budget.",
        "question": "How much can be spent?",
    }


def test_number_question_is_not_asked_without_a_candidate() -> None:
    """With no number in the query there is nothing to pick."""
    assert "number.budget" not in argument_questions("Austin by bus", TRIP, TRIP_SPEC)


def test_open_options_add_an_escape_for_an_unlisted_value() -> None:
    """An open list offers its values with their lines, and an other option."""
    question = argument_questions(QUERY, TRIP, TRIP_SPEC)["choice.mode"]
    assert question.criteria == {
        "transit": "bus, train, or other public transport",
        "car": None,
        "other": "The request names a value that is not one of these.",
    }


def spec_answers(**changes: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build answers to the trip questions for the query, with overrides."""
    words = ("where", "none", "mode", "none", "none", "none", "none", "none", "none")
    answers = {
        **{f"token_{index}": {label: 1.0} for index, label in enumerate(words)},
        "token_9": {"none": 1.0},
        "number.budget": {"1500": 0.9, "5": 0.1},
        "choice.mode": {"transit": 0.8, "car": 0.1, "other": 0.1},
        "date.day.month": {"March": 1.0},
        "date.day.day": {"5": 1.0},
        "date.day.year": {"2023": 1.0},
        "place.where.state": {"TX": 0.9, "none": 0.1},
        "place.where.country": {"United States": 1.0},
        "choice.level": {"2": 0.7, "1": 0.3},
    }
    return {**answers, **changes}


def test_spec_decoding_assembles_each_kind_of_value() -> None:
    """Code builds the number, the date, the place, and typed option values."""
    arguments = decode_arguments(
        QUERY, TRIP, TRIP_SPEC, spec_answers(), {"stated.mode": 0.9}
    )
    assert arguments == {
        "budget": 1500,
        "mode": "transit",
        "day": "2023-03-05",
        "where": "Austin, TX",
        "level": 2,
    }


def test_spec_decoding_copies_an_unlisted_option_from_the_query() -> None:
    """Choosing other falls back to the words labelled with the argument."""
    answers = spec_answers(**{"choice.mode": {"other": 0.9, "car": 0.1}})
    arguments = decode_arguments(QUERY, TRIP, TRIP_SPEC, answers, {"stated.mode": 0.9})
    assert arguments["mode"] == "bus"


def test_spec_decoding_omits_what_the_query_does_not_give() -> None:
    """No number picked, no year, and an unstated option are all left out."""
    answers = spec_answers(
        **{"number.budget": {"none": 1.0}, "date.day.year": {"none": 1.0}}
    )
    arguments = decode_arguments(QUERY, TRIP, TRIP_SPEC, answers, {"stated.mode": 0.2})
    assert arguments == {"where": "Austin, TX", "level": 2}


def test_request_state_is_the_query_alone_without_earlier_messages() -> None:
    """A single-message request sends just the query."""
    example = Example(id="x", split="simple", query="hi", functions=(), gold=None)
    assert request_state(example) == {"query": "hi"}


def test_request_state_carries_earlier_messages_as_context() -> None:
    """A system prompt or earlier turn is shown to the model, not labelled."""
    example = Example(
        id="x",
        split="live_multiple",
        query="and tomorrow?",
        context=(("system", "Be brief."), ("user", "Weather in Paris?")),
        functions=(),
        gold=None,
    )
    assert request_state(example) == {
        "query": "and tomorrow?",
        "context": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
    }


def test_word_question_shows_a_window_of_words_in_a_long_query() -> None:
    """A word far from the edges is shown its 95 nearest words on each side."""
    query = " ".join(f"w{index}" for index in range(300))
    question = ask(query, ROUTE)["token_150"]
    assert isinstance(question.instructions, dict)
    assert question.instructions["words_before"] == " ".join(
        f"w{index}" for index in range(55, 150)
    )
    assert question.instructions["words_after"] == " ".join(
        f"w{index}" for index in range(151, 246)
    )


def test_question_batches_keep_a_short_request_whole() -> None:
    """Questions that fit one request are sent together, in order."""
    asked = ask("Paris at 60", ROUTE)
    assert question_batches(asked) == [asked]


def test_question_batches_split_a_long_request_without_losing_questions() -> None:
    """A long query's questions go out in several requests, each within the budget."""
    asked = ask(" ".join(f"w{index}" for index in range(600)), ROUTE)
    batches = question_batches(asked)
    assert len(batches) > 1
    assert [name for batch in batches for name in batch] == list(asked)
    assert all(
        sum(len(question.model_dump_json()) for question in batch.values()) <= 120_000
        for batch in batches
    )


def test_decode_casts_a_numeric_option_and_keeps_a_named_one() -> None:
    """An integer argument's option is a number if it reads as one."""
    spec = TRIP_SPEC.model_copy(
        update={
            "parameters": {
                **TRIP_SPEC.parameters,
                "level": ParameterSpec(
                    kind="options", question="q", options={"1": None, "any": None}
                ),
            }
        }
    )
    numeric = spec_answers(**{"choice.level": {"1": 0.9, "any": 0.1}})
    named = spec_answers(**{"choice.level": {"any": 0.9, "1": 0.1}})
    nouls = {"stated.mode": 0.9}
    assert decode_arguments(QUERY, TRIP, spec, numeric, nouls)["level"] == 1
    assert decode_arguments(QUERY, TRIP, spec, named, nouls)["level"] == "any"
