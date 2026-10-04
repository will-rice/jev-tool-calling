"""Tests for choosing which functions to call, how often, and decoding several calls."""

from jevtools.jev import (
    decode_parallel,
    parallel_spec,
    select_calls,
    selection_questions,
    value_questions,
)
from jevtools.models import Function, Parameter
from jevtools.spec import FunctionSpec, ParameterSpec, default_spec
from jevtools.values import number_readings, word_spans

PLAY = Function(
    name="play",
    description="Play an artist.",
    parameters=(
        Parameter(name="artist", type="string", description="Artist.", required=True),
        Parameter(name="minutes", type="float", description="Minutes.", required=True),
        Parameter(name="app", type="string", description="App.", required=False),
        Parameter(
            name="tracks",
            type="array",
            description="Tracks.",
            required=False,
            item_type="integer",
        ),
    ),
)
STOP = Function(name="stop", description="Stop playback.", parameters=())


def labels(*names: str) -> dict[str, dict[str, float]]:
    """Build certain token answers for a query, one label per word."""
    return {f"token_{index}": {name: 1.0} for index, name in enumerate(names)}


def test_selection_questions_for_one_function_ask_the_tool_and_its_count() -> None:
    """With one function offered there is nothing to choose between."""
    assert list(selection_questions((PLAY,))) == ["tool", "count.0"]


def test_selection_questions_for_several_functions_ask_which_are_needed() -> None:
    """Several offered functions add how many are needed and a yes/no for each."""
    questions = selection_questions((PLAY, STOP))
    assert list(questions) == [
        "tool",
        "distinct",
        "needed.0",
        "count.0",
        "needed.1",
        "count.1",
    ]
    assert list(questions["distinct"].criteria or {}) == ["1", "2"]
    assert list(questions["count.0"].criteria or {}) == [str(n) for n in range(1, 9)]


def test_select_calls_is_empty_when_the_model_abstains() -> None:
    """Choosing none means no call, whatever the other answers say."""
    choices = {"tool": {"none": 0.9, "play": 0.1}, "count.0": {"3": 1.0}}
    assert select_calls((PLAY,), choices, {}) == []


def test_select_calls_makes_one_call_unless_confident_of_more() -> None:
    """A second call is added only when one call is clearly not enough."""
    tool = {"tool": {"play": 1.0}}
    unsure = {**tool, "count.0": {"1": 0.3, "2": 0.7}}
    confident = {**tool, "count.0": {"1": 0.1, "2": 0.2, "3": 0.7}}
    assert select_calls((PLAY,), unsure, {}) == [(PLAY, 1)]
    assert select_calls((PLAY,), confident, {}) == [(PLAY, 3)]


def test_select_calls_adds_functions_only_when_confident_several_are_needed() -> None:
    """The picked function stands alone unless one function is clearly not enough."""
    answers = {
        "tool": {"play": 0.9, "stop": 0.1},
        "count.0": {"1": 1.0},
        "count.1": {"1": 1.0},
    }
    nouls = {"needed.0": 0.9, "needed.1": 0.8}
    one = {**answers, "distinct": {"1": 0.4, "2": 0.6}}
    two = {**answers, "distinct": {"1": 0.05, "2": 0.95}}
    assert select_calls((PLAY, STOP), one, nouls) == [(PLAY, 1)]
    assert select_calls((PLAY, STOP), two, nouls) == [(PLAY, 1), (STOP, 1)]


def test_select_calls_does_not_abstain_when_several_functions_are_needed() -> None:
    """No single function answers a request for two, so none is not the last word."""
    answers = {
        "tool": {"none": 0.7, "play": 0.2, "stop": 0.1},
        "distinct": {"1": 0.0, "2": 1.0},
        "count.0": {"1": 1.0},
        "count.1": {"1": 1.0},
    }
    both = {"needed.0": 0.6, "needed.1": 0.9}
    only_one = {"needed.0": 0.2, "needed.1": 0.9}
    assert select_calls((PLAY, STOP), answers, both) == [(PLAY, 1), (STOP, 1)]
    assert select_calls((PLAY, STOP), answers, only_one) == []


def test_number_readings_give_each_way_one_number_can_be_read() -> None:
    """A percentage is offered as written and as a fraction."""
    query = "win 30% of 1,200 rounds"
    spans = word_spans(query)
    assert number_readings(query, spans, 1, "float") == {"30.0": None, "0.3": "30%"}
    assert number_readings(query, spans, 4, "integer") == {"1200": "1,200"}
    assert number_readings(query, spans, 0, "float") == {}


def test_parallel_spec_labels_numbers_as_words() -> None:
    """Several calls need every number an argument takes, so numbers are labelled."""
    spec = FunctionSpec(
        name="play",
        parameters={
            **default_spec(PLAY).parameters,
            "minutes": ParameterSpec(kind="number", question="How long?"),
        },
    )
    assert parallel_spec(spec).parameters["minutes"].kind == "text"
    assert parallel_spec(spec).parameters["artist"].kind == "text"


def test_value_questions_ask_how_to_read_an_ambiguous_number() -> None:
    """Only a labelled number with more than one reading gets a question."""
    query = "ABBA for 30% and Queen for 15"
    choices = labels(
        "artist", "none", "minutes", "none", "none", "artist", "none", "minutes"
    )
    questions = value_questions(query, PLAY, default_spec(PLAY), choices)
    assert list(questions) == ["reading.minutes.2"]
    assert questions["reading.minutes.2"].criteria == {
        "30.0": "the number exactly as written",
        "0.3": "written as 30%",
    }


def test_decode_parallel_zips_values_by_position() -> None:
    """The first artist goes with the first duration, and one app is shared."""
    query = "ABBA and Queen for 20 and 15 on Spotify"
    choices = labels(
        "artist", "none", "artist", "none", "minutes", "none", "minutes", "none", "app"
    )
    calls = decode_parallel(query, PLAY, default_spec(PLAY), choices, {})
    assert calls == [
        {"artist": "ABBA", "minutes": 20.0, "app": "Spotify"},
        {"artist": "Queen", "minutes": 15.0, "app": "Spotify"},
    ]


def test_decode_parallel_shares_a_value_repeated_identically() -> None:
    """The same value stated twice is one shared value, not two of three."""
    query = "ABBA 5 Queen 5 Blur"
    choices = labels("artist", "minutes", "artist", "minutes", "artist")
    calls = decode_parallel(query, PLAY, default_spec(PLAY), choices, {})
    assert [call["minutes"] for call in calls] == [5.0, 5.0, 5.0]
    assert [call["artist"] for call in calls] == ["ABBA", "Queen", "Blur"]


def test_decode_parallel_splits_an_array_into_one_per_call() -> None:
    """Elements joined by commas stay together; a longer gap starts a new array."""
    query = "ABBA tracks 1, 2 and 3 then Queen tracks 7, 8"
    choices = labels(
        "artist", "none", "tracks", "none", "tracks", "none", "tracks",
        "none", "artist", "none", "tracks", "none", "tracks",
    )  # fmt: skip
    calls = decode_parallel(query, PLAY, default_spec(PLAY), choices, {})
    assert [call["tracks"] for call in calls] == [[1, 2, 3], [7, 8]]


def test_decode_parallel_uses_the_chosen_reading_of_a_number() -> None:
    """An ambiguous number takes the reading the model chose."""
    query = "ABBA for 30% and Queen for 15"
    choices = {
        **labels(
            "artist", "none", "minutes", "none", "none", "artist", "none", "minutes"
        ),
        "reading.minutes.2": {"0.3": 0.9, "30.0": 0.1},
    }
    calls = decode_parallel(query, PLAY, default_spec(PLAY), choices, {})
    assert [call["minutes"] for call in calls] == [0.3, 15.0]


def test_decode_parallel_does_not_merge_values_a_word_apart() -> None:
    """Gap filling would join two artists separated by "and"; a hyphen still joins."""
    query = "AC-DC and Queen"
    choices = labels("artist", "none", "artist", "none", "artist")
    calls = decode_parallel(query, PLAY, default_spec(PLAY), choices, {})
    assert [call["artist"] for call in calls] == ["AC-DC", "Queen"]
