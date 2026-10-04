"""Tests for question building and answer decoding."""

import pytest
from typesafe_sdk import Choice, Noul

from jevtools.jev import argument_questions, tool_question, word_spans
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
        "none": "The word is not part of any argument's value.",
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
