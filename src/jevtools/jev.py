"""Ask Jev which function to call and with which arguments."""

import re
from collections.abc import Sequence

from typesafe_sdk import Choice, Noul

from jevtools.config import NONE, WORD_PATTERN
from jevtools.models import Function

NO_TOOL = "No offered function can answer the request."
NO_ARGUMENT = "The word is not part of any argument's value."


def tool_question(functions: Sequence[Function]) -> Choice:
    """Build the question that picks one function or abstains.

    Raises:
        ValueError: If a function is named like the abstain option.
    """
    criteria = {function.name: function.description for function in functions}
    if NONE in criteria:
        raise ValueError(f"Function '{NONE}' collides with the abstain option")
    return Choice(
        instructions=(
            "Which function answers the request in `query`? "
            f"Answer {NONE} if no offered function can."
        ),
        criteria={**criteria, NONE: NO_TOOL},
    )


def argument_questions(query: str, function: Function) -> dict[str, Choice | Noul]:
    """Build every question needed to fill one function's arguments.

    Open-valued parameters share one question per word of the query, asking
    which of them the word supplies. The word and the words on either side
    of it are labelled fields, which the model reads more reliably than a
    marker inside a sentence. The none option is described: without that,
    most words outside any value are given a parameter anyway.

    A closed-set parameter is asked directly: a Choice over an enum, a Noul
    for a boolean, a Noul per member of an enum array. An optional enum or
    boolean also gets a Noul asking whether the query mentions it at all.

    Raises:
        ValueError: If an open-valued parameter is named like the none option.
    """
    questions: dict[str, Choice | Noul] = {}
    open_parameters = {
        parameter.name: parameter.description
        for parameter in function.parameters
        if parameter.kind == "words"
    }
    if NONE in open_parameters:
        raise ValueError(f"Parameter '{NONE}' collides with the none option")
    if open_parameters:
        words = [query[start:end] for start, end in word_spans(query)]
        for index, word in enumerate(words):
            questions[f"token_{index}"] = Choice(
                instructions={
                    "function": function.name,
                    "words_before": " ".join(words[:index]),
                    "word": word,
                    "words_after": " ".join(words[index + 1 :]),
                    "question": (
                        "Which argument of `function` does `word` supply in "
                        f"`query`? Answer {NONE} if it supplies no argument."
                    ),
                },
                criteria={**open_parameters, NONE: NO_ARGUMENT},
            )
    for parameter in function.parameters:
        name = parameter.name
        about = {
            "function": function.name,
            "argument": name,
            "description": parameter.description,
        }
        match parameter.kind:
            case "choice":
                questions[f"choice.{name}"] = Choice(
                    instructions={
                        **about,
                        "question": "Which value of `argument` does `query` ask for?",
                    },
                    criteria=dict.fromkeys(parameter.enum),
                )
            case "flag":
                questions[f"flag.{name}"] = Noul(
                    instructions={
                        **about,
                        "question": "Does `query` ask for `argument` to be true?",
                    }
                )
            case "set":
                for value in parameter.enum:
                    questions[f"member.{name}.{value}"] = Noul(
                        instructions={
                            **about,
                            "value": value,
                            "question": "Does `query` ask for `value` in `argument`?",
                        }
                    )
        if parameter.kind in ("choice", "flag") and not parameter.required:
            questions[f"stated.{name}"] = Noul(
                instructions={
                    **about,
                    "question": "Does `query` say anything about `argument`?",
                }
            )
    return questions


def word_spans(query: str) -> list[tuple[int, int]]:
    """Return each word's (start, end) character offsets in the query."""
    return [match.span() for match in re.finditer(WORD_PATTERN, query)]
