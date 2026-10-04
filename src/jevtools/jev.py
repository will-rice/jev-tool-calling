"""Ask Jev which function to call and with which arguments, and decode its answers."""

import re
import string
from collections.abc import Mapping, Sequence
from itertools import groupby

from pydantic import JsonValue
from typesafe_sdk import Choice, Noul, SystemOneResponse, TypeSafeClient

from jevtools.config import (
    LABEL_THRESHOLD,
    MAX_GAP,
    MODEL,
    NONE,
    THRESHOLD,
    WORD_PATTERN,
)
from jevtools.models import Call, Example, Function, Parameter, Prediction

NO_TOOL = "No offered function can answer the request."
NO_ARGUMENT = (
    "The word is not part of any argument's value. A word that only names or "
    "introduces an argument, such as a field name, a preposition, or "
    "punctuation around the value, is none."
)
NUMBER_EDGE = string.punctuation.replace("-", "").replace("+", "").replace(".", "")
INTEGER = re.compile(r"[-+]?\d+")
FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def predict(client: TypeSafeClient, example: Example) -> Prediction:
    """Pick a function for the query, then fill its arguments.

    Arguments are asked only for the function the model picked. A function
    with nothing to ask is called with no arguments and no second request.

    Args:
        client: An open TypeSafe client.
        example: The query and the functions offered for it.

    Returns:
        The predicted call, or no call if the model abstained, with every
        answer's probabilities and each request's token usage.
    """
    state = {"query": example.query}
    tool_response = client.system_one(
        state, {"tool": tool_question(example.functions)}, model=MODEL
    )
    tool = tool_response.choices["tool"]
    call = None
    choices: dict[str, dict[str, float]] = {}
    nouls: dict[str, float] = {}
    argument_input_tokens = 0
    if tool.choice != NONE:
        function = next(f for f in example.functions if f.name == tool.choice)
        questions = argument_questions(example.query, function)
        if questions:
            response = client.system_one(state, questions, model=MODEL)
            choices = {
                name: answer.probabilities for name, answer in response.choices.items()
            }
            nouls = {name: answer.noul for name, answer in response.nouls.items()}
            argument_input_tokens = input_tokens(response)
        call = Call(
            name=function.name,
            arguments=decode_arguments(example.query, function, choices, nouls),
        )
    return Prediction(
        example=example,
        call=call,
        tool_probabilities=tool.probabilities,
        choices=choices,
        nouls=nouls,
        tool_input_tokens=input_tokens(tool_response),
        argument_input_tokens=argument_input_tokens,
        model=tool_response.model,
    )


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
    most words outside any value are given a parameter anyway. It also says
    that a word naming or introducing an argument is none: without that, the
    model labels "base" in "a base of 10" as well as "10".

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


def decode_arguments(
    query: str,
    function: Function,
    choices: Mapping[str, Mapping[str, float]],
    nouls: Mapping[str, float],
) -> dict[str, JsonValue]:
    """Turn the argument answers into the function's arguments.

    An argument is left out when its value does not parse, when an optional
    closed-set argument is not stated, or when a set has no members.

    A string takes its first run of labelled words, and a string array one
    element per run, all of which must be readable. A number is read word
    by word: it takes the first labelled word that parses, and a number
    array every labelled word that parses.

    Args:
        query: The request text.
        function: The function being called.
        choices: Each Choice question's option probabilities, by question name.
        nouls: Each Noul question's yes-probability, by question name.
    """
    arguments: dict[str, JsonValue] = {}
    texts = (
        word_values(query, function, choices)
        if any(parameter.kind == "words" for parameter in function.parameters)
        else {}
    )
    for parameter in function.parameters:
        name = parameter.name
        match parameter.kind:
            case "words":
                value = open_value(parameter, texts.get(name, []))
                if value is not None:
                    arguments[name] = value
            case "set":
                members: list[JsonValue] = [
                    value
                    for value in parameter.enum
                    if nouls[f"member.{name}.{value}"] >= THRESHOLD
                ]
                if members:
                    arguments[name] = members
            case "choice":
                if parameter.required or nouls[f"stated.{name}"] >= THRESHOLD:
                    options = choices[f"choice.{name}"]
                    arguments[name] = max(options, key=lambda option: options[option])
            case "flag":
                if parameter.required or nouls[f"stated.{name}"] >= THRESHOLD:
                    arguments[name] = nouls[f"flag.{name}"] >= THRESHOLD
    return arguments


def open_value(parameter: Parameter, texts: Sequence[str]) -> JsonValue:
    """Read an open-valued parameter's value from its texts, or None to omit it.

    A string takes the first text, and a string array every text, all of
    which must be readable. A number takes the first text that parses, and
    a number array every text that parses.
    """
    values: list[JsonValue] = [coerce(text, parameter.value_type) for text in texts]
    if parameter.value_type != "string":
        values = [value for value in values if value is not None]
    elif parameter.type != "array":
        values = values[:1]
    if not values or None in values:
        return None
    return values if parameter.type == "array" else values[0]


def word_values(
    query: str, function: Function, choices: Mapping[str, Mapping[str, float]]
) -> dict[str, list[str]]:
    """Group labelled words into each open-valued parameter's value texts.

    A word's label is its most probable option, and counts only at
    LABEL_THRESHOLD or above: the words that merely introduce a value are
    the ones the model is unsure of.

    For a string parameter, adjacent words with the same label are one
    value, cut from the query as written. For a scalar string, up to MAX_GAP
    unlabelled words between two words with its label take that label, so
    small words inside a name stay in it. An array gets no filling, or its
    elements would merge. For a number parameter every labelled word is its
    own text, so a labelled word beside the number cannot spoil it.
    """
    spans = word_spans(query)
    labels = []
    for index in range(len(spans)):
        probabilities = choices[f"token_{index}"]
        label = max(probabilities, key=lambda option: probabilities[option])
        labels.append(label if probabilities[label] >= LABEL_THRESHOLD else NONE)
    strings = {p.name for p in function.parameters if p.value_type == "string"}
    scalar_strings = {p.name for p in function.parameters if p.type == "string"}
    for start, label in enumerate(labels):
        if label not in scalar_strings:
            continue
        for end in range(start + 2, min(start + MAX_GAP + 2, len(labels))):
            if labels[end] == label and all(
                between == NONE for between in labels[start + 1 : end]
            ):
                labels[start + 1 : end] = [label] * (end - start - 1)
                break
    values: dict[str, list[str]] = {}
    position = 0
    for label, group in groupby(labels):
        length = len(list(group))
        pieces = (
            [(position, position + length - 1)]
            if label in strings
            else [(index, index) for index in range(position, position + length)]
        )
        if label != NONE:
            values.setdefault(label, []).extend(
                query[spans[first][0] : spans[last][1]] for first, last in pieces
            )
        position += length
    return values


def coerce(text: str, value_type: str | None) -> str | int | float | None:
    """Read a value of the given type from query text, or None if it is not one.

    Punctuation around the text is dropped. Nothing is converted: a number
    word, a unit, or a thousands separator makes a number unreadable.

    Raises:
        ValueError: If the type is not a string or a number.
    """
    number = text.strip(NUMBER_EDGE).rstrip(".")
    match value_type:
        case "string":
            return text.strip(string.punctuation) or None
        case "integer":
            return int(number) if INTEGER.fullmatch(number) else None
        case "float":
            return float(number) if FLOAT.fullmatch(number) else None
        case _:
            raise ValueError(f"Cannot read a {value_type} from words")


def input_tokens(response: SystemOneResponse) -> int:
    """Return a response's input token count.

    Raises:
        ValueError: If the response carries no usage.
    """
    if response.usage.input_tokens is None:
        raise ValueError("Jev response has no input token usage")
    return response.usage.input_tokens
