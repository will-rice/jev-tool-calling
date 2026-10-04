"""Ask Jev which function to call and with which arguments, and decode its answers."""

from collections.abc import Mapping, Sequence
from itertools import groupby

from pydantic import JsonValue
from typesafe_sdk import Choice, Noul, SystemOneResponse, TypeSafeClient

from jevtools.config import (
    BATCH_CHARACTERS,
    BRIDGE_THRESHOLD,
    CONTEXT_WORDS,
    LABEL_THRESHOLD,
    MAX_BRIDGE,
    MAX_CALLS,
    MAX_GAP,
    MODEL,
    NONE,
    ONE_CALL_THRESHOLD,
    ONE_FUNCTION_THRESHOLD,
    THRESHOLD,
)
from jevtools.models import Call, Example, Function, Method, Parameter, Prediction
from jevtools.spec import FunctionSpec, ParameterSpec, check, default_spec
from jevtools.values import (
    COUNTRIES,
    DAYS,
    JOINERS,
    LIST_GLUE,
    MONTHS,
    STATES,
    UNITED_STATES,
    YEARS,
    coerce,
    complete_place,
    format_date,
    number_candidates,
    number_readings,
    word_spans,
)

NO_TOOL = "No offered function can answer the request."
NO_ARGUMENT = (
    "The word is not part of any argument's value. A word that only names or "
    "introduces an argument, such as a field name, a preposition, or "
    "punctuation around the value, is none."
)
NO_NUMBER = "None of these is the requested value, or the request does not give it."
OTHER = "other"
DATE_PARTS = {
    "month": ("Which month is that date in?", MONTHS),
    "day": ("Which day of the month is that date?", DAYS),
    "year": ("Which year is that date in?", YEARS),
}


def predict(
    client: TypeSafeClient,
    example: Example,
    method: Method,
    specs: Mapping[str, FunctionSpec],
) -> Prediction:
    """Decide which functions a query needs and how often, then fill each call.

    One request picks a function or abstains, and asks whether several
    functions are needed and how many times each must be called. Arguments
    are asked only for the functions selected, in the way each one's spec
    says: the authored one under the spec method, or one derived from the
    schema under the words method. A function called once is decoded as a
    single call. A function called several times has its words labelled
    once and its values zipped by position into calls, with one more
    request for values that can be read two ways.

    A function with nothing to ask is called with no arguments and no
    further request, a long query's questions are spread over several
    requests, and a query offered no function is answered with no call and
    no request.

    Args:
        client: An open TypeSafe client.
        example: The query and the functions offered for it.
        method: Whether arguments are asked from an authored spec.
        specs: The authored specs by function key.

    Returns:
        The predicted calls, none if the model abstained, with every
        answer's probabilities and each request's token usage. Argument
        answers are keyed by the position of their function among those
        selected, as "position/question".
    """
    if not example.functions:
        return Prediction(
            example=example,
            method=method,
            calls=(),
            tool_probabilities={NONE: 1.0},
            choices={},
            nouls={},
            tool_input_tokens=0,
            argument_input_tokens=0,
            model=MODEL,
        )
    state = request_state(example)
    selection = client.system_one(
        state, selection_questions(example.functions), model=MODEL
    )
    choices = {name: answer.probabilities for name, answer in selection.choices.items()}
    nouls = {name: answer.noul for name, answer in selection.nouls.items()}
    calls: list[Call] = []
    argument_input_tokens = 0
    selected = select_calls(example.functions, choices, nouls)
    for position, (function, count) in enumerate(selected):
        spec = specs[function.key] if method == "spec" else default_spec(function)
        if count > 1:
            spec = parallel_spec(spec)
        asked: dict[str, dict[str, float]] = {}
        heard: dict[str, float] = {}
        rounds = [argument_questions(example.query, function, spec)]
        for questions in rounds:
            for batch in question_batches(questions):
                response = client.system_one(state, batch, model=MODEL)
                asked |= {
                    name: answer.probabilities
                    for name, answer in response.choices.items()
                }
                heard |= {name: answer.noul for name, answer in response.nouls.items()}
                argument_input_tokens += input_tokens(response)
            if count > 1 and len(rounds) == 1:
                rounds.append(value_questions(example.query, function, spec, asked))
        arguments = (
            decode_parallel(example.query, function, spec, asked, heard)
            if count > 1
            else [decode_arguments(example.query, function, spec, asked, heard)]
        )
        calls.extend(Call(name=function.name, arguments=each) for each in arguments)
        choices |= {f"{position}/{name}": value for name, value in asked.items()}
        nouls |= {f"{position}/{name}": value for name, value in heard.items()}
    return Prediction(
        example=example,
        method=method,
        calls=tuple(calls),
        tool_probabilities=choices["tool"],
        choices=choices,
        nouls=nouls,
        tool_input_tokens=input_tokens(selection),
        argument_input_tokens=argument_input_tokens,
        model=selection.model,
    )


def selection_questions(functions: Sequence[Function]) -> dict[str, Choice | Noul]:
    """Build the questions that decide which functions to call and how often.

    The tool question picks one function or abstains. Each function also
    gets a question for how many times it must be called. When several
    functions are offered, one question asks how many different ones are
    needed and each gets a yes/no for whether it is.
    """
    questions: dict[str, Choice | Noul] = {"tool": tool_question(functions)}
    several = len(functions) > 1
    if several:
        questions["distinct"] = Choice(
            instructions={
                "functions": {f.name: f.summary for f in functions},
                "question": (
                    "How many DIFFERENT functions from `functions` must be called "
                    "to do everything `query` asks? Calling one function several "
                    "times still counts as one."
                ),
            },
            criteria=dict.fromkeys(
                str(n) for n in range(1, min(len(functions), MAX_CALLS) + 1)
            ),
        )
    for index, function in enumerate(functions):
        about = {"function": function.name, **function.summary}
        if several:
            questions[f"needed.{index}"] = Noul(
                instructions={
                    **about,
                    "question": (
                        "Does doing everything `query` asks require calling `function`?"
                    ),
                }
            )
        questions[f"count.{index}"] = Choice(
            instructions={
                **about,
                "question": (
                    "How many separate times must `function` be called to do "
                    "everything `query` asks? Each distinct item, case, or set "
                    "of values needs its own call."
                ),
            },
            criteria=dict.fromkeys(str(n) for n in range(1, MAX_CALLS + 1)),
        )
    return questions


def select_calls(
    functions: Sequence[Function],
    choices: Mapping[str, Mapping[str, float]],
    nouls: Mapping[str, float],
) -> list[tuple[Function, int]]:
    """Decide which functions to call and how many times each.

    The tool question's pick is called once unless the other answers are
    confident otherwise, so a query needing one call is not disturbed by a
    doubtful count. Several functions are selected only when the chance
    that one is enough falls below ONE_FUNCTION_THRESHOLD: then as many as
    the model said, taking those most likely to be needed. A function is
    called more than once only when the chance of one call falls below
    ONE_CALL_THRESHOLD.

    The tool question asks for one function, so it answers none when a
    request needs two. Abstaining therefore stands only if several
    functions are not confidently needed, or fewer than two are each judged
    needed.

    Returns:
        Each selected function with its number of calls, in the order
        offered. Empty if the model abstained.
    """

    def top(options: Mapping[str, float]) -> str:
        return max(options, key=lambda option: options[option])

    def more_than_one(options: Mapping[str, float], threshold: float) -> int:
        if options.get("1", 0.0) >= threshold:
            return 1
        return int(top({k: v for k, v in options.items() if k != "1"}))

    tool = top(choices["tool"])
    picked = [index for index, f in enumerate(functions) if f.name == tool]
    if "distinct" in choices:
        wanted = more_than_one(choices["distinct"], ONE_FUNCTION_THRESHOLD)
        ranked = sorted(
            range(len(functions)), key=lambda index: -nouls[f"needed.{index}"]
        )
        needed = [index for index in ranked if nouls[f"needed.{index}"] >= THRESHOLD]
        if wanted > 1 and (tool != NONE or len(needed) > 1):
            picked = sorted(ranked[:wanted])
    return [
        (
            functions[index],
            more_than_one(choices[f"count.{index}"], ONE_CALL_THRESHOLD),
        )
        for index in picked
    ]


def request_state(example: Example) -> dict[str, JsonValue]:
    """Build the state every question about an example is asked against.

    It is the query, with any earlier messages as context: a system prompt
    or previous turn is shown to the model, but only the query is labelled.
    """
    state: dict[str, JsonValue] = {"query": example.query}
    if example.context:
        state["context"] = [
            {"role": role, "content": content} for role, content in example.context
        ]
    return state


def question_batches(
    questions: Mapping[str, Choice | Noul],
) -> list[dict[str, Choice | Noul]]:
    """Split questions into requests of at most BATCH_CHARACTERS each.

    One request has a limit on its questions' combined size, which a long
    query's per-word questions exceed. Order is kept, and no questions
    means no requests.
    """
    batches: list[dict[str, Choice | Noul]] = []
    size = BATCH_CHARACTERS
    for name, question in questions.items():
        length = len(question.model_dump_json())
        if size + length > BATCH_CHARACTERS:
            batches.append({})
            size = 0
        batches[-1][name] = question
        size += length
    return batches


def tool_question(functions: Sequence[Function]) -> Choice:
    """Build the question that picks one function or abstains.

    Each function is offered with its description and its arguments'
    descriptions.

    Raises:
        ValueError: If a function is named like the abstain option.
    """
    criteria = {function.name: function.summary for function in functions}
    if NONE in criteria:
        raise ValueError(f"Function '{NONE}' collides with the abstain option")
    return Choice(
        instructions=(
            "Which function answers the request in `query`? "
            f"Answer {NONE} if no offered function can."
        ),
        criteria={**criteria, NONE: NO_TOOL},
    )


def argument_questions(
    query: str, function: Function, spec: FunctionSpec
) -> dict[str, Choice | Noul]:
    """Build every question needed to fill one function's arguments.

    Each parameter is asked the way its spec entry says:

    - text, and the city of a place: one question per word of the query,
      shared by all such parameters, asking which of them the word supplies.
      The word and up to CONTEXT_WORDS words on either side of it are
      labelled fields, so a long query's questions do not each repeat it, and
      the none option says that a word naming or introducing an argument is
      none: without that, the model labels "base" in "a base of 10" too.
    - number: one Choice over the numbers code found in the query.
    - options: one Choice over the listed values, each with its line; an
      open list adds an option for a value outside it, which is then copied
      from the query like text.
    - flag: one Noul. set: one Noul per member.
    - date: one Choice each for the month, day, and year.
    - place: one Choice each for the US state and the country.

    An optional parameter whose entry has a stated question also gets a Noul
    asking whether the query mentions it at all.

    Raises:
        ValueError: If the spec does not fit the function, or a word-labelled
            parameter is named like the none option.
    """
    check(function, spec)
    questions: dict[str, Choice | Noul] = {}
    labelled = {
        parameter.name: parameter.description
        for parameter in word_function(function, spec).parameters
    }
    if NONE in labelled:
        raise ValueError(f"Parameter '{NONE}' collides with the none option")
    if labelled:
        words = [query[start:end] for start, end in word_spans(query)]
        for index, word in enumerate(words):
            questions[f"token_{index}"] = Choice(
                instructions={
                    "function": function.name,
                    "words_before": " ".join(
                        words[max(0, index - CONTEXT_WORDS) : index]
                    ),
                    "word": word,
                    "words_after": " ".join(
                        words[index + 1 : index + 1 + CONTEXT_WORDS]
                    ),
                    "question": (
                        "Which argument of `function` does `word` supply in "
                        f"`query`? Answer {NONE} if it supplies no argument."
                    ),
                },
                criteria={**labelled, NONE: NO_ARGUMENT},
            )
    for parameter in function.parameters:
        entry = spec.parameters[parameter.name]
        questions |= parameter_questions(query, function, parameter, entry)
    return questions


def parameter_questions(
    query: str, function: Function, parameter: Parameter, entry: ParameterSpec
) -> dict[str, Choice | Noul]:
    """Build the questions asked directly about one parameter."""
    name = parameter.name
    about = {
        "function": function.name,
        "argument": name,
        "description": parameter.description,
    }
    asked = {**about, "question": entry.question}
    questions: dict[str, Choice | Noul] = {}
    match entry.kind:
        case "number":
            questions |= number_question(query, parameter, asked)
        case "options":
            other = {OTHER: "The request names a value that is not one of these."}
            questions[f"choice.{name}"] = Choice(
                instructions=asked,
                criteria={**entry.options, **(other if entry.open else {})},
            )
        case "flag":
            questions[f"flag.{name}"] = Noul(instructions=asked)
        case "set":
            questions |= {
                f"member.{name}.{value}": Noul(
                    instructions={
                        **about,
                        "value": value,
                        **({} if line is None else {"meaning": line}),
                        "question": entry.question,
                    }
                )
                for value, line in entry.options.items()
            }
        case "date":
            questions |= {
                f"date.{name}.{part}": Choice(
                    instructions={**asked, "part": question},
                    criteria={
                        **dict.fromkeys(options),
                        NONE: f"The request does not give the {part}.",
                    },
                )
                for part, (question, options) in DATE_PARTS.items()
            }
        case "place":
            questions |= place_questions(name, about)
    if entry.stated is not None and entry.kind in ("number", "options", "flag", "date"):
        questions[f"stated.{name}"] = Noul(
            instructions={**about, "question": entry.stated}
        )
    return questions


def number_question(
    query: str, parameter: Parameter, asked: Mapping[str, str | None]
) -> dict[str, Choice | Noul]:
    """Build the question that picks a number among those found in the query.

    Nothing is asked when the query holds no number the parameter could take.
    """
    candidates = number_candidates(query, parameter.type)
    if not candidates:
        return {}
    return {
        f"number.{parameter.name}": Choice(
            instructions=dict(asked),
            criteria={
                **{
                    option: f"written as {source}" if source else None
                    for option, source in candidates.items()
                },
                NONE: NO_NUMBER,
            },
        )
    }


def place_questions(
    name: str, about: Mapping[str, str | None]
) -> dict[str, Choice | Noul]:
    """Build the questions that place a named city in a US state and a country."""
    return {
        f"place.{name}.state": Choice(
            instructions={
                **about,
                "question": "Which US state is the place named for `argument` in?",
            },
            criteria={
                **STATES,
                NONE: "The place is not in a US state, or none is named.",
            },
        ),
        f"place.{name}.country": Choice(
            instructions={
                **about,
                "question": "Which country is the place named for `argument` in?",
            },
            criteria={
                **dict.fromkeys([UNITED_STATES, *COUNTRIES]),
                NONE: "No place is named.",
            },
        ),
    }


def word_function(function: Function, spec: FunctionSpec) -> Function:
    """Return the function with only the parameters filled by labelling words.

    Those are text, the city of a place, and a string with an open list of
    options, whose unlisted values are copied from the query.
    """
    return function.model_copy(
        update={
            "parameters": tuple(
                parameter
                for parameter in function.parameters
                if spec.parameters[parameter.name].kind in ("text", "place")
                or (
                    spec.parameters[parameter.name].kind == "options"
                    and spec.parameters[parameter.name].open
                    and parameter.type == "string"
                )
            )
        }
    )


def decode_arguments(
    query: str,
    function: Function,
    spec: FunctionSpec,
    choices: Mapping[str, Mapping[str, float]],
    nouls: Mapping[str, float],
) -> dict[str, JsonValue]:
    """Turn the argument answers into the function's arguments.

    An argument is left out when its value cannot be read or assembled,
    when an optional argument is not stated, or when a set has no members.

    Args:
        query: The request text.
        function: The function being called.
        spec: How each of its parameters was asked.
        choices: Each Choice question's option probabilities, by question name.
        nouls: Each Noul question's yes-probability, by question name.
    """
    words = word_function(function, spec)
    runs = word_runs(query, words, choices) if words.parameters else {}
    arguments: dict[str, JsonValue] = {}
    for parameter in function.parameters:
        name = parameter.name
        entry = spec.parameters[name]
        if f"stated.{name}" in nouls and nouls[f"stated.{name}"] < THRESHOLD:
            continue
        copied = open_value(
            parameter, open_texts(query, parameter, runs.get(name, []), choices)
        )
        value = decode_value(parameter, entry, copied, choices, nouls)
        if value is not None:
            arguments[name] = value
    return arguments


def typed(option: str, value_type: str) -> JsonValue:
    """Read a chosen option as its parameter's type, or keep it as written.

    An integer parameter's options are usually numbers but can be a word
    such as "dontcare", which is passed through.
    """
    value = coerce(option, value_type)
    return option if value is None else value


def decode_value(
    parameter: Parameter,
    entry: ParameterSpec,
    copied: JsonValue,
    choices: Mapping[str, Mapping[str, float]],
    nouls: Mapping[str, float],
) -> JsonValue:
    """Read one argument's value from its answers, or None to omit it.

    Args:
        parameter: The parameter being filled.
        entry: How it was asked.
        copied: The value copied from the words labelled with it, if any.
        choices: Each Choice question's option probabilities, by question name.
        nouls: Each Noul question's yes-probability, by question name.
    """
    name = parameter.name

    def chosen(question: str) -> str:
        options = choices[question]
        return max(options, key=lambda option: options[option])

    match entry.kind:
        case "text":
            return copied
        case "number":
            number = chosen(f"number.{name}") if f"number.{name}" in choices else NONE
            return None if number == NONE else typed(number, parameter.type)
        case "options":
            option = chosen(f"choice.{name}")
            return (
                copied
                if entry.open and option == OTHER
                else typed(option, parameter.type)
            )
        case "flag":
            return nouls[f"flag.{name}"] >= THRESHOLD
        case "set":
            members: list[JsonValue] = [
                value
                for value in entry.options
                if nouls[f"member.{name}.{value}"] >= THRESHOLD
            ]
            return members or None
        case "date":
            return format_date(
                chosen(f"date.{name}.month"),
                chosen(f"date.{name}.day"),
                chosen(f"date.{name}.year"),
                str(entry.format),
            )
        case "place":
            if not isinstance(copied, str):
                return None
            return complete_place(
                copied,
                chosen(f"place.{name}.state"),
                chosen(f"place.{name}.country"),
                str(entry.place_format),
                str(entry.no_state),
            )
        case _:
            return None


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


def open_texts(
    query: str,
    parameter: Parameter,
    runs: Sequence[tuple[int, int]],
    choices: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Return the texts an open-valued parameter's value is read from.

    A scalar string has one: the run with the most probability behind it,
    so a stray labelled word elsewhere in the query does not win by coming
    first. Every other parameter has one text per run, in query order.
    """
    spans = word_spans(query)
    if parameter.type == "string" and runs:
        runs = [
            max(
                runs,
                key=lambda run: sum(
                    choices[f"token_{index}"].get(parameter.name, 0.0)
                    for index in range(run[0], run[1] + 1)
                ),
            )
        ]
    return [query[spans[first][0] : spans[last][1]] for first, last in runs]


def word_runs(
    query: str,
    function: Function,
    choices: Mapping[str, Mapping[str, float]],
    fill: bool = True,
) -> dict[str, list[tuple[int, int]]]:
    """Group labelled words into each open-valued parameter's runs.

    A run is the index of its first and last word. A word's label is its
    most probable option, and counts only at LABEL_THRESHOLD or above: the
    words that merely introduce a value are the ones the model is unsure of.

    For a string parameter, adjacent words with the same label are one run.
    For a scalar string, unlabelled words between two words with its label
    join the run in two cases: up to MAX_GAP of them, so small words inside
    a name stay in it; and up to MAX_BRIDGE of them if the model gave each
    at least BRIDGE_THRESHOLD for the label, so a hyphen or comma it was
    unsure of does not cut a value in two. An array gets neither, or its
    elements would merge. For a number parameter every labelled word is its
    own run, so a labelled word beside the number cannot spoil it.

    With `fill` off, as when one function is called several times, no gap
    is filled, since that would join two calls' values across an "and". A
    hyphen or apostrophe between two words with the same label still joins
    them.
    """
    labels = word_labels(query, function, choices, fill)
    strings = {p.name for p in function.parameters if p.value_type == "string"}
    runs: dict[str, list[tuple[int, int]]] = {}
    position = 0
    for label, group in groupby(labels):
        length = len(list(group))
        if label != NONE:
            runs.setdefault(label, []).extend(
                [(position, position + length - 1)]
                if label in strings
                else [(index, index) for index in range(position, position + length)]
            )
        position += length
    return runs


def word_labels(
    query: str,
    function: Function,
    choices: Mapping[str, Mapping[str, float]],
    fill: bool,
) -> list[str]:
    """Return each word's label after thresholding, joining, and gap filling.

    See `word_runs` for the rules.
    """
    answers = [choices[f"token_{index}"] for index in range(len(word_spans(query)))]
    labels = []
    for probabilities in answers:
        label = max(probabilities, key=lambda option: probabilities[option])
        labels.append(label if probabilities[label] >= LABEL_THRESHOLD else NONE)
    scalar_strings = {p.name for p in function.parameters if p.type == "string"}
    words = [query[start:end] for start, end in word_spans(query)]
    for index in range(1, len(labels) - 1):
        if (
            not fill
            and labels[index] == NONE
            and words[index] in JOINERS
            and labels[index - 1] == labels[index + 1] != NONE
        ):
            labels[index] = labels[index - 1]
    for start, label in enumerate(labels):
        if not fill or label not in scalar_strings:
            continue
        for end in range(start + 2, min(start + MAX_BRIDGE + 2, len(labels))):
            gap = range(start + 1, end)
            if labels[end] != label or any(labels[index] != NONE for index in gap):
                continue
            if len(gap) <= MAX_GAP or all(
                answers[index].get(label, 0.0) >= BRIDGE_THRESHOLD for index in gap
            ):
                labels[start + 1 : end] = [label] * len(gap)
            break
    return labels


def parallel_spec(spec: FunctionSpec) -> FunctionSpec:
    """Return a spec for a function that will be called several times.

    Picking one number per argument is not enough when each call has its
    own, so a number is labelled in the query like text and every labelled
    number is read.
    """
    return spec.model_copy(
        update={
            "parameters": {
                name: entry.model_copy(update={"kind": "text"})
                if entry.kind == "number"
                else entry
                for name, entry in spec.parameters.items()
            }
        }
    )


def value_questions(
    query: str,
    function: Function,
    spec: FunctionSpec,
    choices: Mapping[str, Mapping[str, float]],
) -> dict[str, Choice | Noul]:
    """Build the questions asked about single labelled values.

    A labelled number that can be read two ways, such as "30%" as 30 or
    0.3, gets a Choice between its readings. A labelled city of a place
    parameter gets the state and country questions for that city.
    """
    labelled = word_function(function, spec)
    if not labelled.parameters:
        return {}
    spans = word_spans(query)
    questions: dict[str, Choice | Noul] = {}
    runs = word_runs(query, labelled, choices, fill=False)
    for parameter in labelled.parameters:
        name = parameter.name
        about = {
            "function": function.name,
            "argument": name,
            "description": parameter.description,
        }
        for first, last in runs.get(name, []):
            written = query[spans[first][0] : spans[last][1]]
            if spec.parameters[name].kind == "place":
                questions |= {
                    f"place.{name}.{first}.{part}": question.model_copy(
                        update={"instructions": {**about, "place": written}}
                    )
                    for part, question in (
                        ("state", place_questions(name, about)[f"place.{name}.state"]),
                        (
                            "country",
                            place_questions(name, about)[f"place.{name}.country"],
                        ),
                    )
                }
            elif parameter.value_type != "string":
                readings = number_readings(
                    query, spans, first, str(parameter.value_type)
                )
                if len(readings) > 1:
                    questions[f"reading.{name}.{first}"] = Choice(
                        instructions={
                            **about,
                            "written": written,
                            "question": (
                                "Which of these is the value to pass for "
                                "`argument`, in the unit and form the function "
                                "expects?"
                            ),
                        },
                        criteria={
                            reading: f"written as {source}"
                            if source
                            else "the number exactly as written"
                            for reading, source in readings.items()
                        },
                    )
    return questions


def decode_parallel(
    query: str,
    function: Function,
    spec: FunctionSpec,
    choices: Mapping[str, Mapping[str, float]],
    nouls: Mapping[str, float],
) -> list[dict[str, JsonValue]]:
    """Turn one function's answers into the arguments of several calls.

    Every value labelled for a parameter is kept, in query order, and the
    values are zipped by position: the first of each goes to the first
    call. A parameter with one value, or the same value repeated, shares it
    with every call, as do arguments that are not labelled words. An
    array's elements stay together while only commas or "and" separate
    them, and a longer gap starts the next call's array.

    Args:
        query: The request text.
        function: The function being called.
        spec: How each of its parameters was asked.
        choices: Each Choice question's option probabilities, by question name.
        nouls: Each Noul question's yes-probability, by question name.

    Returns:
        One argument mapping per call, at least one.
    """
    labelled = word_function(function, spec)
    runs = (
        word_runs(query, labelled, choices, fill=False) if labelled.parameters else {}
    )
    words = [query[start:end] for start, end in word_spans(query)]
    per_call: dict[str, list[JsonValue]] = {}
    shared: dict[str, JsonValue] = {}
    for parameter in function.parameters:
        name = parameter.name
        entry = spec.parameters[name]
        if f"stated.{name}" in nouls and nouls[f"stated.{name}"] < THRESHOLD:
            continue
        if entry.kind not in ("text", "place"):
            value = decode_value(parameter, entry, None, choices, nouls)
            if value is not None:
                shared[name] = value
            continue
        found = [
            (
                first,
                last,
                labelled_value(query, parameter, entry, (first, last), choices),
            )
            for first, last in runs.get(name, [])
        ]
        found = [
            (first, last, value) for first, last, value in found if value is not None
        ]
        per_call[name] = values_per_call(parameter, found, words)
    count = max([len(values) for values in per_call.values()] + [1])
    calls = []
    for index in range(count):
        arguments = dict(shared)
        for name, values in per_call.items():
            if len(values) == 1:
                arguments[name] = values[0]
            elif index < len(values):
                arguments[name] = values[index]
        calls.append(arguments)
    return calls


def values_per_call(
    parameter: Parameter,
    found: Sequence[tuple[int, int, JsonValue]],
    words: Sequence[str],
) -> list[JsonValue]:
    """Arrange one parameter's labelled values as one value per call.

    A scalar's values are taken in order, and collapse to one if they are
    all the same. An array's elements stay in one array while only commas
    or "and" separate them, and a longer gap starts the next call's array.

    Args:
        parameter: The parameter the values were labelled for.
        found: Each value with the indexes of its first and last word.
        words: The query's words.
    """
    if parameter.type != "array":
        values = [value for _, _, value in found]
        return values[:1] if len(set(map(str, values))) == 1 else values
    groups: list[tuple[int, list[JsonValue]]] = []
    for first, last, value in found:
        glued = groups and all(
            word.lower() in LIST_GLUE for word in words[groups[-1][0] + 1 : first]
        )
        if glued:
            groups[-1] = (last, [*groups[-1][1], value])
        else:
            groups.append((last, [value]))
    return [group for _, group in groups]


def labelled_value(
    query: str,
    parameter: Parameter,
    entry: ParameterSpec,
    run: tuple[int, int],
    choices: Mapping[str, Mapping[str, float]],
) -> JsonValue:
    """Read the value of one run of labelled words, or None if it is not one.

    A number takes the reading chosen for it, or its only reading. A city
    of a place parameter is completed with the state or country chosen for
    it.
    """
    spans = word_spans(query)
    first, last = run
    name = parameter.name

    def chosen(question: str) -> str:
        options = choices[question]
        return max(options, key=lambda option: options[option])

    if parameter.value_type != "string":
        readings = number_readings(query, spans, first, str(parameter.value_type))
        if not readings:
            return None
        reading = (
            chosen(f"reading.{name}.{first}")
            if f"reading.{name}.{first}" in choices
            else next(iter(readings))
        )
        return typed(reading, str(parameter.value_type))
    text = coerce(query[spans[first][0] : spans[last][1]], "string")
    if entry.kind != "place" or not isinstance(text, str):
        return text
    if f"place.{name}.{first}.state" not in choices:
        return text
    return complete_place(
        text,
        chosen(f"place.{name}.{first}.state"),
        chosen(f"place.{name}.{first}.country"),
        str(entry.place_format),
        str(entry.no_state),
    )


def input_tokens(response: SystemOneResponse) -> int:
    """Return a response's input token count.

    Raises:
        ValueError: If the response carries no usage.
    """
    if response.usage.input_tokens is None:
        raise ValueError("Jev response has no input token usage")
    return response.usage.input_tokens
