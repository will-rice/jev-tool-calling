"""Score predicted calls with BFCL's AST-match rule."""

import re
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import get_args

from pydantic import JsonValue

from jevtools.config import PARALLEL, UNANSWERED
from jevtools.jev import coerce, word_spans
from jevtools.models import (
    Call,
    Example,
    Function,
    Gold,
    Parameter,
    Prediction,
    Split,
)


def evaluate(predictions: Sequence[Prediction]) -> dict[str, float]:
    """Score one run's predictions, split by split.

    Call accuracy is BFCL's AST match over every call of a query, or the
    right decision to call or not on a split with no gold calls. A
    single-call split also reports tool accuracy (the share of queries that
    made one call, to the gold function), argument accuracy, and the ceiling
    (the share of queries whose every needed argument could be produced by
    labelling words). A parallel split reports count accuracy, the share of
    queries with the right number of calls.

    Raises:
        ValueError: If there are no predictions.
    """
    if not predictions:
        raise ValueError("No predictions to evaluate")
    metrics: dict[str, float] = {}
    for split in get_args(Split):
        rows = [p for p in predictions if p.example.split == split]
        if not rows:
            continue
        metrics[f"{split}/call_accuracy"] = fmean(
            matches(p.example, p.calls) for p in rows
        )
        if split in UNANSWERED:
            continue
        if split in PARALLEL:
            metrics[f"{split}/count_accuracy"] = fmean(
                len(p.calls) == len(p.example.gold) for p in rows
            )
            continue
        metrics[f"{split}/tool_accuracy"] = fmean(
            [call.name for call in p.calls] == [p.example.target[0].name] for p in rows
        )
        metrics[f"{split}/argument_accuracy"] = argument_accuracy(rows)
        metrics[f"{split}/ceiling"] = fmean(reachable(p.example) for p in rows)
    return metrics


def argument_accuracy(predictions: Sequence[Prediction]) -> float:
    """Score arguments on the queries that made one call, to the gold function.

    Only arguments the gold answer does not allow to be omitted are scored.

    Raises:
        ValueError: If no query made exactly that call.
    """
    scored = []
    for prediction in predictions:
        function, accepted = prediction.example.target
        if [call.name for call in prediction.calls] != [function.name]:
            continue
        [call] = prediction.calls
        scored.extend(
            name in call.arguments and accepts(call.arguments[name], values)
            for name, values in accepted.items()
            if "" not in values
        )
    if not scored:
        raise ValueError("No arguments to score: no query picked its gold function")
    return fmean(scored)


def matches(example: Example, calls: Sequence[Call]) -> bool:
    """Decide whether the predicted calls are correct under BFCL's AST match.

    There must be as many calls as gold calls, and each gold call must be
    matched by a different predicted call, in any order. With no gold call
    the correct prediction is no call, except on a relevance split, where it
    is any call.
    """
    if example.split == "live_relevance":
        return len(calls) > 0
    if len(calls) != len(example.gold):
        return False
    remaining = list(calls)
    for gold in example.gold:
        function = example.function(gold.name)
        match = next(
            (call for call in remaining if call_matches(function, call, gold)), None
        )
        if match is None:
            return False
        remaining.remove(match)
    return True


def call_matches(function: Function, call: Call, gold: Gold) -> bool:
    """Decide whether one call matches one gold call.

    The name is right, every required parameter is present, every argument
    has accepted values and its value is one of them, and every gold
    argument is present unless it may be omitted.
    """
    if call.name != function.name:
        return False
    if any(p.required and p.name not in call.arguments for p in function.parameters):
        return False
    if any(
        name not in gold.accepted or not accepts(value, gold.accepted[name])
        for name, value in call.arguments.items()
    ):
        return False
    return all(
        name in call.arguments or "" in values for name, values in gold.accepted.items()
    )


def reachable(example: Example) -> bool:
    """Decide whether every needed gold argument can be produced from words.

    A gold argument the function does not have cannot be.
    """
    function, accepted = example.target
    parameters = {parameter.name: parameter for parameter in function.parameters}
    return all(
        name in parameters and expressible(example.query, parameters[name], values)
        for name, values in accepted.items()
        if "" not in values
    )


def expressible(
    query: str, parameter: Parameter, accepted: Sequence[JsonValue]
) -> bool:
    """Decide whether some labelling of the query's words yields an accepted value.

    A closed-set argument always can; a type that is never asked never can.
    An open value can if a run of consecutive words reads as an accepted
    value, and an array if that holds for every element of an accepted list.
    Overlap between arguments is ignored, so this is an upper bound.
    """
    if parameter.kind != "words":
        return parameter.kind is not None
    spans = word_spans(query)
    candidates = [
        value
        for index, (start, _) in enumerate(spans)
        for _, end in spans[index:]
        if (value := coerce(query[start:end], parameter.value_type)) is not None
    ]
    if parameter.type != "array":
        return any(accepts(value, accepted) for value in candidates)
    return any(
        isinstance(option, list)
        and len(option) > 0
        and all(
            any(accepts(value, [element]) for value in candidates) for element in option
        )
        for option in accepted
    )


def accepts(value: JsonValue, accepted: Sequence[JsonValue]) -> bool:
    """Decide whether a value is one of an argument's accepted values.

    Strings are compared after BFCL's standardisation. BFCL also promotes an
    integer given for a float parameter; decoding always reads a float for
    one, and Python compares the two as equal, so nothing is needed here.
    """
    return standardize(value) in [standardize(option) for option in accepted]


def standardize(value: JsonValue) -> JsonValue:
    """Apply BFCL's string standardisation to a string or a list's strings.

    Spaces and `,./-_*^` are removed, case is folded, and single quotes
    become double quotes.
    """
    if isinstance(value, str):
        return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')
    if isinstance(value, list):
        return [standardize(item) for item in value]
    return value


def summarize(
    runs: Sequence[Mapping[str, float]],
) -> dict[str, tuple[float, float, float]]:
    """Return each metric's mean, minimum, and maximum across runs."""
    return {
        name: (
            fmean(run[name] for run in runs),
            min(run[name] for run in runs),
            max(run[name] for run in runs),
        )
        for name in runs[0]
    }
