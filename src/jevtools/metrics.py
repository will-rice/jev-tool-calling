"""Score predicted calls with BFCL's AST-match rule."""

import re
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import get_args

from pydantic import JsonValue

from jevtools.jev import coerce, word_spans
from jevtools.models import Call, Example, Parameter, Prediction, Split


def evaluate(predictions: Sequence[Prediction]) -> dict[str, float]:
    """Score one run's predictions, split by split.

    Call accuracy is BFCL's AST match, or abstention on irrelevance. Tool
    accuracy is the share of queries that picked the gold function. The
    ceiling is the share of queries whose every needed argument could be
    produced by labelling words.

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
            matches(p.example, p.call) for p in rows
        )
        if split == "irrelevance":
            continue
        metrics[f"{split}/tool_accuracy"] = fmean(
            p.call is not None and p.call.name == p.example.target[0].name for p in rows
        )
        metrics[f"{split}/argument_accuracy"] = argument_accuracy(rows)
        metrics[f"{split}/ceiling"] = fmean(reachable(p.example) for p in rows)
    return metrics


def argument_accuracy(predictions: Sequence[Prediction]) -> float:
    """Score arguments on the queries that picked the gold function.

    Only arguments the gold answer does not allow to be omitted are scored.

    Raises:
        ValueError: If no query picked its gold function.
    """
    scored = []
    for prediction in predictions:
        function, accepted = prediction.example.target
        call = prediction.call
        if call is None or call.name != function.name:
            continue
        scored.extend(
            name in call.arguments and accepts(call.arguments[name], values)
            for name, values in accepted.items()
            if "" not in values
        )
    if not scored:
        raise ValueError("No arguments to score: no query picked its gold function")
    return fmean(scored)


def matches(example: Example, call: Call | None) -> bool:
    """Decide whether a predicted call is correct under BFCL's AST match.

    With no gold call, the correct prediction is no call, except on a
    relevance split, where it is any call.
    """
    if example.split == "live_relevance":
        return call is not None
    if example.gold is None:
        return call is None
    function, accepted = example.target
    if call is None or call.name != function.name:
        return False
    if any(p.required and p.name not in call.arguments for p in function.parameters):
        return False
    if any(
        name not in accepted or not accepts(value, accepted[name])
        for name, value in call.arguments.items()
    ):
        return False
    return all(
        name in call.arguments or "" in values for name, values in accepted.items()
    )


def reachable(example: Example) -> bool:
    """Decide whether every needed gold argument can be produced from words."""
    function, accepted = example.target
    parameters = {parameter.name: parameter for parameter in function.parameters}
    return all(
        expressible(example.query, parameters[name], values)
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
