"""Per-function specs saying how each argument is asked for.

A spec is the function-calling cookbook's idea applied to a schema: a
plain-language question per argument, a described line per option, and a
question asking whether an optional argument is stated at all. It also says
which open values code can find or assemble: numbers, dates, and places.
"""

import json
from typing import Literal

from pydantic import BaseModel

from jevtools.config import MAX_OPTIONS, SPEC_DIR
from jevtools.models import NUMBERS, Function, Stage

SpecKind = Literal["text", "number", "options", "flag", "set", "date", "place", "skip"]
PlaceFormat = Literal["city, state_abbr", "city, state_name", "city, country", "city"]
CHOICE_QUESTION = "Which value of `argument` does `query` ask for?"
FLAG_QUESTION = "Does `query` ask for `argument` to be true?"
MEMBER_QUESTION = "Does `query` ask for `value` in `argument`?"
STATED_QUESTION = "Does `query` say anything about `argument`?"


class ParameterSpec(BaseModel, frozen=True):
    """How one argument is asked for and assembled.

    `options` maps each value the function accepts to a line describing it,
    for the options and set kinds; `open` says a value outside them is also
    possible. `format` is a date's strftime format. `place_format` and
    `no_state` say how a place is written in and outside a US state.
    """

    kind: SpecKind
    question: str
    stated: str | None = None
    options: dict[str, str | None] = {}
    open: bool = False
    format: str | None = None
    place_format: PlaceFormat | None = None
    no_state: Literal["city, country", "city"] | None = None


class FunctionSpec(BaseModel, frozen=True):
    """One function's spec: an entry per parameter."""

    name: str
    parameters: dict[str, ParameterSpec]


def load_specs(stage: Stage) -> dict[str, FunctionSpec]:
    """Load a stage's authored specs, keyed by function key."""
    entries = json.loads((SPEC_DIR / f"{stage}.json").read_text())
    return {key: FunctionSpec.model_validate(entry) for key, entry in entries.items()}


def default_spec(function: Function) -> FunctionSpec:
    """Derive a spec from the schema alone.

    Open values are copied from the request, enums and booleans are asked
    with a generic question, and types that cannot be asked are skipped.
    """
    parameters = {}
    for parameter in function.parameters:
        stated = None if parameter.required else STATED_QUESTION
        match parameter.kind:
            case "words":
                entry = ParameterSpec(kind="text", question=CHOICE_QUESTION)
            case "choice":
                entry = ParameterSpec(
                    kind="options",
                    question=CHOICE_QUESTION,
                    stated=stated,
                    options=dict.fromkeys(parameter.enum),
                )
            case "flag":
                entry = ParameterSpec(
                    kind="flag", question=FLAG_QUESTION, stated=stated
                )
            case "set":
                entry = ParameterSpec(
                    kind="set",
                    question=MEMBER_QUESTION,
                    options=dict.fromkeys(parameter.enum),
                )
            case _:
                entry = ParameterSpec(kind="skip", question=CHOICE_QUESTION)
        parameters[parameter.name] = entry
    return FunctionSpec(name=function.name, parameters=parameters)


def check(function: Function, spec: FunctionSpec) -> None:
    """Reject a spec that does not fit its function.

    Raises:
        ValueError: If a parameter has no entry, its kind does not fit its
            type, a field the kind needs is missing, or it has too many options.
    """
    for parameter in function.parameters:
        where = f"{function.name}.{parameter.name}"
        if parameter.name not in spec.parameters:
            raise ValueError(f"{where} has no spec entry")
        entry = spec.parameters[parameter.name]
        fits = {
            "text": parameter.kind == "words",
            "number": parameter.type in NUMBERS,
            "options": parameter.type in ("string", "integer") and bool(entry.options),
            "flag": parameter.type == "boolean",
            "set": parameter.type == "array" and bool(entry.options),
            "date": parameter.type == "string" and entry.format is not None,
            "place": parameter.type == "string"
            and entry.place_format is not None
            and entry.no_state is not None,
            "skip": True,
        }[entry.kind]
        if not fits:
            raise ValueError(f"{where} cannot be asked as {entry.kind}")
        if len(entry.options) > MAX_OPTIONS - 1:
            raise ValueError(f"{where} has more than {MAX_OPTIONS - 1} options")
