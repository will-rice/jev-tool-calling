"""Pydantic models for parsed data and saved predictions."""

from hashlib import sha1
from typing import Literal, Self

from pydantic import BaseModel, JsonValue, model_validator

Split = Literal[
    "simple",
    "multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_irrelevance",
    "live_relevance",
]
Stage = Literal["dev", "test"]
Method = Literal["words", "spec"]
Kind = Literal["words", "choice", "flag", "set"]
NUMBERS = ("integer", "float")
SCALARS = ("string", *NUMBERS)


class Parameter(BaseModel, frozen=True):
    """One parameter of an offered function.

    For an array, `enum` and `item_type` describe its items.
    """

    name: str
    type: str
    description: str | None
    required: bool
    enum: tuple[str, ...] = ()
    item_type: str | None = None

    @property
    def value_type(self) -> str | None:
        """The type of one value: an array's item type, else the type."""
        return self.item_type if self.type == "array" else self.type

    @property
    def kind(self) -> Kind | None:
        """How the parameter is asked, or None if it cannot be filled."""
        if self.type == "boolean":
            return "flag"
        if self.enum:
            return "set" if self.type == "array" else "choice"
        if self.value_type in SCALARS:
            return "words"
        return None


class Function(BaseModel, frozen=True):
    """A function offered to the model."""

    name: str
    description: str
    parameters: tuple[Parameter, ...]

    @property
    def key(self) -> str:
        """A stable identifier for this exact definition, used to find its spec."""
        return sha1(self.model_dump_json().encode()).hexdigest()[:16]


class Gold(BaseModel, frozen=True):
    """The expected call: each argument's accepted values.

    An accepted value of "" means the argument may be omitted.
    """

    name: str
    accepted: dict[str, list[JsonValue]]


class Example(BaseModel, frozen=True):
    """A query with the functions offered for it and its gold call, if any.

    The query is the last user message. `context` holds any messages before
    it as (role, content) pairs.
    """

    id: str
    split: Split
    query: str
    context: tuple[tuple[str, str], ...] = ()
    functions: tuple[Function, ...]
    gold: Gold | None

    @model_validator(mode="after")
    def check_gold_is_offered(self) -> Self:
        """Reject a gold call to a function the example does not offer."""
        offered = [function.name for function in self.functions]
        if self.gold is not None and self.gold.name not in offered:
            raise ValueError(
                f"Gold function {self.gold.name} is not offered: {offered}"
            )
        return self

    @property
    def target(self) -> tuple[Function, dict[str, list[JsonValue]]]:
        """The gold function and each of its arguments' accepted values.

        Raises:
            ValueError: If the example has no gold call.
        """
        if self.gold is None:
            raise ValueError(f"{self.id} has no gold call")
        name = self.gold.name
        function = next(
            function for function in self.functions if function.name == name
        )
        return function, self.gold.accepted


class Call(BaseModel, frozen=True):
    """A predicted function call."""

    name: str
    arguments: dict[str, JsonValue]


class Prediction(BaseModel, frozen=True):
    """The saved record for one query.

    `call` is None when the model abstained. `choices` and `nouls` hold the
    argument request's answers by question name.
    """

    example: Example
    method: Method
    call: Call | None
    tool_probabilities: dict[str, float]
    choices: dict[str, dict[str, float]]
    nouls: dict[str, float]
    tool_input_tokens: int
    argument_input_tokens: int
    model: str
