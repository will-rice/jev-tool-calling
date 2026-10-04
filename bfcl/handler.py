"""A BFCL model handler that answers with Jev.

BFCL's runner hands a handler one test entry at a time and scores what it
returns. This handler turns the entry into a `jevtools` example, runs the
same prediction used everywhere else in this project, and returns the calls
in the form the runner's checkers expect.
"""

import json
import time
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from dotenv import find_dotenv, load_dotenv
from typesafe_sdk import TypeSafeClient

from jevtools.data import parse_function
from jevtools.jev import predict
from jevtools.models import Example, Prediction
from jevtools.spec import default_spec, load_specs

PYTHON_HINT = " Note that the provided function is in Python 3 syntax."
UNSPECCED = ("simple_java", "simple_javascript")


class JevHandler(BaseHandler):
    """Answer BFCL test entries with Jev.

    The registry name's suffix picks the method: `jev-words` asks arguments
    from the schema alone, `jev-spec` from the authored specs.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.method = registry_name.removeprefix("jev-")
        load_dotenv(find_dotenv(usecwd=True))
        self.client = TypeSafeClient()
        self.specs = load_specs("dev") | load_specs("test")

    def _pre_query_processing_FC(  # noqa: N802
        self, inference_data: dict, test_entry: dict
    ) -> dict:
        """Start an entry with no messages."""
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """Parse the entry's function docs into the project's models.

        BFCL appends a note about Python syntax to every Python function's
        description. It is removed, so the function is the one its spec was
        written for and is found by its key.
        """
        inference_data["id"] = test_entry["id"]
        inference_data["functions"] = tuple(
            parse_function(
                {**doc, "description": doc["description"].removesuffix(PYTHON_HINT)}
            )
            for doc in test_entry["function"]
        )
        return inference_data

    def add_first_turn_message_FC(  # noqa: N802
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """Keep the turn's messages; the last one is the query."""
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _query_FC(self, inference_data: dict) -> tuple[Prediction, float]:  # noqa: N802
        """Predict the entry's calls with Jev and time it."""
        messages = inference_data["message"]
        example = Example(
            id=inference_data["id"],
            split="simple",
            query=messages[-1]["content"],
            context=tuple((m["role"], m["content"]) for m in messages[:-1]),
            functions=inference_data["functions"],
            gold=(),
        )
        # Java and JavaScript functions have no authored spec yet, so they are
        # asked from their schema alone. Any other function must have one.
        if example.id.startswith(UNSPECCED):
            for function in example.functions:
                self.specs.setdefault(function.key, default_spec(function))
        started = time.time()
        prediction = predict(self.client, example, self.method, self.specs)
        return prediction, time.time() - started

    def _parse_query_response_FC(self, api_response: Prediction) -> dict:  # noqa: N802
        """Report the calls and the tokens they cost."""
        return {
            "model_responses": [
                {call.name: json.dumps(call.arguments)} for call in api_response.calls
            ],
            "input_token": api_response.tool_input_tokens
            + api_response.argument_input_tokens,
            "output_token": 0,
        }

    def decode_ast(
        self, result: list[dict[str, str]], language: object, has_tool_call_tag: bool
    ) -> list[dict[str, dict]]:
        """Give the AST checker each call as its name and arguments."""
        return [
            {name: json.loads(arguments)}
            for call in result
            for name, arguments in call.items()
        ]

    def decode_execute(
        self, result: list[dict[str, str]], has_tool_call_tag: bool
    ) -> list[str]:
        """Give the execution checker each call as source text."""
        return convert_to_function_call(result)
