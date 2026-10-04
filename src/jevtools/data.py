"""Load BFCL queries, offered functions, and gold calls."""

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from jevtools.config import DATASET_REPO, DATASET_REVISION, UNANSWERED
from jevtools.models import Example, Function, Gold, Parameter, Split


def load_examples(split: Split) -> list[Example]:
    """Load a split's queries with their offered functions and gold calls.

    The query is the last message of a row and any earlier messages are its
    context. On a split that offers one function per query, the gold
    function is that function, whatever name the gold file gives it. Splits
    scored only on whether a call was made have no gold calls, and parallel
    splits have several per query.

    Gold answers are paired with rows by position: the two files are in the
    same order, and one live row's id is misspelled in its gold file.

    Args:
        split: The BFCL category to load.

    Returns:
        Every row of the split, in file order.
    """
    rows = read_rows(f"BFCL_v3_{split}.json")
    answers = (
        [None] * len(rows)
        if split in UNANSWERED
        else read_rows(f"possible_answer/BFCL_v3_{split}.json")
    )
    examples = []
    for row, answer in zip(rows, answers, strict=True):
        functions = tuple(parse_function(spec) for spec in row["function"])
        [messages] = row["question"]
        gold = ()
        if answer is not None:
            gold = tuple(
                Gold(
                    name=name if len(functions) > 1 else functions[0].name,
                    accepted=accepted,
                )
                for call in answer["ground_truth"]
                for name, accepted in call.items()
            )
        examples.append(
            Example(
                id=row["id"],
                split=split,
                query=messages[-1]["content"],
                context=tuple(
                    (message["role"], message["content"]) for message in messages[:-1]
                ),
                functions=functions,
                gold=gold,
            )
        )
    return examples


def parse_function(spec: dict[str, Any]) -> Function:
    """Parse one offered function and its parameters."""
    required = spec["parameters"]["required"]
    return Function(
        name=spec["name"],
        description=spec["description"],
        parameters=tuple(
            Parameter(
                name=name,
                type=schema["type"],
                description=schema.get("description"),
                required=name in required,
                enum=tuple(
                    str(value)
                    for value in schema.get(
                        "enum", schema.get("items", {}).get("enum", ())
                    )
                ),
                item_type=schema.get("items", {}).get("type"),
            )
            for name, schema in spec["parameters"]["properties"].items()
        ),
    )


def read_rows(filename: str) -> list[dict[str, Any]]:
    """Download a JSON-lines file from the pinned revision and parse its rows."""
    path = hf_hub_download(
        DATASET_REPO, filename, repo_type="dataset", revision=DATASET_REVISION
    )
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
