"""Load BFCL queries, offered functions, and gold calls."""

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from jevtools.config import DATASET_REPO, DATASET_REVISION
from jevtools.models import Example, Function, Gold, Parameter, Split


def load_examples(split: Split) -> list[Example]:
    """Load a split's queries with their offered functions and gold calls.

    On a split that offers one function per query, the gold function is
    that function, whatever name the gold file gives it.

    Args:
        split: The BFCL category to load.

    Returns:
        The split's examples in file order.
    """
    answers = (
        {}
        if split == "irrelevance"
        else {
            row["id"]: row["ground_truth"][0]
            for row in read_rows(f"possible_answer/BFCL_v3_{split}.json")
        }
    )
    examples = []
    for row in read_rows(f"BFCL_v3_{split}.json"):
        functions = tuple(parse_function(spec) for spec in row["function"])
        gold = None
        if split != "irrelevance":
            [(name, accepted)] = answers[row["id"]].items()
            gold = Gold(
                name=name if len(functions) > 1 else functions[0].name,
                accepted=accepted,
            )
        examples.append(
            Example(
                id=row["id"],
                split=split,
                query=row["question"][0][0]["content"],
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
                enum=schema.get("enum", schema.get("items", {}).get("enum", ())),
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
