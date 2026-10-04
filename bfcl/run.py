"""Run BFCL's official command line with Jev registered as a model.

BFCL finds models in a registry inside its package. This adds Jev's two
methods to that registry and then hands over to BFCL's own `bfcl` command,
so every argument is BFCL's:

    python bfcl/run.py generate --model jev-spec --test-category simple_python
    python bfcl/run.py evaluate --model jev-spec --test-category simple_python
"""

from bfcl_eval.__main__ import cli
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from handler import JevHandler

METHODS = {"words": "Jev (word labelling)", "spec": "Jev (with spec)"}


def main() -> None:
    """Register Jev and run BFCL's command line."""
    for method, display_name in METHODS.items():
        MODEL_CONFIG_MAPPING[f"jev-{method}"] = ModelConfig(
            model_name="jev-latest",
            display_name=display_name,
            url="https://docs.typesafe.ai",
            org="TypeSafe AI",
            license="Proprietary",
            model_handler=JevHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=False,
        )
    cli()


if __name__ == "__main__":
    main()
