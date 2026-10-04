# Scorer parity with `bfcl-eval`

`metrics.py` ports one rule from the official scorer,
`bfcl_eval.eval_checker.ast_eval.ast_checker.simple_function_checker`. The
package is not a dependency: it pins about 35 others, including
`numpy==1.26.4`, `vllm`, `faiss-cpu`, and every provider SDK. This note
records how the port was checked against it, so the check can be repeated.

## Setup

A scratch environment holds `bfcl-eval` without its dependencies, plus this
project:

```bash
uv venv .superpowers/bfcl
uv pip install --python .superpowers/bfcl/bin/python --no-deps bfcl-eval
uv pip install --python .superpowers/bfcl/bin/python -e .
```

`ast_checker` imports BFCL's model registry, which imports the provider
SDKs. The registry is consulted only to decide whether to rewrite dots in
function names, so the script replaces it with a stub that answers "no":

```python
import collections
import sys
import types

stub = types.ModuleType("bfcl_eval.constants.model_config")
stub.MODEL_CONFIG_MAPPING = collections.defaultdict(
    lambda: types.SimpleNamespace(underscore_to_dot=False)
)
sys.modules["bfcl_eval.constants.model_config"] = stub

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import simple_function_checker
```

## The comparison

For every saved prediction in `results/simple-run*.jsonl` and
`results/multiple-run*.jsonl` that called the gold function:

```python
official = simple_function_checker(
    raw_function,                    # the function's dict from the dataset
    {call.name: call.arguments},
    raw_gold,                        # the row's ground_truth[0]
    Language.PYTHON,
    "parity",
)["valid"]
ours = matches(prediction.example, prediction.call)
```

`raw_function` and `raw_gold` come from `jevtools.data.read_rows`.

## Results

- **Predicted calls.** The two scorers agree on every call in the committed
  test runs that picked the gold function. The count is in the README.
- **Constructed calls.** Before any model output existed, one call per query
  was built for all 600 `simple` and `multiple` queries by reading each gold
  argument out of the query where a run of words parses to an accepted
  value, and otherwise putting in a wrong value of the right type or leaving
  it out. 464 of these are correct and 136 are wrong. The scorers agree on
  all 600.

BFCL also type-checks each value. That check is not ported, because decoding
only produces a value of the parameter's declared type: `coerce` returns an
`int`, `float`, or `str` by type, a flag is a `bool`, and enum and set
values are strings. The predicted-call comparison would show a disagreement
if that were not so.
