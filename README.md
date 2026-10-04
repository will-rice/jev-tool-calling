# Jev on BFCL

Zero-shot evaluation of [Jev](https://docs.typesafe.ai), TypeSafe AI's
"System One" decision model, on tool calling: three single-turn splits of the
[Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
(BFCL) v3.

Jev does not generate text. It takes some state and a set of typed questions
and returns typed answers with probabilities. A tool call needs a function
name and argument values, and most BFCL arguments are open values such as
numbers and names. This project asks how far a model that can only select
gets when each word of the request is labelled with the argument it supplies.

## Results

Model `jev-1.13.0`. Mean of three runs, with the range across runs in
brackets.

| Split         | Queries | Tool accuracy    | Argument accuracy | Call accuracy    | Ceiling |
| ------------- | ------- | ---------------- | ----------------- | ---------------- | ------- |
| `simple`      | 400     | 99.0 (99.0–99.0) | 83.4 (83.1–83.7)  | 66.5 (66.0–66.8) | 76.2    |
| `multiple`    | 200     | 99.0 (99.0–99.0) | 85.4 (85.4–85.6)  | 69.8 (69.5–70.0) | 79.5    |
| `irrelevance` | 240     |                  |                   | 90.4 (90.4–90.4) |         |

- **Call accuracy** is BFCL's AST match, the rule behind the leaderboard's
  AST columns. On `irrelevance` it is the share of queries where Jev called
  nothing.
- **Tool accuracy** is the share of queries that picked the gold function.
  On `simple` only one function is offered, so it is the share where Jev did
  not abstain.
- **Argument accuracy** is over arguments the gold answer requires, on
  queries that picked the gold function. It is counted per argument, so it
  can exceed the ceiling, which is counted per query.
- **Ceiling** is the share of queries whose every required argument appears
  in the request as written. The rest need a conversion Jev is not asked to
  make.

What the numbers say:

- **Picking the function is nearly solved.** Jev picks the gold function on
  99% of queries, whether one function is offered or up to four, and calls
  nothing on 90% of the queries no offered function fits.
- **Where the value is in the request, labelling words finds it.** Of the
  queries within reach, 87% come out entirely right (86.6% on `simple`,
  87.4% on `multiple`, first run). No query outside the ceiling is ever
  right.
- **The ceiling is the larger loss.** About a quarter of `simple` queries and
  a fifth of `multiple` queries need a value that is not in the request as
  written: `$1M` for `1000000`, `50mH` for `0.05`, `x^2` for `x**2`, a date
  in another format, or a dict. That is 112 of the 162 required arguments
  missed on `simple` (first run, queries that picked the gold function).
- **The remaining misses are mostly string boundaries.** 29 of the 50 other
  misses on `simple` are strings cut one word too long or too short.
- **How `none` is described matters most.** On a dev split, telling the
  model that a word which names or introduces an argument is `none` lifted
  call accuracy from 24.5% to 42.5%. See [What we tried](#what-we-tried).

Input tokens per run of all three splits: 317,876 for the tool question and
2,626,216 for the arguments.

## Method

Each query gets two requests, both with the query as state.

1. **Tool.** One `Choice` over the offered functions, each with its
   description, plus a described `none`. Choosing `none` is abstaining.
2. **Arguments**, for the function Jev picked:
   - **Open values** (strings, numbers, and arrays of them): one `Choice` per
     word of the query over those parameters plus `none`. The instructions
     are labelled fields: the function, the word, the words before it, the
     words after it, and the question "Which argument of `function` does
     `word` supply in `query`?". The `none` option says that a word which
     only names or introduces an argument is `none`.
   - **Enums**: one `Choice` over the enum's values.
   - **Booleans**: one `Noul`.
   - **Arrays of an enum**: one `Noul` per member.
   - **Optional enums and booleans**: one more `Noul` asking whether the
     query says anything about the argument. If not, it is left out and the
     function's default applies.

Words are runs of letters, numbers, or single punctuation marks, so `A(3,4)`
is six words and a value is never glued to the bracket beside it.

A word's label counts when its probability is at least 0.7. Then:

- A **string** is the first run of adjacent words with its label, cut from
  the query as written, with surrounding punctuation removed. Up to two
  unlabelled words between two labelled ones join the run, so the small
  words inside a name stay in it.
- A **number** is the first labelled word that parses as one.
- An **array** has one element per run (strings) or per word that parses
  (numbers).

A value that does not parse is left out. Nothing is converted, so `$1M`,
`50mH`, `11 PM`, and "three" are misses.

The gold function is never used to choose which arguments to ask about, so
picking the wrong function costs the arguments too.

### Dev and test

BFCL has no train split. The first version scored 28% on a 20-query check,
and fixing it on the reported splits would have tuned the method on its own
test set. The `none` wording, the 0.7 threshold, and the number rule were
chosen on `live_simple`, a BFCL split this benchmark does not report (233 of
its 258 queries; the rest have a system message, an enum of non-strings, or
more than 95 words). The test splits were then run once with that
configuration. The 20 queries per test split seen in the earlier check are
the only test data looked at before the reported run.

### Scoring

`metrics.py` ports the AST-match rule from `bfcl-eval`
(`simple_function_checker`): the right function, every required parameter
present, every value among the gold answer's accepted values, and strings
compared after removing case, spaces, and light punctuation. The package
itself pins dozens of unrelated dependencies, so it is not a dependency. The
port agrees with it on all 1,902 predicted calls that picked the gold
function, and on 600 constructed calls of which 136 are wrong.

### Known limitations

- Arguments of type `dict`, `tuple`, or `any`, and arrays of dicts or arrays
  (37 of 3,375 parameters) are never asked.
- A scalar argument takes the first run or word carrying its label.
- The ceiling ignores overlap between arguments, so it is an upper bound.
- Parallel, multi-turn, live, and non-Python BFCL splits are not evaluated.

## What we tried

Measured on the dev split, one run each. Its ceiling is 54.9%. Details are
in
[docs/research/2026-10-04-argument-decoding.md](docs/research/2026-10-04-argument-decoding.md).

| Change                                                             | Call accuracy | Argument accuracy |
| ------------------------------------------------------------------ | ------------- | ----------------- |
| First version: plain `none`, adjacent labelled words merged        | 24.5          | 42.4              |
| Numbers read word by word                                          | 27.9          | 49.6              |
| The same, labels below 0.9 probability ignored                     | 41.6          | 61.7              |
| `none` reworded, numbers word by word, no threshold                | 42.5          | 64.6              |
| `none` reworded, numbers word by word, threshold 0.7 (**adopted**) | 43.8          | 66.3              |
| Adopted, plus a "which of these numbers" question per number       | 43.8          | 66.8              |

The committed dev run of the adopted configuration scores 43.3 and 65.9;
re-running a configuration moves call accuracy by about half a point.

## Setup

```bash
uv sync
```

```bash
cp .env.example .env
```

Add your `TYPESAFE_API_KEY` to `.env`.

## Usage

Run the three reported splits three times:

```bash
uv run run test
```

Run the dev split once:

```bash
uv run run dev
```

Score saved runs again without calling the API:

```bash
uv run report test
```

Check the pipeline on the first 20 queries of each split:

```bash
uv run run test --limit 20
```

A limited run writes to its own files and leaves the full results in place.

## Output

Each run is committed as `results/{split}-run{n}.jsonl`, one record per
query: the query, the offered functions, the gold answer, the predicted call
(or `null` if Jev abstained), the tool question's probabilities, every
argument question's probabilities, and the input tokens of each request.
Because every word's option probabilities are kept, a decoding rule can be
changed and rescored without asking the model again.

Records are `jevtools.models.Prediction` objects:

```python
from pathlib import Path

from jevtools.models import Prediction

lines = Path("results/simple-run1.jsonl").read_text().splitlines()
predictions = [Prediction.model_validate_json(line) for line in lines]
```

## Project structure

```
src/jevtools/
├── config.py   # Constants: dataset, model, splits, word pattern, thresholds
├── models.py   # Parameter, Function, Example, Call, Prediction
├── data.py     # Load queries, functions, and gold calls
├── jev.py      # Build questions, call Jev, decode answers into a call
├── metrics.py  # AST match, accuracies, ceiling
└── scripts/
    ├── run.py     # Run a stage's splits
    └── report.py  # Score the saved runs
```

## Development

```bash
uv run pre-commit run -a
```

This formats, lints, type-checks, and runs the tests. The data tests download
the dataset from the Hugging Face Hub.

## Data

[`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard),
pinned to a fixed revision in `config.py`. See the dataset card for how to
cite BFCL.

## License

See [LICENSE](LICENSE).
