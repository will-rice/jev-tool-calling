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
| `simple`      | 400     | 99.1 (99.0–99.2) | 83.2 (82.9–83.5)  | 66.3 (66.0–66.5) | 76.2    |
| `multiple`    | 200     | 99.0 (99.0–99.0) | 85.3 (84.7–85.8)  | 70.2 (69.5–71.0) | 79.5    |
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
- **Jev is right about 95% of the time on every kind of argument it can
  produce, and never on the rest.** Required arguments on queries that
  picked the gold function, first run, `simple` and `multiple` together:

  | Argument                                    | Correct        | Accuracy |
  | ------------------------------------------- | -------------- | -------- |
  | Closed set: enum, boolean, array of an enum | 115 of 121     | 95.0     |
  | Open value, in the request as written       | 1,084 of 1,134 | 95.6     |
  | Open value, needs conversion                | 0 of 160       | 0        |
  | Type that is never asked: dict, tuple, any  | 0 of 14        | 0        |

- **Call accuracy is those two groups multiplied across a call.** One
  argument that needs conversion fails the query: `$1M` for `1000000`,
  `50mH` for `0.05`, `x^2` for `x**2`, a date in another format. About a
  quarter of `simple` queries and a fifth of `multiple` queries have one,
  which is the ceiling. Of the queries within it, 87% come out entirely
  right, and none outside it ever does.
- **The misses within reach are mostly string boundaries**: `human cell` for
  `human`, `company XYZ` for `XYZ`.
- **How `none` is described matters most.** On a dev split, telling the
  model that a word which names or introduces an argument is `none` lifted
  call accuracy from 27.9% to 42.5% with the decoding held fixed. See
  [What we tried](#what-we-tried).
- **A dev gain did not transfer.** A second round of string decoding rules
  lifted the dev split by 2.6 points and the test splits by nothing.
- **The range across runs is not a confidence interval.** It shows how much
  Jev's answers vary between identical requests, which is about a point.

BFCL rewards rewriting a value, which a model that only selects cannot do,
so these scores sit near the bottom of the
[BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) on
`simple` and `multiple` (medians of 92% for both on the Python columns, as
of its 2026-04-12 update) and in the top quarter on `irrelevance` (median
84.6%). The leaderboard runs a later release of the dataset than the one
pinned here.

Input tokens per run of all three splits: 317,876 for the tool question and
about 2,620,000 for the arguments.

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

- A **string** is a run of adjacent words with its label, cut from the query
  as written, with surrounding punctuation removed. Unlabelled words between
  two labelled ones join the run if there are at most two, so the small
  words inside a name stay in it, or at most six that the model gave at
  least 0.3 for the label, so a hyphen it was unsure of does not cut
  `my-bot-id` in two. If several runs carry the label, the one with the most
  probability behind it is used.
- A **number** is the first labelled word that parses as one.
- An **array** has one element per run (strings) or per word that parses
  (numbers).

A value that does not parse is left out. Nothing is converted, so `$1M`,
`50mH`, `11 PM`, and "three" are misses.

The gold function is never used to choose which arguments to ask about, so
picking the wrong function costs the arguments too.

### Dev and test

BFCL has no train split, so the method was not developed blind to the
reported splits. What saw them, and what did not:

- **Designed against the test queries and gold answers, without the model.**
  What counts as a word, which parameter types are asked and how, the rule
  that arrays get no gap filling, the 95-word limit, and the ceiling were
  worked out offline on `simple` and `multiple`. The word pattern was chosen
  because it raised the ceiling.
- **Seen with the model before the reported run.** A probe of 21 test
  queries and a check of the first 20 of each split. The check scored 28% on
  `simple` and showed Jev labelling the word that names an argument.
- **Chosen on a dev split.** The `none` wording, the 0.7 threshold, and the
  number rule were chosen on `live_simple`, a BFCL split this benchmark does
  not report (233 of its 258 queries; the rest have a system message, an
  enum of non-strings, or more than 95 words).
- **Fixed after a first full test run.** A code review of that run found
  that `1,000` was read as `1` and `5-10` as `5` and `-10`. The word pattern
  was corrected and every split run again; the first run scored 66.5, 69.8,
  and 90.4, and the second 66.8, 69.8, and 90.4.
- **A second round on the dev split.** Bridging unsure words and taking the
  strongest run were chosen on dev and are in the reported run, which scores
  66.3, 70.2, and 90.4. A boundary question was also chosen on dev, run on
  the test splits, and then removed because it fixed as many test queries as
  it broke. Removing it used the test result; the scores are the same with
  and without it.

### Scoring

`metrics.py` ports the AST-match rule from `bfcl-eval`
(`simple_function_checker`): the right function, every required parameter
present, every value among the gold answer's accepted values, and strings
compared after removing case, spaces, and light punctuation. The package
itself pins dozens of unrelated dependencies, so it is not a dependency. The
port agrees with it on all 1,783 calls in the committed test runs that
picked the gold function, and on 600 constructed calls of which 136 are
wrong. How to repeat the check is in
[docs/research/2026-10-04-scorer-parity.md](docs/research/2026-10-04-scorer-parity.md).

### Known limitations

- Arguments of type `dict`, `tuple`, or `any`, and arrays of dicts or arrays
  (37 of 3,375 parameters) are never asked.
- A scalar string takes one run of words carrying its label, and a scalar
  number the first such word that parses.
- The ceiling ignores overlap between arguments, so it is an upper bound.
- Parallel, multi-turn, live, and non-Python BFCL splits are not evaluated.

## What we tried

Measured on the dev split, one run each. Its ceiling is 54.9%. Details are
in
[docs/research/2026-10-04-argument-decoding.md](docs/research/2026-10-04-argument-decoding.md).

| Change                                                              | Call accuracy | Argument accuracy |
| ------------------------------------------------------------------- | ------------- | ----------------- |
| First version: plain `none`, adjacent labelled words merged         | 24.5          | 42.4              |
| Numbers read word by word                                           | 27.9          | 49.6              |
| The same, labels below 0.9 probability ignored                      | 41.6          | 61.7              |
| `none` reworded, numbers word by word, no threshold                 | 42.5          | 64.6              |
| `none` reworded, numbers word by word, threshold 0.7 (**adopted**)  | 43.8          | 66.3              |
| Adopted, plus a "which of these numbers" question per number        | 43.8          | 66.8              |
| Adopted, unsure words bridged, strongest run (**current**)          | 45.9          | 67.6              |
| Current, plus a question choosing among each run and its neighbours | 47.2          | 68.8              |

The committed dev run of the current configuration scores 45.9 and 67.3;
re-running a configuration moves call accuracy by about half a point. Only
that run is kept, so the other rows cannot be rescored.

The last row did not survive the test splits: offered a run with one more
word, the model prefers the longer natural phrase (`C sharp major` for
`C sharp`), and it broke as many test queries as it fixed.

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
