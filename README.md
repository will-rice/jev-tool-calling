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
| `simple`      | 400     | 99.0 (99.0–99.0) | 83.5 (83.4–83.5)  | 66.8 (66.8–66.8) | 76.2    |
| `multiple`    | 200     | 99.0 (99.0–99.0) | 85.1 (84.9–85.4)  | 69.8 (69.0–70.5) | 79.5    |
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
  queries within reach, 88% come out entirely right (87.5% on `simple`,
  88.1% on `multiple`, first run). No query outside the ceiling is ever
  right.
- **The ceiling is the larger loss.** About a quarter of `simple` queries and
  a fifth of `multiple` queries need a value that is not in the request as
  written: `$1M` for `1000000`, `50mH` for `0.05`, `x^2` for `x**2`, or a date
  in another format. On `simple` (first run, queries that picked the gold
  function) 158 required arguments are missed: 113 are not in the request
  as written and 10 are of a type that is never asked, such as a dict.
- **The remaining misses are mostly strings.** 24 of the other 35 are
  strings decoded to text the gold answer does not accept.
- **How `none` is described matters most.** On a dev split, telling the
  model that a word which names or introduces an argument is `none` lifted
  call accuracy from 27.9% to 42.5% with the decoding held fixed. See
  [What we tried](#what-we-tried).
- **The range across runs is not a confidence interval.** It shows how much
  Jev's answers vary between identical requests, which is under a point.

Input tokens per run of all three splits: 317,876 for the tool question and
2,618,254 for the arguments.

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
  and 90.4, and the reported one 66.8, 69.8, and 90.4.

### Scoring

`metrics.py` ports the AST-match rule from `bfcl-eval`
(`simple_function_checker`): the right function, every required parameter
present, every value among the gold answer's accepted values, and strings
compared after removing case, spaces, and light punctuation. The package
itself pins dozens of unrelated dependencies, so it is not a dependency. The
port agrees with it on all 1,782 calls in the committed test runs that
picked the gold function, and on 600 constructed calls of which 136 are
wrong. How to repeat the check is in
[docs/research/2026-10-04-scorer-parity.md](docs/research/2026-10-04-scorer-parity.md).

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

The committed dev run of the adopted configuration scores 43.3 and 66.1;
re-running a configuration moves call accuracy by about half a point. Only
that run is kept, so the other rows cannot be rescored.

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
