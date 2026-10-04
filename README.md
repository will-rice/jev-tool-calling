# Jev on BFCL

Zero-shot evaluation of [Jev](https://docs.typesafe.ai), TypeSafe AI's
"System One" decision model, on tool calling: three single-turn splits of the
[Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
(BFCL) v3.

Jev does not generate text. It takes some state and a set of typed questions
and returns typed answers with probabilities. A tool call needs a function
name and argument values, and most BFCL arguments are open values such as
numbers, names, and dates. This project asks how far a model that can only
select gets, first by labelling each word of the request with the argument
it supplies, then with a per-function spec that lets code find and assemble
the values the model chooses among.

## Results

Model `jev-1.13.0`. Mean of three runs, with the range across runs in
brackets. Call accuracy is BFCL's AST match.

| Split         | Queries | Word labelling   | With spec        |
| ------------- | ------- | ---------------- | ---------------- |
| `simple`      | 400     | 66.2 (66.0–66.5) | 74.9 (74.5–75.5) |
| `multiple`    | 200     | 70.0 (69.0–71.0) | 76.8 (76.5–77.5) |
| `irrelevance` | 240     | 90.4 (90.4–90.4) | 90.4 (90.4–90.4) |

The two methods differ only in how arguments are asked for. Picking the
function is the same request in both.

| Metric                           | Split      | Word labelling | With spec    |
| -------------------------------- | ---------- | -------------- | ------------ |
| Tool accuracy                    | `simple`   | 99.0           | 99.1         |
|                                  | `multiple` | 98.8           | 99.0         |
| Argument accuracy                | `simple`   | 83.1           | 89.2         |
|                                  | `multiple` | 85.3           | 90.0         |
| Input tokens per run, all splits |            | 2.93 million   | 2.44 million |

- **Call accuracy**: the whole call is right. On `irrelevance` it is the
  share of queries where Jev called nothing.
- **Tool accuracy**: the share of queries that picked the gold function. On
  `simple` only one function is offered, so it is the share where Jev did
  not abstain.
- **Argument accuracy**: over arguments the gold answer requires, on queries
  that picked the gold function.

What the numbers say:

- **Picking the function is nearly solved.** Jev picks the gold function on
  99% of queries, whether one function is offered or up to four, and calls
  nothing on 90% of the queries no offered function fits.
- **Jev is about 95% right on any value it can select, and the spec widens
  what it can select.** Required arguments on queries that picked the gold
  function, first run, `simple` and `multiple` together:

  | Argument                                    | Count | Word labelling | With spec |
  | ------------------------------------------- | ----- | -------------- | --------- |
  | Closed set: enum, boolean, array of an enum | 121   | 95.0           | 92.6      |
  | Open value, in the request as written       | 1,134 | 95.5           | 96.0      |
  | Open value, needs conversion                | 160   | 0              | 47.5      |
  | Type that is never asked: dict, tuple, any  | 14    | 0              | 0         |

- **Word labelling has a ceiling.** It can only copy. 76.2% of `simple`
  queries and 79.5% of `multiple` queries have every required value in the
  request as written; it gets 87% of those entirely right and none of the
  others.
- **The spec gets past it.** Of the 136 queries outside that ceiling, the
  spec method gets 47 right (34.6%), by reading `$1M` as `1000000`, writing
  "March 5, 2023" as `2023-03-05`, or choosing `vaporization` for "liquid to
  steam". By the kind the spec gave each argument: dates go from 36% to
  100%, numbers from 84% to 92%, options from 82% to 96%, places from 91%
  to 98%.
- **It is also cheaper.** A number or date no longer needs a question per
  word, so input tokens fall by 17%.
- **What is left needs reasoning.** Unit conversion (`50mH` to `0.05`),
  arithmetic (`2 pi` to `6.2832`), inference ("from rest" to `0`), outside
  knowledge (`Apple` to `AAPL`), and structured values.
- **The range across runs is not a confidence interval.** It shows how much
  Jev's answers vary between identical requests, which is about a point.

BFCL rewards rewriting a value, which a model that only selects cannot do
alone. On the
[BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)'s
Python columns (109 models, medians of 92% on both, as of its 2026-04-12
update) the spec method would rank about 101st of 110 on `simple` and 96th
on `multiple`, and 23rd on `irrelevance` (median 84.6%). The leaderboard
runs a later release of the dataset than the one pinned here.

## Method

Each query gets two requests, both with the query as state.

1. **Tool.** One `Choice` over the offered functions, each with its
   description, plus a described `none`. Choosing `none` is abstaining.
2. **Arguments**, for the function Jev picked, asked the way that function's
   spec says. The gold function is never used, so picking the wrong function
   costs the arguments too.

### The spec

A spec has one entry per parameter, following TypeSafe's
[function calling](https://docs.typesafe.ai/cookbooks/function_calling.md),
[pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md),
and [date extraction](https://docs.typesafe.ai/cookbooks/date_extraction_cookbook.md)
cookbooks. Each entry has a plain-language question, a question asking
whether an optional argument is stated at all, and a kind:

| Kind    | What Jev is asked                                        | What code does                                       |
| ------- | -------------------------------------------------------- | ---------------------------------------------------- |
| text    | For each word of the query, which argument it supplies   | Cuts the labelled words from the query               |
| number  | Which of the numbers found in the query is this argument | Finds the numbers, reads `$1M`, `40%`, `three`       |
| options | Which listed value, each with a line describing it       | Passes the value; an open list can fall back to text |
| flag    | Yes or no                                                |                                                      |
| set     | Yes or no, per member                                    | Collects the members                                 |
| date    | Which month, which day, which year                       | Formats the date as the function wants               |
| place   | Which words are the city; which US state; which country  | Joins them in the function's format                  |
| skip    | Nothing                                                  | Leaves the argument out                              |

**Word labelling** is the method with no authored spec. The spec is derived
from the schema: strings, numbers, and arrays of them are text; enums are
options with no lines; booleans are flags. So it only ever copies words.

**With spec** uses a spec written by a language model from the function
definitions alone (`specs/dev.json`, `specs/test.json`). It was given
[`specs/FORMAT.md`](specs/FORMAT.md) and the schemas in batches and told to
read nothing else: no queries, no gold answers, no results. Code then
normalises it: a kind the parameter's type cannot take falls back to the
schema's, a list of options with no enum behind it is made open, and a set
with no enum behind it becomes text.

### Labelling words

Words are runs of letters, numbers, or single punctuation marks, so `A(3,4)`
is six words and a value is never glued to the bracket beside it. Each word
question's instructions are labelled fields: the function, the word, the
words before it, the words after it, and the question "Which argument of
`function` does `word` supply in `query`?". The `none` option says that a
word which only names or introduces an argument is `none`.

A word's label counts when its probability is at least 0.7. Then:

- A **string** is a run of adjacent words with its label, cut from the query
  as written, with surrounding punctuation removed. Unlabelled words between
  two labelled ones join the run if there are at most two, so the small
  words inside a name stay in it, or at most six that the model gave at
  least 0.3 for the label, so a hyphen it was unsure of does not cut
  `my-bot-id` in two. If several runs carry the label, the one with the most
  probability behind it is used.
- A **number** copied this way is the first labelled word that parses as one.
- An **array** has one element per run (strings) or per word that parses
  (numbers).

A value that does not parse is left out. Word labelling converts nothing.

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
- **The spec method.** The idea came from classifying the test splits'
  out-of-reach arguments by what producing them would take, and the kinds a
  spec can have were chosen from that classification. The spec files were
  written without sight of any query or answer. The runtime was developed
  on the dev split, where it scored 55.8 in a pilot, and the test splits
  were then run once with it.

### Scoring

`metrics.py` ports the AST-match rule from `bfcl-eval`
(`simple_function_checker`): the right function, every required parameter
present, every value among the gold answer's accepted values, and strings
compared after removing case, spaces, and light punctuation. The package
itself pins dozens of unrelated dependencies, so it is not a dependency. The
port agrees with it on all 3,564 calls in the committed test runs, under
both methods, that picked the gold function, and on 600 constructed calls of which 136 are
wrong. How to repeat the check is in
[docs/research/2026-10-04-scorer-parity.md](docs/research/2026-10-04-scorer-parity.md).

### Known limitations

- Arguments of type `dict`, `tuple`, or `any`, and arrays of dicts or arrays
  (37 of 3,375 parameters) are never asked.
- The spec is only as good as its author's reading of each description. A
  parameter wrongly made a closed set loses values outside it, which is why
  closed-set accuracy is slightly lower with the spec (92.6 against 95.0).
- Number candidates cover common written forms only. Units, arithmetic, and
  inferred values are not attempted.
- A place is completed with a US state or one of about 120 countries.
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
| Current, each word shown the argument values assembled so far       | 45.9          | 67.6              |
| Current, each word shown every earlier word's label                 | 42.9          | 64.4              |
| Current, with the function's description in the state               | 46.4          | 68.0              |
| **A spec per function** (pilot)                                     | 55.8          | 76.5              |

"Current" is word labelling as committed. The committed dev runs score 45.1
and 67.1 for word labelling and 57.1 and 77.2 with the spec; re-running a
configuration moves call accuracy by about a point. Only those runs are
kept, so the other rows cannot be rescored.

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

Run the three reported splits three times with the authored spec:

```bash
uv run run test spec
```

Run them with word labelling only:

```bash
uv run run test words
```

Run the dev split once:

```bash
uv run run dev spec
```

Score saved runs again without calling the API:

```bash
uv run report test spec
```

Check the pipeline on the first 20 queries of each split:

```bash
uv run run test spec --limit 20
```

A limited run writes to its own files and leaves the full results in place.

## Output

Each run is committed as `results/{split}-{method}-run{n}.jsonl`, one record
per query: the query, the offered functions, the gold answer, the predicted call
(or `null` if Jev abstained), the tool question's probabilities, every
argument question's probabilities, and the input tokens of each request.
Because every word's option probabilities are kept, a decoding rule can be
changed and rescored without asking the model again.

Records are `jevtools.models.Prediction` objects:

```python
from pathlib import Path

from jevtools.models import Prediction

lines = Path("results/simple-spec-run1.jsonl").read_text().splitlines()
predictions = [Prediction.model_validate_json(line) for line in lines]
```

## Project structure

```
src/jevtools/
├── config.py   # Constants: dataset, model, splits, word pattern, thresholds
├── models.py   # Parameter, Function, Example, Call, Prediction
├── data.py     # Load queries, functions, and gold calls
├── spec.py     # Spec entries, the spec derived from a schema, loading
├── values.py   # Find numbers; assemble dates and places; read words
├── jev.py      # Build questions, call Jev, decode answers into a call
├── metrics.py  # AST match, accuracies, ceiling
└── scripts/
    ├── run.py     # Run a stage's splits
    └── report.py  # Score the saved runs
```

`specs/` holds the authored specs and the format they were written to.

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
