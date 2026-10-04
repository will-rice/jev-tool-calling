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
brackets. Call accuracy is BFCL's AST match. Every row of every split is
scored; none is left out.

| Split              | Queries | Word labelling   | With spec        |
| ------------------ | ------- | ---------------- | ---------------- |
| `simple`           | 400     | 66.3 (66.0–66.8) | 74.6 (74.2–74.8) |
| `multiple`         | 200     | 70.0 (69.5–70.5) | 76.2 (76.0–76.5) |
| `irrelevance`      | 240     | 90.4 (90.4–90.4) | 90.4 (90.4–90.4) |
| `live_multiple`    | 1,053   | 46.9 (46.8–46.9) | 66.3 (66.1–66.6) |
| `live_irrelevance` | 882     | 68.9 (68.7–69.2) | 68.7 (68.6–69.0) |
| `live_relevance`   | 18      | 100.0 (100–100)  | 98.1 (94.4–100)  |

The two methods differ only in how arguments are asked for. Picking the
function is the same request in both.

| Metric            | Split           | Word labelling | With spec |
| ----------------- | --------------- | -------------- | --------- |
| Tool accuracy     | `simple`        | 99.1           | 99.0      |
|                   | `multiple`      | 98.8           | 99.0      |
|                   | `live_multiple` | 95.4           | 95.4      |
| Argument accuracy | `simple`        | 83.2           | 89.1      |
|                   | `multiple`      | 85.2           | 89.5      |
|                   | `live_multiple` | 69.8           | 84.9      |

Input tokens per run of all six splits: 12.7 million with word labelling and
13.2 million with the spec.

- **Call accuracy**: the whole call is right. On the irrelevance splits it
  is the share of queries where Jev called nothing, and on `live_relevance`
  the share where it called something.
- **Tool accuracy**: the share of queries that picked the gold function. On
  `simple` only one function is offered, so it is the share where Jev did
  not abstain.
- **Argument accuracy**: over arguments the gold answer requires, on queries
  that picked the gold function.

### Which of BFCL v3 this covers

| Categories                                               | Queries | Status                                            |
| -------------------------------------------------------- | ------- | ------------------------------------------------- |
| `simple`, `multiple`, `irrelevance`                      | 840     | Reported above                                    |
| `live_multiple`, `live_irrelevance`, `live_relevance`    | 1,953   | Reported above                                    |
| `live_simple`                                            | 258     | Dev split: the method was tuned on it (see below) |
| `parallel`, `parallel_multiple`, and their live versions | 440     | Not yet run: the method emits one call per query  |
| `java`, `javascript`                                     | 150     | Not yet run: values are read and scored as Python |
| `sql`, `rest`, `exec_*`                                  | 410     | Not yet run: scored by executing the call         |
| Multi-turn (5 categories)                                | 1,000   | Not yet run: needs state across turns             |

A category that is not yet run has no score here. It is not counted as zero
and not folded into any average.

What the numbers say:

- **Picking the function is nearly solved.** Jev picks the gold function on
  99% of queries, whether one function is offered or up to four, and calls
  nothing on 90% of the queries no offered function fits.
- **Jev is about 95% right on any value it can select, and the spec widens
  what it can select.** Required arguments on queries that picked the gold
  function, first run, `simple` and `multiple` together:

  | Argument                                    | Count | Word labelling | With spec |
  | ------------------------------------------- | ----- | -------------- | --------- |
  | Closed set: enum, boolean, array of an enum | 121   | 95.0           | 93.4      |
  | Open value, in the request as written       | 1,134 | 95.5           | 95.9      |
  | Open value, needs conversion                | 160   | 0              | 48.1      |
  | Type that is never asked: dict, tuple, any  | 14    | 0              | 0         |

- **Word labelling has a ceiling.** It can only copy. 76.2% of `simple`
  queries and 79.5% of `multiple` queries have every required value in the
  request as written; it gets 87% of those entirely right and none of the
  others.
- **The spec gets past it.** Of the 136 queries outside that ceiling, the
  spec method gets 48 right (35.3%), by reading `$1M` as `1000000`, writing
  "March 5, 2023" as `2023-03-05`, or choosing `vaporization` for "liquid to
  steam". By the kind the spec gave each argument: dates go from 36% to
  100%, numbers from 84% to 93%, options from 85% to 96%, places from 91%
  to 98%.
- **On `simple` and `multiple` it is also cheaper.** A number or date no
  longer needs a question per word, so those two splits use 17% fewer input
  tokens. Over all six splits the spec uses 4% more, because the live
  functions have many options, each with a described line.
- **The spec matters most on live data.** `live_multiple` goes from 46.9% to
  66.3%, past its word-labelling ceiling of 57.4%: its functions take many
  dates, places, and listed values.
- **Abstaining is harder on live data.** Jev calls nothing on 90% of
  `irrelevance` queries but only 69% of `live_irrelevance` ones, where the
  offered functions are closer to the request.
- **What is left needs reasoning.** Unit conversion (`50mH` to `0.05`),
  arithmetic (`2 pi` to `6.2832`), inference ("from rest" to `0`), outside
  knowledge (`Apple` to `AAPL`), and structured values.
- **The range across runs is not a confidence interval.** It shows how much
  Jev's answers vary between identical requests, which is about a point.

BFCL rewards rewriting a value, which a model that only selects cannot do
alone. Against the
[BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)'s
matching Python columns (109 models, as of its 2026-04-12 update), the spec
method would rank about:

| Split              | With spec | Leaderboard median | Rank of 110 |
| ------------------ | --------- | ------------------ | ----------- |
| `simple`           | 74.6      | 92.0               | 101st       |
| `multiple`         | 76.2      | 92.0               | 96th        |
| `irrelevance`      | 90.4      | 84.6               | 23rd        |
| `live_multiple`    | 66.3      | 71.0               | 73rd        |
| `live_irrelevance` | 68.7      | 77.3               | 78th        |
| `live_relevance`   | 98.1      | 81.3               | 8th         |

The leaderboard runs a later release of the dataset than the one pinned
here, so these ranks are approximate.

## Method

Each query gets two requests, both with the query as state.

1. **Tool.** One `Choice` over the offered functions, each with its
   description, plus a described `none`. Choosing `none` is abstaining. A
   query offered no function at all is answered with no call.
2. **Arguments**, for the function Jev picked, asked the way that function's
   spec says. The gold function is never used, so picking the wrong function
   costs the arguments too.

The query is a row's last message. Earlier messages, such as a system prompt
or a previous turn, are sent with it as context and are not labelled. A long
query's questions are spread over several requests, and each word question
shows 95 words on either side of its word.

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
  not report. At the time only 233 of its 258 queries could be posed; the
  rest had a system message, an enum of non-strings, or more than 95 words.
  All 258 are posed now.
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
- **Every row.** The loader used to leave out rows it could not pose. That
  would flatter a reported result, so it now poses every row: earlier
  messages as context, number enums as options, long queries in several
  requests. The first three splits have no such rows and were unaffected.
- **The live splits.** `live_multiple`, `live_irrelevance`, and
  `live_relevance` were run once, after everything above, with no tuning on
  them. Their specs were written from function definitions only.
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
port agrees with it on all 9,591 calls in the committed test runs, under
both methods, that picked the gold function, and on 600 constructed calls of which 136 are
wrong. How to repeat the check is in
[docs/research/2026-10-04-scorer-parity.md](docs/research/2026-10-04-scorer-parity.md).

### Known limitations

- Arguments of type `dict`, `tuple`, or `any`, and arrays of dicts or arrays
  (37 of 3,375 parameters) are never asked.
- The spec is only as good as its author's reading of each description. A
  parameter wrongly made a closed set loses values outside it, which is why
  closed-set accuracy is slightly lower with the spec (93.4 against 95.0).
- Number candidates cover common written forms only. Units, arithmetic, and
  inferred values are not attempted.
- A place is completed with a US state or one of about 120 countries.
- A scalar string takes one run of words carrying its label, and a scalar
  number the first such word that parses.
- The ceiling ignores overlap between arguments, so it is an upper bound.
- Parallel, multi-turn, executed, and non-Python BFCL splits are not yet
  evaluated; see the table under Results.
- `live_relevance` has 18 queries, so one query is 5.6 points.

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

"Current" is word labelling as committed at the time, on the 233 dev queries
that could then be posed. The committed dev runs cover all 258 and score
43.8 and 64.3 for word labelling and 54.3 and 74.1 with the spec; re-running a
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
