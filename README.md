# Jev on BFCL

Zero-shot evaluation of [Jev](https://docs.typesafe.ai), TypeSafe AI's
"System One" decision model, on the
[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
(BFCL), run through BFCL's official runner.

Jev does not generate text. It takes some state and a set of typed questions
and returns typed answers with probabilities. A tool call needs a function
name and argument values, and most BFCL arguments are open values such as
numbers, names, and dates. This project asks how far a model that can only
select gets: by labelling each word of the request with the argument it
supplies, and then with a per-function spec that lets code find and
assemble the values the model chooses among.

## Results

These are BFCL's own numbers: generated and scored by `bfcl-eval` on BFCL v4,
with Jev plugged in as a model. The tables BFCL writes are in
[`bfcl/score`](bfcl/score) and its raw results in [`bfcl/result`](bfcl/result).
One run, `jev-1.13.0`.

**Non-live**

| Category              | Queries | Word labelling | With spec |
| --------------------- | ------- | -------------- | --------- |
| Python simple         | 400     | 67.3           | 75.3      |
| Java simple           | 100     | 38.0           | 39.0      |
| JavaScript simple     | 50      | 48.0           | 46.0      |
| Multiple              | 200     | 70.0           | 76.0      |
| Parallel              | 200     | 45.5           | 43.5      |
| Parallel multiple     | 200     | 37.0           | 41.0      |
| Irrelevance detection | 240     | 90.4           | 90.0      |
| **Non-live overall**  |         | **50.9**       | **53.5**  |

**Live**

| Category              | Queries | Word labelling | With spec |
| --------------------- | ------- | -------------- | --------- |
| Simple                | 258     | 43.0           | 55.4      |
| Multiple              | 1,053   | 47.2           | 66.1      |
| Parallel              | 16      | 6.3            | 31.3      |
| Parallel multiple     | 24      | 33.3           | 33.3      |
| Irrelevance detection | 884     | 72.9           | 73.3      |
| Relevance detection   | 16      | 93.8           | 93.8      |
| **Live overall**      |         | **45.7**       | **63.1**  |

**BFCL's overall score: 19.8 with the spec, 17.8 with word labelling.** It
averages in the categories below, which are zero.

### What is not run

| Categories                 | Why                                                                                |
| -------------------------- | ---------------------------------------------------------------------------------- |
| Multi-turn (4)             | Not built yet. It is scored on function calls, so Jev can attempt it               |
| Web search (2), memory (3) | Scored on the right answer appearing in the model's text reply. Jev writes no text |
| Format sensitivity         | BFCL runs it only for prompt-mode models                                           |

BFCL counts these as zero in its overall score, and so do the numbers above.

### Notes on these numbers

- **Word labelling** is Jev with no authored spec: arguments are asked from
  each function's schema, so values are only ever copied from the request.
  **With spec** adds a spec per function, written by a language model from
  the function definitions alone.
- **Java and JavaScript have no spec yet**, so both columns use the schema
  there. BFCL passes their arguments as strings.
- **Live parallel has 16 queries and live relevance 16**, so one query is
  about 6 points.
- **`live_simple` was the dev split**: the argument questions and decoding
  were tuned on it, so its row is not a clean result.
- **Cost and speed**, as BFCL measured them: under a dollar for the whole
  run, and a 95th-percentile latency of 0.67 seconds per query. The mean is
  skewed by a few requests that hit long retries.

### Against the leaderboard

BFCL's leaderboard had 109 models at its 2026-04-12 update. Where the spec
method would rank on each column:

| Column                     | Jev  | Leaderboard median | Rank of 110 |
| -------------------------- | ---- | ------------------ | ----------- |
| Overall                    | 19.8 | 35.5               | 101st       |
| Non-live overall           | 53.5 | 83.0               | 100th       |
| Non-live Python simple     | 75.3 | 92.0               | 100th       |
| Non-live multiple          | 76.0 | 92.0               | 96th        |
| Non-live parallel          | 43.5 | 88.0               | 102nd       |
| Non-live parallel multiple | 41.0 | 82.5               | 98th        |
| Non-live irrelevance       | 90.0 | 84.6               | 23rd        |
| Live overall               | 63.1 | 70.8               | 81st        |
| Live multiple              | 66.1 | 71.0               | 75th        |
| Live irrelevance           | 73.3 | 77.3               | 68th        |
| Live relevance             | 93.8 | 81.2               | 8th         |

### What the numbers say

- **Deciding is Jev's strength.** It picks the right function about 99% of
  the time when one call is needed, declines to call on 90% of the non-live
  queries no function fits, and calls something on 94% of those one does.
- **Jev is about 95% right on any value it can select, and the spec widens
  what it can select.** Required arguments on `simple` and `multiple`
  queries that made one call to the gold function, from this project's own
  runner:

  | Argument                                    | Count | Word labelling | With spec |
  | ------------------------------------------- | ----- | -------------- | --------- |
  | Closed set: enum, boolean, array of an enum | 120   | 95.0           | 94.2      |
  | Open value, in the request as written       | 1,139 | 95.6           | 96.2      |
  | Open value, needs conversion                | 159   | 0              | 48.4      |
  | Type that is never asked: dict, tuple, any  | 14    | 0              | 0         |

- **Word labelling has a ceiling.** It can only copy, and about a quarter of
  `simple` queries need a value that is not in the request as written:
  `$1M` for `1000000`, "March 5, 2023" as `2023-03-05`, "liquid to steam"
  for `vaporization`.
- **The spec gets past it, most of all on live data**, whose functions take
  many dates, places, and listed values: 45.7 to 63.1 overall.
- **Several calls from one request work about 40% of the time**, with no
  tuning on those categories.
- **What is left needs reasoning**: unit conversion (`50mH` to `0.05`),
  arithmetic (`2 pi` to `6.2832`), inference ("from rest" to `0`), outside
  knowledge (`Apple` to `AAPL`), and structured values.
- **Against language models Jev is near the bottom on filling a call and
  mid-table or better on deciding whether to make one.** BFCL rewards
  rewriting a value, which a model that only selects cannot do alone.

## Running the benchmark

BFCL's runner pins dependencies that conflict with this project's, so it
runs in its own environment. From the repository root:

```bash
uv venv .bfcl --python 3.13
```

```bash
printf 'numpy>=2.1\nfilelock>=3.20\n' > .bfcl/overrides.txt
```

```bash
uv pip install --python .bfcl/bin/python --override .bfcl/overrides.txt bfcl-eval soundfile
```

```bash
uv pip install --python .bfcl/bin/python --no-deps -e .
```

```bash
uv pip install --python .bfcl/bin/python typesafe-sdk tqdm python-dotenv
```

Add your `TYPESAFE_API_KEY` to `.env`. Then generate and score with BFCL's
own commands; [`bfcl/run.py`](bfcl/run.py) registers `jev-words` and
`jev-spec` as models and hands over to BFCL's `bfcl` command line:

```bash
BFCL_PROJECT_ROOT=$PWD/bfcl .bfcl/bin/python bfcl/run.py generate --model jev-spec --test-category single_turn --num-threads 8
```

```bash
BFCL_PROJECT_ROOT=$PWD/bfcl .bfcl/bin/python bfcl/run.py evaluate --model jev-spec --test-category single_turn
```

[`bfcl/handler.py`](bfcl/handler.py) is the whole integration: BFCL hands it
one test entry at a time, it runs this project's prediction, and returns the
calls for BFCL's checkers.

## Method

BFCL does not say which queries need one call, several, or none, so every
query goes through the same steps.

1. **Which functions, and how often.** One request asks:
   - a `Choice` over the offered functions plus a described `none`. Each
     function is shown with its description and its arguments'
     descriptions. Choosing `none` is abstaining;
   - how many times each function must be called;
   - when several functions are offered, how many different ones are needed
     and, for each, whether it is.

   The picked function is called once unless the other answers are
   confident: more than one function needs the chance of one to be under
   0.1, and more than one call needs it to be under 0.2. A confident
   "several functions" answer also overrides `none`, since no single
   function answers a request that needs two.

2. **Arguments**, for each function selected, asked the way that function's
   spec says. The gold function is never used, so picking the wrong function
   costs the arguments too.
3. **Several calls to one function.** Its words are labelled once and every
   value found for a parameter is kept in order. The values are zipped by
   position, so the first of each goes to the first call; a parameter with
   one value shares it. A number that can be read two ways, such as "30%",
   gets one more question choosing the reading.

A query offered no function at all is answered with no call.

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
- **Several calls.** How to select functions and count calls was worked out
  on `exec_parallel` and `exec_parallel_multiple`, two legacy BFCL splits
  that are not leaderboard categories. The two confidence thresholds were
  set with those and with how the count questions behaved on single-call
  queries from `simple`, `multiple`, and `live_multiple`. The override of
  `none` was added after seeing that nearly every `parallel_multiple` query
  abstained. So the parallel categories were not tuned for accuracy, but
  they were not untouched either.
- **Showing each function's arguments** when picking a function was added
  without a dev comparison and is in every number above.
- **The spec method.** The idea came from classifying the test splits'
  out-of-reach arguments by what producing them would take, and the kinds a
  spec can have were chosen from that classification. The spec files were
  written without sight of any query or answer. The runtime was developed
  on the dev split, where it scored 55.8 in a pilot, and the test splits
  were then run once with it.

### Scoring

The reported numbers are scored by BFCL itself. This project's own runner
uses `metrics.py`, a port of BFCL's AST match: the right function, every
required parameter present, every value among the gold answer's accepted
values, strings compared after removing case, spaces, and light punctuation,
and, for several calls, each gold call matched by a different predicted call
in any order. The port was checked call by call against BFCL's checker
before the official runner was wired in; see
[docs/research/2026-10-04-scorer-parity.md](docs/research/2026-10-04-scorer-parity.md).

### Known limitations

- Arguments of type `dict`, `tuple`, or `any`, and arrays of dicts or arrays
  (37 of 3,375 parameters) are never asked.
- The spec is only as good as its author's reading of each description. A
  parameter wrongly made a closed set loses values outside it, which is why
  closed-set accuracy is slightly lower with the spec (94.2 against 95.0).
- Number candidates cover common written forms only. Units, arithmetic, and
  inferred values are not attempted.
- A place is completed with a US state or one of about 120 countries.
- A scalar string takes one run of words carrying its label, and a scalar
  number the first such word that parses.
- The ceiling ignores overlap between arguments, so it is an upper bound.
- When one function is called several times, options, flags, and dates are
  asked once and shared by every call, and values are matched by the order
  the request gives them in.
- Java and JavaScript functions have no authored spec.
- Multi-turn, web search, and memory categories are not run; see the table
  under Results.

## This project's own runner

Before the official runner was wired in, the project had its own loader and
a port of BFCL's AST match, on BFCL v3. It is kept for development: it is
fast, it saves every answer's probabilities, and it reports things BFCL
does not, such as argument accuracy and the ceiling. `uv run run test spec`
runs it and `uv run report test spec` scores the saved runs in `results/`.

It agrees with BFCL. Word labelling, mean of three runs on v3, against
BFCL's score on v4:

| Category      | This runner | BFCL |
| ------------- | ----------- | ---- |
| Python simple | 66.8        | 67.3 |
| Multiple      | 69.3        | 70.0 |
| Irrelevance   | 89.7        | 90.4 |
| Live multiple | 47.4        | 47.2 |
| Parallel      | 45.8        | 45.5 |

Its committed runs predate one fix, the override that stops Jev abstaining
when several functions are needed, so its parallel-multiple numbers are
lower than BFCL's above.

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

Several calls from one request, on `exec_parallel` (50 queries), as the
share of queries with every call right:

| Approach                                                    | All calls right |
| ----------------------------------------------------------- | --------------- |
| Label once, zip each parameter's values by position         | 48              |
| The same, a repeated value shared and arrays split per call | 52              |
| The same, Jev choosing between two readings of a number     | 58              |
| Asking each call's arguments separately ("the second call") | 22 to 24        |
| Asking which call each value belongs to                     | 50              |

Asked directly, Jev gave the right number of calls on 98% of those queries.

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

These commands are this project's own runner, used for development. The
reported results come from BFCL's runner; see
[Running the benchmark](#running-the-benchmark).

Run the test splits three times with the authored spec:

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
├── jev.py      # Select functions, build questions, call Jev, decode calls
├── metrics.py  # AST match, accuracies, ceiling
└── scripts/
    ├── run.py     # Run a stage's splits
    └── report.py  # Score the saved runs
```

`specs/` holds the authored specs and the format they were written to.
`bfcl/` holds the handler and launcher for BFCL's official runner, and the
results and score tables it wrote.

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
