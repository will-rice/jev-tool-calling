# Jev on BFCL: zero-shot tool calling

Status: implemented. See README.md for results.

## Goal

Measure how TypeSafe AI's Jev (a non-generative "System One" decision model)
performs zero-shot at tool calling: picking a function and filling its
arguments from a natural-language request.

Success: one command evaluates three single-turn splits of the Berkeley
Function Calling Leaderboard (BFCL) v3 and reports tool accuracy, argument
accuracy, and full-call accuracy, with per-query predictions saved for later
analysis. Full-call accuracy uses BFCL's AST-match rule, so it is comparable
to the public leaderboard's AST columns.

## Background

- Jev takes a `state` and a map of typed questions (`Choice`, `Score`,
  `Noul`) and returns typed answers with probabilities in one parallel pass.
  It generates no text and has no tool-calling primitive.
- TypeSafe's function calling cookbook builds tool calling from questions:
  one `Choice` picks the function, one `Choice` per closed-set argument, one
  `Noul` per boolean, and a `Noul` "is this stated?" gate so an unmentioned
  optional argument keeps its default. It leaves free strings and numbers
  unfilled. Its `Dispatcher` is example code, not part of `typesafe-sdk`.
- Access is `typesafe-sdk`: `TypeSafeClient().system_one(state, questions)`,
  authenticated by `TYPESAFE_API_KEY`, with built-in retry on 429/529.
- Data is `gorilla-llm/Berkeley-Function-Calling-Leaderboard` on the Hugging
  Face Hub (Apache-2.0), JSON lines. Each row has `id`, `question` (one turn,
  one user message in the splits used here), and `function` (name,
  description, JSON-schema-like `parameters`). Gold answers are in
  `possible_answer/`: per argument, a list of accepted values, where `""`
  means the argument may be omitted.
- Measured on revision `61fc0608`:

  | Split         | Queries | Functions offered | Median / max words |
  | ------------- | ------- | ----------------- | ------------------ |
  | `simple`      | 400     | 1                 | 14 / 48            |
  | `multiple`    | 200     | 2 to 4            | 14 / 37            |
  | `irrelevance` | 240     | 1, none fits      | not measured       |

- BFCL arguments are mostly open values. Across `simple` and `multiple`,
  about 1,700 gold arguments are strings, integers, floats, or arrays and
  about 65 are enums. The cookbook's closed-set recipe therefore covers
  almost none of them.
- Roughly two-thirds of queries have every required argument verbatim in the
  query text (a lower bound: the probe did not match `5` to `5.0` or handle
  arrays). The rest need conversion: `$1M` to `1000000`, `50mH` to `0.05`,
  `11 PM` to `23`, `x^2` to `x**2`, `past decade` to `10`.
- The SNIPS benchmark found that labelling each word (token classification)
  beats picking spans: 76.4 against 72.2 slot F1, at 58% of the tokens. It
  also found that labelled-field instructions and a described `none` option
  each matter by double-digit points.
- The official scorer, `bfcl-eval`, pins 35 dependencies (`numpy==1.26.4`,
  `vllm`, `faiss-cpu`, every provider SDK) and its `ast_checker` imports the
  model registry at module load. It is not a usable dependency.

## Scope

In scope: zero-shot evaluation on `simple`, `multiple`, and `irrelevance`.

Out of scope: the parallel, multi-turn, live, Java, JavaScript, SQL, and REST
splits; a names-only ablation; number-word, unit, or date conversion; handing
off to a language model. Each would be a separate experiment.

## Design

### Package layout

Rename `agent_harness` to `jevtools` and delete the template's `Agent`,
`Task`, `Harness`, and `Result`, for the reasons recorded in the SNIPS spec:
they turn exceptions into a zero score, and an API failure must stop the run.

| Module              | Responsibility                                          |
| ------------------- | ------------------------------------------------------- |
| `config.py`         | Constants: repo, revision, splits, model, workers, runs |
| `models.py`         | Frozen pydantic models for parsed data and predictions  |
| `data.py`           | Download and parse the splits and their gold answers    |
| `jev.py`            | Build questions, call Jev, decode answers into a call   |
| `metrics.py`        | AST match, accuracies, and the ceiling                  |
| `scripts/run.py`    | Entry point: predict concurrently, save, score, log     |
| `scripts/report.py` | Compare saved runs per split                            |

### Models

- `Parameter`: `name`, `type`, `description`, `enum` (optional), `required`.
- `Function`: `name`, `description`, `parameters`.
- `Example`: `id`, `split`, `query`, `functions`, `gold` (function name to
  accepted values per argument; empty for `irrelevance`).
- `Call`: `name` and `arguments`, or no call (abstain).
- `Prediction`: the `Example`, the predicted `Call`, the tool probabilities,
  each argument question's probabilities, input tokens per request, and the
  model version the API returned.

Records are written with `model_dump_json` and reload with
`model_validate_json`.

### Questions

Two requests per query, both with `{"query": ...}` as state.

1. **Tool.** One `Choice` named `tool`. Options are the offered function
   names with their BFCL descriptions, plus `none` with its own description
   ("no offered function can answer the request"). `none` is abstention.
2. **Arguments.** Skipped when the tool is `none`. One request for the picked
   function, carrying:
   - **Open-valued parameters** (string, integer, float, array): one `Choice`
     named `token_{i}` per query word. Options are those parameter names with
     their descriptions, plus a described `none`. Instructions are labelled
     fields: `function`, `words_before`, `word`, `words_after`, and the
     question "Which argument of `function` does `word` supply in `query`?".
   - **Enum parameters**: one `Choice` over the enum values.
   - **Boolean parameters**: one `Noul`.
   - **Optional enum and boolean parameters**: one more `Noul` asking whether
     the query says anything about the parameter. If not, it is omitted.

Words are the query split on whitespace.

### Decoding

- Tool: the `choice` of the `tool` answer.
- Words labelled with the same parameter and adjacent to each other form one
  value. Up to `MAX_GAP = 2` unlabelled words between two words of the same
  parameter join it.
- A value is coerced by the parameter's declared type:
  - string: the text, with punctuation stripped from both ends
  - integer, float: `int()` or `float()` of that text
  - array: one element per run of words, each coerced by the item type
- For a non-array parameter with more than one run, the first run is used.
- A value that fails to coerce means the argument is omitted. No fallback.
- Enum: the `choice`. Boolean: `probability >= 0.5`. Optional and not stated:
  omitted.

### Metrics

Reported per split, as the mean and range over `RUNS = 3`.

- **Tool accuracy** (`multiple`): queries where the picked function is the
  gold function.
- **Irrelevance accuracy** (`irrelevance`): queries where the tool is `none`.
- **Argument accuracy** (`simple`, `multiple`): over gold arguments that may
  not be omitted, on queries with the right tool, the share whose predicted
  value is accepted.
- **Full-call accuracy** (`simple`, `multiple`): BFCL's AST match. The name
  is right, every required parameter is present, no unknown parameter is
  present, and every value is in its accepted list. Strings are compared
  after BFCL's normalisation (lower-cased, with spaces and `,./-_*^`
  removed).
- **Ceiling** (`simple`, `multiple`): full-call accuracy of an oracle that
  labels every word correctly, under the same decoding. It separates "Jev
  mislabelled a word" from "the value cannot be expressed by labelling
  words".

`metrics.py` ports the AST-match rule for one Python call. Parity with
`bfcl-eval`'s `simple_function_checker` is checked once during development on
the gold answers and on a sample of predictions, and the result is recorded
in the README.

### Run

`scripts/run.py` loads `.env`, builds one `TypeSafeClient`, and maps the
prediction function over each split with `thread_map`. Each run is saved as
`results/{split}-run{n}.jsonl`. Metrics, input tokens per request type, and
the returned model version are logged.

`--limit N` evaluates the first N queries of each split and writes to
`results/{split}-run{n}-first{N}.jsonl`, so it cannot overwrite a full run.
Full-run results are committed; limited-run files are git-ignored.

### Errors

No fallbacks. SDK retries handle rate limits. Any other API error, a missing
key, or a row that fails to parse raises and stops the run. There is no
resume logic.

## Revisions from planning

Found by prototyping the decoding offline against the pinned data, with no
API calls. Where these differ from the sections above, these apply.

- **Words split punctuation off.** Words are matches of
  `[-+]?\d[\d.]*(?:[eE][-+]?\d+)?|\w+|[^\w\s]`, not whitespace-separated
  chunks, and a value is the query text from its first word's start to its
  last word's end. The ceiling rises from 69% to 76% on `simple` and from 74%
  to 80% on `multiple`, because `A(3,4)` and `$5` no longer glue a value to
  its punctuation. It costs more questions per query.
- **Enum arrays are sets.** 74 parameters are arrays whose items are an enum.
  Each gets one `Noul` per member, as in the cookbook, and the argument is
  the members answered yes. An empty set is omitted.
- **No gap filling for arrays.** Filling unlabelled words between two words
  of the same parameter would merge `Santa Barbara and Monterey` into one
  element. Gap filling applies to non-array parameters only.
- **An array is omitted if any element fails to coerce.**
- **Unsupported types get no question.** `dict`, `tuple`, `any`, and arrays
  of dicts or arrays (37 parameters) are never filled.
- **Five functions have no fillable parameter.** Their argument request is
  skipped and the call has no arguments.
- **The ceiling is per argument, not a simulated labelling.** An argument is
  expressible if some run of consecutive words coerces to an accepted value
  (for an array, if every element of some accepted list does). Closed-set
  arguments always are; unsupported types never are. A query counts toward
  the ceiling when every argument that may not be omitted is expressible.
  This ignores overlap between arguments, so it is an upper bound.
- **Tool accuracy is also reported for `simple`**, where it is the share of
  queries on which Jev did not abstain.
- **Seven parameters have no description**; their option is offered
  undescribed.
- **One gold answer (`simple_363`) names its function without the module
  prefix.** On `simple` the gold function is the one offered function, which
  is what BFCL's checker does.

## Probe results

21 queries were sent to `jev-latest` before the full run: the first 8 of
`simple` plus its longest query, the first 6 of `multiple`, and the first 6
of `irrelevance`. The API returned model `jev-1.13.0`.

- **Size.** No request was rejected. The tool request used 387 input tokens
  on average. The word request used 5,465 on average with punctuation split
  off and 3,807 with whitespace words. The longest query (`simple_337`, 95
  words once split, 48 by whitespace) used 24,381 and 11,302.
- **Abstention.** With a described `none`, 0 of 9 `simple` queries abstained
  and 6 of 6 `irrelevance` queries did.
- **Projected cost.** About 10.8 million input tokens for three runs of all
  three splits.
- **Splitting helps where the ceiling said it would.** For `A(3,4) and
B(1,2)`, whitespace words give `pointA` the single word `A(3,4)`; split
  words label `3` and `4` separately. For `inches?`, whitespace keeps the
  question mark on the word.
- **Jev also labels the word that names an argument.** In "a base of 10
  units and height of 5 units" it labels `base` and `10` as `base`, and
  `height` and `5` as `height`. Decoding as specified then fills the gap,
  reads `base of 10`, fails to parse a number, and omits the argument. This
  was not seen on SNIPS, where slot names rarely appear in the utterance.

## Dev-split revision

Added after the first live run, which scored 28% call accuracy on 20
`simple` queries whose ceiling was 100%. BFCL has no train split, so the
fixes were developed on `live_simple` (233 usable queries), which is run as
a `dev` stage and never reported as a result. See
`docs/research/2026-10-04-argument-decoding.md`. This supersedes the word
question and decoding described above.

- The `none` option of the word question says that a word which only names
  or introduces an argument is `none`.
- A word's label counts only at `LABEL_THRESHOLD = 0.7` probability or more.
- A number is read word by word: a number parameter takes the first labelled
  word that parses, and a number array every labelled word that parses.
  Merging and gap filling apply to strings only.
- `run` and `report` take a required `stage` argument: `dev` runs
  `live_simple` once, `test` runs the three reported splits three times.
- Rows the questions cannot pose are left out when loading: more than one
  message, an enum of non-strings, or more than `MAX_WORDS = 95` words. The
  test splits have none; `live_simple` has 25.

## Testing

pytest, functional style, no mocks:

- Question building: the tool question offers every function plus `none`;
  the argument request has one question per word over the open-valued
  parameters, a `Choice` per enum, a `Noul` per boolean, and a stated gate
  only for optional closed-set parameters.
- Decoding: single-word and multi-word values, gap filling, two runs into an
  array, integer and float coercion, a failed coercion omitting the argument.
- Metrics: AST match on hand-built calls covering an accepted alternative, an
  omitted optional argument, a missing required argument, an unknown
  argument, and string normalisation; the ceiling on hand-built examples.
- Models: a `Prediction` survives a JSON round trip unchanged.
- Data: the real files parse to 400, 200, and 240 examples, and every gold
  function name is among its query's offered functions.

The live API path is verified by a `--limit` run, not by tests.

## Dependencies

Add `typesafe-sdk`, `huggingface-hub`, `tqdm`. Replace the key names in
`.env.example` with `TYPESAFE_API_KEY`.

## Documentation

`README.md` is rewritten for this project: what is measured, the results
table beside the ceiling, the method, setup, usage, the output format, and
the known limitations.

## References

- TypeSafe function calling cookbook:
  https://docs.typesafe.ai/cookbooks/function_calling.md
- TypeSafe docs index: https://docs.typesafe.ai/llms.txt
- BFCL dataset:
  https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
