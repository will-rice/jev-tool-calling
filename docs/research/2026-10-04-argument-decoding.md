# Argument decoding: what was tried on the dev split

BFCL has no train split, so nothing could be tuned on `simple` or `multiple`
without tuning on the reported results. Everything below was measured on
`live_simple`, which the benchmark does not report: 233 of its 258 queries
(the rest have a system message, an enum of non-strings, or more than 95
words). Its ceiling is 54.9% and Jev picked the gold function on 92.3% of
queries, abstaining on the other 18.

All scores are full-call accuracy (BFCL's AST match) and argument accuracy
from one run. Re-running the same configuration moved call accuracy by 0.4
points.

## The problem

The first version asked, for each word, "Which argument of `function` does
`word` supply in `query`?", with `none` described as "The word is not part
of any argument's value." Adjacent words with the same label were merged,
with gaps of up to two words filled.

Jev labelled the word that names an argument as well as its value:

| Query fragment          | Words labelled    | Decoded text  | Result       |
| ----------------------- | ----------------- | ------------- | ------------ |
| "a base of 10 units"    | `base`, `10`      | `base of 10`  | not a number |
| "a = 1, b = -3"         | `a`, `=`, `1`     | `a = 1`       | not a number |
| "from 2020 Addison St…" | `from`, `2020`, … | `from 2020…`  | wrong string |
| "version 'v2'"          | `version`, `v2`   | `version 'v2` | wrong string |

On dev this scored 24.5% call accuracy and 42.4% argument accuracy. Of the
arguments that were missed but within reach, 91 were strings decoded to the
wrong text and 33 were numbers that did not parse.

## What was tried

| Change                                                             | Call | Argument | API calls |
| ------------------------------------------------------------------ | ---- | -------- | --------- |
| First version                                                      | 24.5 | 42.4     |           |
| Numbers read word by word, not merged                              | 27.9 | 49.6     | none      |
| The same, labels below 0.7 probability ignored                     | 32.2 | 56.4     | none      |
| The same, threshold 0.9                                            | 41.6 | 61.7     | none      |
| The same, threshold 0.95                                           | 33.9 | 56.7     | none      |
| `none` reworded, numbers word by word, no threshold                | 42.5 | 64.6     | one run   |
| `none` reworded, numbers word by word, threshold 0.7 (**adopted**) | 43.8 | 66.3     | one run   |
| `none` reworded, threshold 0.9                                     | 39.5 | 60.8     | one run   |
| Adopted, plus a "which of these numbers" question per number       | 43.8 | 66.8     | one run   |

The reworded `none` is: "The word is not part of any argument's value. A
word that only names or introduces an argument, such as a field name, a
preposition, or punctuation around the value, is none."

Rows marked "none" under API calls were rescored from the saved answers:
every record keeps each word's option probabilities, so a decoding rule can
be changed without asking the model again.

The committed dev run, `results/live_simple-run1.jsonl`, is a fresh run of
the adopted configuration and scores 43.3 and 66.1. On a run of that
configuration the threshold made no difference to call accuracy (43.3 with
or without it), so its 1.3-point gain in the table is within run-to-run
variation. It is kept because it never scored lower.

The committed run also has a corrected word pattern. The runs in the table
split `1,000` into three words and read `5-10` as `5` and `-10`; with
numbers read word by word that decoded `1,000` as `1`. The fix came from
code review after the first test run and moved dev call accuracy by nothing
and argument accuracy from 65.9 to 66.1.

The earlier runs in the table were not kept: the first version's answers
were overwritten by the committed run.

## What was learned

- **Say what `none` covers.** Telling the model that a word which names or
  introduces an argument is `none` was worth 14.6 points of call accuracy on
  its own, more than any decoding rule. On SNIPS the same lesson was that
  `none` needs a description at all.
- **Cue words are the ones the model is unsure of.** Among words given a
  string parameter's label, those inside the gold value had a median
  probability of 0.96 and those outside 0.65. A threshold therefore removes
  cue words, and with the first wording a high one (0.9) recovered most of
  the loss. With the reworded `none` the model stops labelling them, a high
  threshold only removes value words, and 0.7 is best.
- **A number is one word.** Merging adjacent labelled words is right for a
  name and wrong for a number, where one extra word makes the text
  unparseable. Reading the first labelled word that parses is free.
- **Gap filling still matters for strings.** Without it call accuracy falls
  from 42.5 to 34.3: multi-word values lose their small words.
- **A separate question per number argument adds nothing.** Offering the
  query's numbers as the options of one `Choice` per argument changed no
  call's outcome once the word labels were fixed.

## Second round: string boundaries

After the first test run, the reachable errors left on the test splits were
mostly strings with a word too many or too few. Three things were tried on
dev, starting from the adopted configuration (43.3 call, 66.1 argument).

| Change                                                               | Call | Argument | API calls   |
| -------------------------------------------------------------------- | ---- | -------- | ----------- |
| Adopted configuration                                                | 43.3 | 66.1     |             |
| First readable run, punctuation gaps in arrays, de-duplicated arrays | 43.3 | 66.1     | none        |
| Unsure words bridged inside a scalar string                          | 44.2 | 66.6     | none        |
| The same, strongest run instead of first (**adopted**)               | 45.9 | 67.6     | none        |
| The same, plus a boundary question over each run and its trims       | 46.4 | 67.8     | one request |
| The same, with the run widened by a word on each side as well        | 47.2 | 68.8     | one request |

- **Bridging.** A scalar string's run continues across up to six unlabelled
  words if the model gave each at least 0.3 for the label. This repairs
  `my-bot-id`, where the hyphens scored 0.35 to 0.45 and cut the value to
  `my`.
- **Strongest run.** Of several runs with a scalar string's label, the one
  with the most probability summed over its words is used, not the first.
  This repairs "Yosemite National Park which locates at Mariposa, CA".
- **Boundary question.** One `Choice` per scalar string whose options are
  each run, the run without its first or last word, and the run with one
  more word on either side.

The first row's three fixes were proposed from errors seen on the test
splits (`S&P 500` split at the ampersand, names listed twice). Dev has
almost no string arrays, so it could not show whether they help, and they
were not adopted.

### What happened on the test splits

| Split      | Before | Bridging and strongest run | Plus boundary question |
| ---------- | ------ | -------------------------- | ---------------------- |
| dev        | 43.3   | 45.9                       | 47.6                   |
| `simple`   | 66.8   | 66.8                       | 66.6                   |
| `multiple` | 69.8   | 70.0                       | 69.5                   |

Neither change moved the test splits beyond run-to-run variation.

- Bridging and the strongest run fix failures that are common in dev's
  longer, messier queries and rare in `simple` and `multiple`.
- The boundary question fixed 7 `simple` queries and broke 7, and fixed 4
  `multiple` queries and broke 4. Every break is the model choosing the
  option with one more word: `C sharp major` for `C sharp` (probability
  0.99), `Rosewood Finish` for `Rosewood` (0.88), `deluxe room` for `deluxe`
  (0.89). The word labels had those right. On dev it fixed 4 and broke none.

The boundary question was removed: a third request per query for a gain that
does not transfer. Bridging and the strongest run were kept, since they cost
nothing and never scored lower.

Removing it used the test result, which is not a dev-only decision. It
cannot have raised the reported scores, which are the same with and without
it.

### What was learned

- **`live_simple` is a poor proxy for string boundaries on the test
  splits.** It found the cue-word problem, which was large everywhere, but
  a 4-point gain on it was worth nothing on `simple` and `multiple`.
- **Asking again with a wider option invites a longer answer.** Offered
  `deluxe` and `deluxe room`, the model picks the phrase that reads
  naturally. The per-word question does not have this bias, because each
  word is judged alone.

The committed runs are of the code without the boundary question: 45.9 on
dev, 66.3 on `simple`, and 70.2 on `multiple`.

## Third round: more context for each word

Two ways of giving a word's question more to go on, both on dev against the
adopted configuration (45.9 call, 67.3 argument).

### Autoregressive labelling

Words were labelled in order, one request per word, and each question was
shown the decisions already made for the earlier words.

| What each question was shown                        | Call | Argument | Fixed | Broke |
| --------------------------------------------------- | ---- | -------- | ----- | ----- |
| Nothing (independent labelling)                     | 45.9 | 67.3     |       |       |
| The argument values assembled so far                | 45.9 | 67.6     | 1     | 1     |
| The earlier words that were given an argument       | 44.6 | 66.3     | 1     | 4     |
| Every earlier word with its label, including `none` | 42.9 | 64.4     | 1     | 8     |

Showing earlier decisions made the model give fewer words an argument: 907
words under independent labelling, 873 with the values so far, 759 with the
full history. The words lost had a median probability of about 0.78 before
and 0.49 to 0.59 after, and the loss grew along the query (8%, 18%, and 21%
of labelled words in its first, middle, and last third with the full
history). It costs one request per word in sequence, about 20 round trips
per query. Not adopted.

### The function's description

The word questions carry the picked function's name but not its description.

| Variant                            | Call | Argument | Word-question tokens |
| ---------------------------------- | ---- | -------- | -------------------- |
| Control: same questions, run again | 45.5 | 67.3     | 1,277,103            |
| Description in every word question | 45.5 | 67.8     | 1,423,357            |
| Description once, in the state     | 46.4 | 68.0     | 1,286,549            |

Against the control the first variant fixed 2 queries and broke 2, and the
second fixed 5 and broke 3. The control differs from the committed run on
one query. Functions with opaque names (`ChaFod`, `ThinQ_Connect`) did not
benefit. Not adopted.

## Fourth round: a spec per function

Every experiment above selects words from the request, and none can pass
the ceiling. The values out of reach were classified by what producing them
would take (required arguments, `simple` and `multiple` together):

| Out-of-reach argument                                   | Arguments |
| ------------------------------------------------------- | --------- |
| Number in a common written form (`$1M`, `40%`, `three`) | 56        |
| String value listed in the parameter's description      | 22        |
| Date or time in a set format                            | 14        |
| Place completed with a state or country                 | 1         |
| Number needing unit conversion, arithmetic, inference   | 34        |
| Other string rewrite (`Apple` to `AAPL`)                | 25        |
| Arrays, dicts, tuples                                   | 22        |

The model knows more than it can say: asked which US state "a vegan
restaurant in New York" is in, with the 50 states as options, it answers
`NY` at probability 1.00. What it lacks is a way to produce a value that
was not offered. TypeSafe's cookbooks supply that from code:

- **Function calling.** A spec written from the signatures: a
  plain-language question per argument, a described line per option, and a
  question asking whether an optional argument is stated at all.
- **Pre-parsed value extraction.** Code finds candidate values with a
  pattern tuned to over-find, the model picks one per role, and code
  normalises the pick.
- **Date extraction.** The model chooses the month, day, and year; code
  assembles the date.

The spec here combines them. Each parameter is one of: text (labelled words,
as before), number (found, picked, normalised), options (described lines,
optionally open), flag, set, date (parts chosen, then formatted as the
function wants), place (city from the request, state or country chosen), or
skipped. `specs/FORMAT.md` is the instruction the spec was written from.

### How the spec was written

By a language model, from the function definitions alone. It was given the
format and the schemas in batches of about 62 functions and told to read
nothing else: no queries, no gold answers, no results. 131 dev functions
and 988 test functions were written this way and then normalised in code:
an authored kind the parameter's type cannot take falls back to the kind the
schema implies, a list of options with no enum behind it is made open so an
unlisted value can still be copied from the request, and a set with no enum
behind it becomes text.

### Dev pilot

One run answered every question, so each kind could be switched on alone.

| Configuration                        | Call | Argument |
| ------------------------------------ | ---- | -------- |
| Word labelling (committed run)       | 45.9 | 67.3     |
| Control: word labelling, asked again | 44.6 | 66.8     |
| Control plus dates                   | 51.5 | 71.9     |
| Control plus places                  | 47.6 | 69.0     |
| Control plus numbers                 | 45.1 | 67.3     |
| Control plus options                 | 44.2 | 68.8     |
| Everything                           | 55.8 | 76.5     |

The pilot's runtime was a scratch script; the numbers reported in the README
come from the committed code, which differs in one way: an open option's
unlisted value is copied from the request by word labelling, where the pilot
reused the earlier run's value.

## Fifth round: several calls from one request

BFCL's parallel categories ask for two to eight calls from one request,
mostly one function called with different values. Developed on
`exec_parallel` (50 queries) and `exec_parallel_multiple` (40), two legacy
BFCL splits that are not leaderboard categories.

### Filling the calls

| Approach                                                    | All calls right |
| ----------------------------------------------------------- | --------------- |
| Label once, zip each parameter's values by position         | 48              |
| The same, a repeated value shared and arrays split per call | 52              |
| The same, Jev choosing between two readings of a number     | 58              |
| Asking each call's arguments separately ("the second call") | 22 to 24        |
| Asking which call each value belongs to                     | 50              |

- **Zipping by position** keeps every value labelled for a parameter, in
  order, and gives the first of each to the first call. It needs no request
  beyond the labelling.
- **Asking per call** told the model there were n calls and to answer for
  the kth. It labelled words from every call anyway.
- **The grouping question** named each call by one parameter's value and
  asked which call every other value belongs to. It fixed nothing: of 24
  failing queries, 16 needed a value that is not in the request, 5 had the
  wrong number of calls, and 3 were grouped or labelled wrongly.
- **Readings.** A labelled number that can be read two ways gets a Choice
  between them, with the parameter's description. Four queries had such a
  number and three became right.

### Choosing how many

Asked directly how many times a function must be called, Jev was right on
98% of `exec_parallel` queries; counting the zipped values was right on
76%. Asked how many different functions a request needs, it was right on
97.5% of `exec_parallel_multiple` queries.

BFCL does not say which queries are parallel, so these questions are asked
of every query, and they must answer one on a single-call query:

| Decision                                  | `simple` | `multiple` | `live_multiple` | parallel dev |
| ----------------------------------------- | -------- | ---------- | --------------- | ------------ |
| Count, taking its top answer              | 97.3     |            |                 | 98.0         |
| Count, more than one only if p(one) < 0.2 | 99.3     |            |                 | 96.0         |
| A yes/no per function, at 0.5             |          | 86.5       | 73.7            | 90.0         |
| Different functions, top answer           |          | 97.0       | 95.3            | 97.5         |
| The same, several only if p(one) < 0.1    |          | 98.5       | 99.7            | 97.5         |

A yes/no per function says yes to too many functions on a single-call
query. The count of different functions, acted on only when confident,
keeps single-call queries intact, and the yes/no answers then only rank
which functions to take.

One interaction was found on the test split itself: the tool question asks
for one function and answers `none` for a request that needs two, so 169 of
200 `parallel_multiple` queries abstained. A confident "several functions"
answer now overrides `none` when at least two functions are each judged
needed. On `live_irrelevance` that turns 6 of 882 correct abstentions into
calls.

## The official runner

The numbers in the README are BFCL's: `bfcl/handler.py` plugs this project's
prediction into `bfcl-eval`, which loads BFCL v4, runs every entry, and
scores it. The first official run found a bug the project's own runner
could not have: BFCL appends a sentence about Python syntax to every
function description, so no function matched the key its spec was written
under, and a fallback silently used the schema-derived spec. Both methods
scored the same, with identical token counts. The handler now removes the
sentence and the fallback applies only to Java and JavaScript, which have
no spec.

With word labelling, where that bug made no difference, BFCL's scores and
this project's runner agree to within a point on every shared category.

## What is left

With the adopted configuration, of dev's 233 queries 18 are lost to
abstention and 105 are out of reach for word labelling. The abstentions have
`none` probabilities spread from 0.50 to 0.92, on queries such as "Order me
pizza" where the offered function is a food-ordering assistant. A threshold
on abstention was not tried: it trades against `irrelevance`, and there is
no dev split that measures both.
