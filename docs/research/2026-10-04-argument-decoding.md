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

## What is left

With the adopted configuration, of dev's 233 queries 18 are lost to
abstention and 105 are out of reach for word labelling. The abstentions have
`none` probabilities spread from 0.50 to 0.92, on queries such as "Order me
pizza" where the offered function is a food-ordering assistant. A threshold
on abstention was not tried: it trades against `irrelevance`, and there is
no dev split that measures both.
