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
the adopted configuration and scores 43.3 and 65.9. On that run the
threshold makes no difference to call accuracy (43.3 with or without it),
so its 1.3-point gain in the table is within run-to-run variation. It is
kept because it never scored lower.

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

## What is left

With the adopted configuration, of dev's 233 queries 18 are lost to
abstention and 105 are out of reach for word labelling. The abstentions have
`none` probabilities spread from 0.50 to 0.92, on queries such as "Order me
pizza" where the offered function is a food-ordering assistant. A threshold
on abstention was not tried: it trades against `irrelevance`, and there is
no dev split that measures both.
