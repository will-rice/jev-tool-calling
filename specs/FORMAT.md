# Spec format

You are writing a spec for tool-calling functions, in the style of TypeSafe's function-calling cookbook.
A decision model (it cannot write text; it can only pick one option from a list, or answer yes/no) will use
the spec to fill each function's arguments from a user's natural-language request. Code assembles the values.

Input: a JSON list of functions. Each has `name`, `description`, and `parameters` (each with `name`, `type`,
`description`, `required`, `enum`, `item_type`). Use each function's INDEX in the list as its id.

Output: a JSON object mapping the index (as a string) to `{"name": <function name>, "parameters": {<param name>: <param spec>}}`.
Every parameter of every function must have an entry.

## Param spec

```json
{"kind": "...", "question": "...", "stated": "...", ...kind-specific fields}
```

- `question`: one plain-language question about the IDEA the argument captures, as a user would think of it.
  Do not name it after the parameter ("Which resolution?" is bad; "How fine-grained should the bars be?" is good).
- `stated`: a yes/no question asking whether the request says anything at all about this argument. Include it
  only when `required` is false.

### kind (choose the first that fits)

1. `"options"` — the argument takes one of a known, listable set of string values: it has an `enum`, OR its
   description lists the allowed values (e.g. "Can be 'melting', 'freezing', 'vaporization'"), OR it is a unit /
   mode / category with a small fixed set clearly implied by the description.
   Fields: `"options": {"<exact value the function accepts>": "<one line saying what this option means, in words a user might use>"}`,
   `"open": true|false` (true if a value outside the list could also be valid, e.g. the description says "such as" or "e.g.").
   The option keys must be the exact strings the function accepts. Max 200 options.
   For a boolean parameter use kind `"flag"` instead. For an array whose items have an enum use `"set"` with the same `options` field.
2. `"flag"` — a boolean. `question` must be a yes/no question that is true when the argument should be true.
3. `"number"` — an integer or float whose value the user states as a number (a quantity, amount, count, id, size, year...).
   Fields: `"unit": "<the unit the function expects, or null>"`.
4. `"date"` — a string holding a calendar date in a fixed format (not a time of day, not a range).
   Fields: `"format": "<Python strftime format the function expects, e.g. %Y-%m-%d or %m/%d/%Y>"`.
   Only use this when the description states or clearly implies the format.
5. `"place"` — a string naming a city-level location that must be completed in a stated format.
   Fields: `"place_format"`: one of `"city, state_abbr"`, `"city, state_name"`, `"city, country"`, `"city"`,
   and `"no_state": "city, country" | "city"` saying what to produce when the place is not in a US state.
   Do not use this for street addresses or coordinates.
6. `"text"` — any other string (names, titles, free text, identifiers, addresses, queries, code), or an array of
   strings or numbers. The value will be copied from the request.
7. `"skip"` — dict / tuple / any, or an array of dicts or arrays. It will not be filled.

Rules:

- Work only from the function definitions. You have no example requests and must not invent any.
- Be conservative with `"options"` when no enum exists: use it only if the description really does enumerate or
  fix the values. If unsure, use `"text"`.
- An integer parameter whose description maps names to numbers (e.g. "1 for Bangkok, 2 for Chiang Mai") is `"options"` with the numbers as string keys.
- Output must be valid JSON, nothing else in the file.
