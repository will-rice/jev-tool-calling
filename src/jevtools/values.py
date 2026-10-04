"""Find candidate values in a request and assemble values in code.

The model only ever chooses; these functions find what it chooses among and
turn its choices into the value a function expects.
"""

import re
import string
from calendar import monthrange
from collections.abc import Sequence
from datetime import date

from jevtools.config import NONE, WORD_PATTERN

NUMBER_EDGE = string.punctuation.replace("-", "").replace("+", "").replace(".", "")
INTEGER = re.compile(r"[-+]?\d+")
FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
SEPARATED = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
MAGNITUDES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "billion": 1e9,
}
PERCENT = ("%", "percent")
LIST_GLUE = (",", "and", "&", ";", "or", "[", "]", "(", ")")
JOINERS = ("-", "'", "&")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DAYS = tuple(str(day) for day in range(1, 32))
YEARS = tuple(str(year) for year in range(1900, 2051))
UNITED_STATES = "United States"
STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}
COUNTRIES = (
    "Afghanistan, Albania, Algeria, Argentina, Armenia, Australia, Austria, "
    "Azerbaijan, Bahamas, Bahrain, Bangladesh, Belarus, Belgium, Bolivia, "
    "Bosnia and Herzegovina, Botswana, Brazil, Bulgaria, Cambodia, Cameroon, "
    "Canada, Chile, China, Colombia, Costa Rica, Croatia, Cuba, Cyprus, "
    "Czech Republic, Denmark, Dominican Republic, Ecuador, Egypt, El Salvador, "
    "Estonia, Ethiopia, Fiji, Finland, France, Georgia, Germany, Ghana, Greece, "
    "Guatemala, Haiti, Honduras, Hungary, Iceland, India, Indonesia, Iran, Iraq, "
    "Ireland, Israel, Italy, Jamaica, Japan, Jordan, Kazakhstan, Kenya, Kuwait, "
    "Laos, Latvia, Lebanon, Libya, Lithuania, Luxembourg, Malaysia, Maldives, "
    "Malta, Mexico, Moldova, Monaco, Mongolia, Morocco, Mozambique, Myanmar, "
    "Nepal, Netherlands, New Zealand, Nicaragua, Nigeria, North Korea, Norway, "
    "Oman, Pakistan, Panama, Paraguay, Peru, Philippines, Poland, Portugal, "
    "Qatar, Romania, Russia, Rwanda, Saudi Arabia, Senegal, Serbia, Singapore, "
    "Slovakia, Slovenia, South Africa, South Korea, Spain, Sri Lanka, Sudan, "
    "Sweden, Switzerland, Syria, Taiwan, Tanzania, Thailand, Tunisia, Turkey, "
    "Uganda, Ukraine, United Arab Emirates, United Kingdom, Uruguay, Uzbekistan, "
    "Venezuela, Vietnam, Yemen, Zambia, Zimbabwe"
).split(", ")


def number_candidates(query: str, value_type: str) -> dict[str, str | None]:
    """Find every reading of every number in the request.

    Args:
        query: The request text.
        value_type: "integer" or "float"; an integer is offered whole numbers only.

    Returns:
        Each candidate as the text of its value, mapped to how the request
        wrote it when that differs, in request order.
    """
    spans = word_spans(query)
    found: dict[str, str | None] = {}
    for index in range(len(spans)):
        for option, source in number_readings(query, spans, index, value_type).items():
            found.setdefault(option, source)
    return found


def number_readings(
    query: str, spans: Sequence[tuple[int, int]], index: int, value_type: str
) -> dict[str, str | None]:
    """Find every way the word at an index can be read as a number.

    Tuned to over-find: the number as written, and also read with its
    thousands separators removed, scaled by a magnitude ("1.2M", "5
    million"), as a fraction if it is a percentage, and from a number word.
    A single letter is a magnitude only when it touches the number, since
    "5 m" is five metres.

    Args:
        query: The request text.
        spans: Each word's character offsets in the query.
        index: Which word to read.
        value_type: "integer" or "float"; an integer is offered whole numbers only.

    Returns:
        Each reading as the text of its value, mapped to how the request
        wrote it when that differs. Empty if the word is not a number.
    """
    word = query[spans[index][0] : spans[index][1]]
    readings: list[tuple[float, str | None]] = []
    value = coerce(word, "float")
    if isinstance(value, float):
        readings.append((value, None))
    elif SEPARATED.fullmatch(word):
        value = float(word.replace(",", ""))
        readings.append((value, word))
    elif word.lower() in NUMBER_WORDS:
        readings.append((float(NUMBER_WORDS[word.lower()]), word))
    if isinstance(value, float) and index + 1 < len(spans):
        following = query[spans[index + 1][0] : spans[index + 1][1]].lower()
        touching = spans[index + 1][0] == spans[index][1]
        written = query[spans[index][0] : spans[index + 1][1]]
        if following in MAGNITUDES and (len(following) > 1 or touching):
            readings.append((value * MAGNITUDES[following], written))
        if following in PERCENT:
            readings.append((value / 100, written))
    found: dict[str, str | None] = {}
    for reading, source in readings:
        if value_type == "integer" and reading != int(reading):
            continue
        option = str(int(reading) if value_type == "integer" else reading)
        found.setdefault(option, source)
    return found


def format_date(month: str, day: str, year: str, date_format: str) -> str | None:
    """Write a date from its chosen parts in the format a function expects.

    Only the parts the format uses are needed, so a month-and-year format
    does not need a day.

    Returns:
        The formatted date, or None if a needed part was not given or the
        parts do not make a date.
    """
    needed = (
        (month, any(code in date_format for code in ("%m", "%b", "%B"))),
        (day, "%d" in date_format),
        (year, "%Y" in date_format or "%y" in date_format),
    )
    if any(part == NONE and need for part, need in needed):
        return None
    month_number = 1 if month == NONE else MONTHS.index(month) + 1
    day_number = 1 if day == NONE else int(day)
    year_number = 2000 if year == NONE else int(year)
    if day_number > monthrange(year_number, month_number)[1]:
        return None
    return date(year_number, month_number, day_number).strftime(date_format)


def complete_place(
    city: str, state: str, country: str, place_format: str, no_state: str
) -> str:
    """Complete a city with its state or country as a function's format asks.

    A city already written with a comma is taken to be complete.

    Args:
        city: The place as cut from the request.
        state: The chosen US state abbreviation, or none.
        country: The chosen country, or none.
        place_format: The format when the place is in a US state.
        no_state: The format when it is not.
    """
    if "," in city:
        return city
    in_state = state != NONE and country in (UNITED_STATES, NONE)
    if in_state and place_format == "city, state_abbr":
        return f"{city}, {state}"
    if in_state and place_format == "city, state_name":
        return f"{city}, {STATES[state]}"
    abroad = country not in (UNITED_STATES, NONE)
    if (country != NONE and place_format == "city, country") or (
        abroad and no_state == "city, country"
    ):
        return f"{city}, {country}"
    return city


def word_spans(query: str) -> list[tuple[int, int]]:
    """Return each word's (start, end) character offsets in the query."""
    return [match.span() for match in re.finditer(WORD_PATTERN, query)]


def coerce(text: str, value_type: str | None) -> str | int | float | None:
    """Read a value of the given type from query text, or None if it is not one.

    Punctuation around the text is dropped. Nothing is converted: a number
    word, a unit, or a thousands separator makes a number unreadable.

    Raises:
        ValueError: If the type is not a string or a number.
    """
    number = text.strip(NUMBER_EDGE).rstrip(".")
    match value_type:
        case "string":
            return text.strip(string.punctuation) or None
        case "integer":
            return int(number) if INTEGER.fullmatch(number) else None
        case "float":
            return float(number) if FLOAT.fullmatch(number) else None
        case _:
            raise ValueError(f"Cannot read a {value_type} from words")
