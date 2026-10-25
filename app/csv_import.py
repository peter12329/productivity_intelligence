"""Validate a CSV file of habit logs (the reverse of /export.csv).

Pure Python: no Flask or database here, so the rules can be tested alone.
parse_csv() either returns clean rows or a list of errors. The route only
writes to the database when there are NO errors (all or nothing).

File format (same as the export):
    date,study_hours,sleep_hours,...      <- header, `date` required, any order
    2026-10-18,4,7.5,...                  <- one row per day, YYYY-MM-DD

Rules:
  * UTF-8 (a BOM from Excel is fine), comma separated, at most MAX_BYTES / MAX_ROWS
  * every column must be `date` or a known habit field (unknown columns are an error)
  * dates are real calendar dates, between MIN_DATE and today, no duplicates
  * empty cells are ignored (an import never erases a value)
  * numbers are plain decimals inside the habit's range; whole-number habits
    must be whole; choices must be one of the allowed values;
    checks are 0/1/true/false
"""
import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date

from app.habits import CONTEXT, HABITS

MAX_BYTES = 1_000_000          # 1 MB
MAX_ROWS = 5000                # about 13 years of daily logs
MAX_ERRORS = 20                # errors listed; the total is still counted
MIN_DATE = date(2000, 1, 1)

FIELDS = {h["field"]: h for h in HABITS + CONTEXT}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DECIMAL_RE = re.compile(r"^\d+(\.\d+)?$")
_WHOLE_RE = re.compile(r"^\d+(\.0+)?$")
_TRUE, _FALSE = {"1", "true"}, {"0", "false"}


@dataclass
class Parsed:
    rows: dict = field(default_factory=dict)      # date -> {field: value}
    errors: list = field(default_factory=list)    # first MAX_ERRORS messages
    error_total: int = 0
    rows_read: int = 0
    blank_rows: int = 0                           # a date with nothing else

    @property
    def ok(self):
        return self.error_total == 0 and bool(self.rows)


def _add_error(result, message):
    result.error_total += 1
    if len(result.errors) < MAX_ERRORS:
        result.errors.append(message)


def _parse_value(habit, text):
    """Return (value, None) or (None, reason)."""
    s = text.strip()
    kind = habit.get("type")
    if kind == "check":
        low = s.lower()
        if low in _TRUE:
            return True, None
        if low in _FALSE:
            return False, None
        return None, "use 1/0 or true/false"
    if kind == "choice":
        allowed = [value for value, _ in habit["choices"]]
        if habit["cast"] is str:
            low = s.lower()
            return (low, None) if low in allowed else (
                None, "must be one of: " + ", ".join(allowed))
        if not _WHOLE_RE.match(s):
            return None, "must be a whole number"
        number = int(float(s))
        return (number, None) if number in allowed else (
            None, f"must be one of: {', '.join(str(a) for a in allowed)}")
    # plain number
    whole = habit["cast"] is int
    if not (_WHOLE_RE if whole else _DECIMAL_RE).match(s):
        return None, "must be a whole number" if whole else "must be a plain number like 7.5"
    number = int(float(s)) if whole else float(s)
    if not habit["min"] <= number <= habit["max"]:
        return None, f"must be between {habit['min']} and {habit['max']}"
    return number, None


def parse_csv(raw, today=None):
    """Check `raw` (bytes) and return a Parsed result."""
    today = today or date.today()
    result = Parsed()

    if len(raw) > MAX_BYTES:
        _add_error(result, f"The file is larger than {MAX_BYTES // 1_000_000} MB.")
        return result
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        _add_error(result, "The file is not UTF-8 text. In Excel use 'CSV UTF-8'.")
        return result
    if "\x00" in text:
        _add_error(result, "The file contains binary data, not CSV text.")
        return result
    if not text.strip():
        _add_error(result, "The file is empty.")
        return result

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = [h.strip().lower() for h in next(reader)]
    except (StopIteration, csv.Error):
        _add_error(result, "The file has no header row.")
        return result

    if "date" not in header:
        _add_error(result, "The header needs a 'date' column.")
    if len(set(header)) != len(header):
        _add_error(result, "The header repeats a column name.")
    unknown = [h for h in header if h != "date" and h not in FIELDS]
    if unknown:
        shown = ", ".join(repr(u) for u in unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        _add_error(result, f"Unknown column(s): {shown}. Use the names from the export.")
    if len(header) < 2:
        _add_error(result, "The header has no habit columns.")
    if result.error_total:
        return result

    seen = {}
    try:
        for line_no, cells in enumerate(reader, start=2):
            if not any(c.strip() for c in cells):
                continue                                # fully blank line
            result.rows_read += 1
            if result.rows_read > MAX_ROWS:
                _add_error(result, f"More than {MAX_ROWS} rows. Split the file.")
                break
            if len(cells) > len(header):
                _add_error(result, f"Row {line_no}: more cells than columns.")
                continue
            row = dict(zip(header, cells))

            date_text = (row.get("date") or "").strip()
            if not _DATE_RE.match(date_text):
                _add_error(result, f"Row {line_no}: date '{date_text[:20]}' must look like 2026-10-18.")
                continue
            try:
                day = date.fromisoformat(date_text)
            except ValueError:
                _add_error(result, f"Row {line_no}: {date_text} is not a real date.")
                continue
            if day > today:
                _add_error(result, f"Row {line_no}: {date_text} is in the future.")
                continue
            if day < MIN_DATE:
                _add_error(result, f"Row {line_no}: {date_text} is before {MIN_DATE}.")
                continue
            if day in seen:
                _add_error(result, f"Row {line_no}: {date_text} appears twice (first on row {seen[day]}).")
                continue
            seen[day] = line_no

            values, row_ok = {}, True
            for name in header:
                if name == "date":
                    continue
                cell = (row.get(name) or "").strip()
                if not cell:
                    continue                            # empty = nothing to import
                value, reason = _parse_value(FIELDS[name], cell)
                if reason:
                    _add_error(result, f"Row {line_no} ({date_text}), {name}: '{cell[:20]}' {reason}.")
                    row_ok = False
                else:
                    values[name] = value
            if not row_ok:
                continue
            if values:
                result.rows[day] = values
            else:
                result.blank_rows += 1
    except csv.Error as exc:
        _add_error(result, f"The file could not be read as CSV ({exc}).")

    if not result.error_total and not result.rows:
        _add_error(result, "No rows with values were found.")
    return result
