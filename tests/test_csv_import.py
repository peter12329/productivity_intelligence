"""Rules of the CSV import (app/csv_import.py). No database needed.

Run from the project root:  python -m pytest tests/test_csv_import.py -q
"""
from datetime import date

import pytest

from app.csv_import import MAX_ROWS, parse_csv

TODAY = date(2026, 10, 25)


def parse(text, **kw):
    return parse_csv(text.encode() if isinstance(text, str) else text, today=TODAY, **kw)


def test_valid_file_is_converted_to_the_right_types():
    r = parse("date,study_hours,sleep_hours,caffeine_mg,stress_level,journaled,junk_food,day_type,weather\n"
              "2026-10-18,4,7.5,7,3,1,0,class,Sunny\n")
    assert r.ok
    assert r.rows[date(2026, 10, 18)] == {
        "study_hours": 4.0, "sleep_hours": 7.5, "caffeine_mg": 7, "stress_level": 3,
        "journaled": True, "junk_food": 0, "day_type": "class", "weather": "sunny"}


def test_excel_bom_crlf_and_any_column_order():
    assert parse("\ufeffsleep_hours,date\r\n8,2026-10-18\r\n").ok


def test_empty_cells_are_skipped_and_never_erase():
    r = parse("date,study_hours,sleep_hours\n2026-10-18,,8\n2026-10-19,,\n")
    assert r.rows == {date(2026, 10, 18): {"sleep_hours": 8.0}}
    assert r.blank_rows == 1


@pytest.mark.parametrize("text, needle", [
    ("sleep_hours\n8\n", "'date' column"),
    ("date,notes\n2026-10-18,hi\n", "Unknown column"),
    ("date\n2026-10-18\n", "no habit columns"),
    ("date,sleep_hours,sleep_hours\n2026-10-18,8,8\n", "repeats"),
    ("date,sleep_hours\n18/10/2026,8\n", "must look like"),
    ("date,sleep_hours\n2026-02-30,8\n", "not a real date"),
    ("date,sleep_hours\n2026-10-26,8\n", "future"),
    ("date,sleep_hours\n1999-12-31,8\n", "before"),
    ("date,sleep_hours\n2026-10-18,8\n2026-10-18,7\n", "twice"),
    ("date,sleep_hours\n2026-10-18,25\n", "between 0 and 24"),
    ("date,stress_level\n2026-10-18,0\n", "between 1 and 10"),
    ("date,stress_level\n2026-10-18,3.5\n", "whole number"),
    ("date,sleep_hours\n2026-10-18,-1\n", "plain number"),
    ("date,sleep_hours\n2026-10-18,nan\n", "plain number"),
    ("date,sleep_hours\n2026-10-18,1e1\n", "plain number"),
    ("date,sleep_hours\n2026-10-18,=1+1\n", "plain number"),
    ("date,caffeine_mg\n2026-10-18,11\n", "one of"),
    ("date,task_difficulty\n2026-10-18,6\n", "one of"),
    ("date,junk_food\n2026-10-18,2\n", "one of"),
    ("date,weather\n2026-10-18,snowy\n", "one of"),
    ("date,journaled\n2026-10-18,maybe\n", "1/0"),
    ("date,sleep_hours\n2026-10-18,8,9\n", "more cells"),
    ("", "empty"),
    ("date,sleep_hours\n", "No rows"),
    ("date,sleep_hours\n2026-10-18,8\x00\n", "binary"),
])
def test_bad_input_is_rejected_with_a_clear_message(text, needle):
    r = parse(text)
    assert not r.ok and needle in " ".join(r.errors)


def test_whole_number_with_point_zero_is_accepted():
    assert parse("date,stress_level\n2026-10-18,3.0\n").rows[date(2026, 10, 18)]["stress_level"] == 3


def test_not_utf8_and_too_large_are_rejected():
    assert "UTF-8" in " ".join(parse(b"\xff\xfe\x00\x00date").errors)
    assert "larger" in " ".join(parse(b"date,sleep_hours\n" + b"x" * 1_000_001).errors)


def test_too_many_rows_is_rejected():
    start = date(2005, 1, 1).toordinal()
    body = "\n".join(f"{date.fromordinal(start + i)},8" for i in range(MAX_ROWS + 5))   # distinct, valid days
    assert "rows" in " ".join(parse("date,sleep_hours\n" + body).errors).lower()


def test_one_bad_row_rejects_the_whole_file_and_errors_are_capped():
    good = "2026-10-01,8\n"
    bad = "".join(f"2026-01-{(i % 28) + 1:02d},99\n" for i in range(100))
    r = parse("date,sleep_hours\n" + good + bad)
    assert not r.ok and len(r.errors) == 20 and r.error_total > 20
