"""Cleaning of day notes (app/text_clean.py). No database needed."""
import pytest

from app.text_clean import clean_note


def test_trims_and_normalises_line_endings():
    assert clean_note("  hello\r\nworld  ", 500) == "hello\nworld"


def test_empty_values_give_empty_text():
    assert clean_note("", 500) == "" and clean_note(None, 500) == "" and clean_note("  \n ", 500) == ""


def test_control_characters_are_dropped_but_tabs_stay():
    assert clean_note("a\x00b\x07c", 500) == "abc"
    assert clean_note("a\tb", 500) == "a\tb"


def test_runs_of_blank_lines_are_limited():
    assert clean_note("a\n\n\n\n\nb", 500) == "a\n\nb"


def test_length_limit_counts_a_browser_newline_once():
    assert len(clean_note("x" * 500, 500)) == 500
    assert clean_note("a\r\n" * 100, 500)           # 100 lines are well under 500


def test_too_long_is_rejected_not_cut():
    with pytest.raises(ValueError):
        clean_note("x" * 501, 500)
