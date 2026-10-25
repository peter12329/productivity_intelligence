"""Clean free text typed by the user (day notes).

Pure Python (no Flask), so it can be tested alone.
"""
import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")      # keeps \t and \n


def clean_note(text, max_len):
    """Return the note ready to store ('' = no note). Raises ValueError if too long.

    Line endings become \\n (browsers send \\r\\n, which would count twice
    against the limit), control characters are dropped, blank lines are
    limited to one, and spaces around the text are trimmed.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_len:
        raise ValueError(f"Keep the note to {max_len} characters or fewer (it has {len(text)}).")
    return text
