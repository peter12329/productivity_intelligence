"""Optional free-text note for a day ("exam week", "slept badly").

Kept in its OWN table (`day_notes`), not on productivity_logs, on purpose:
  * a note alone never makes a day count as "logged", so streaks, the heatmap,
    the dashboard and the estimate are unaffected
  * the table is created by db.create_all() at startup, so no migration is needed
  * every function is scoped to one user_id
"""
from datetime import datetime, timezone

from app.extensions import db
from app.text_clean import clean_note

NOTE_MAX = 500


class DayNote(db.Model):
    __tablename__ = "day_notes"
    __table_args__ = (db.UniqueConstraint("user_id", "note_date", name="uq_note_user_date"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)      # plain integer, like UserPreference
    note_date = db.Column(db.Date, nullable=False)
    text = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))


def get_note(user_id, day):
    row = DayNote.query.filter_by(user_id=user_id, note_date=day).first()
    return row.text if row else None


def notes_for(user_id, days):
    """{date: text} for the given dates, in one query."""
    days = list(days)
    if not days:
        return {}
    rows = DayNote.query.filter(DayNote.user_id == user_id, DayNote.note_date.in_(days)).all()
    return {r.note_date: r.text for r in rows}


def save_note(user_id, day, text):
    """Store, replace or (for empty text) remove the note. Returns 'saved' or 'cleared'.

    Raises ValueError when the text is longer than NOTE_MAX.
    """
    text = clean_note(text, NOTE_MAX)
    row = DayNote.query.filter_by(user_id=user_id, note_date=day).first()
    if not text:
        if row:
            db.session.delete(row)
            db.session.commit()
        return "cleared"
    if row is None:
        db.session.add(DayNote(user_id=user_id, note_date=day, text=text))
    else:
        row.text = text
    db.session.commit()
    return "saved"


def delete_note(user_id, day):
    DayNote.query.filter_by(user_id=user_id, note_date=day).delete()
