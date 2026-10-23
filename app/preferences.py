"""Which habits each user chooses to track.

One row per user in `user_preferences`. The table is created on first use, so no
migration step is needed. A user with no row has not done the setup yet
(`is_onboarded` is False) and is treated as tracking everything until then.
"""
import json
from datetime import datetime

from app.extensions import db

# Habits (by database field) that cannot be turned off: the estimate needs them.
CORE_FIELDS = {"study_hours", "sleep_hours", "stress_level"}


class UserPreference(db.Model):
    __tablename__ = "user_preferences"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, unique=True, index=True)
    tracked = db.Column(db.Text, nullable=False, default="[]")      # JSON list of habit keys
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


_table_ready = False


def _ensure_table():
    global _table_ready
    if not _table_ready:
        UserPreference.__table__.create(bind=db.engine, checkfirst=True)
        _table_ready = True


def _row(user_id):
    _ensure_table()
    return UserPreference.query.filter_by(user_id=user_id).first()


def is_onboarded(user_id) -> bool:
    return _row(user_id) is not None


def tracked_keys(user_id, all_keys, core_keys) -> set:
    """Habit keys this user tracks. Everything until they choose; core habits always."""
    row = _row(user_id)
    if row is None:
        return set(all_keys)
    try:
        chosen = set(json.loads(row.tracked))
    except ValueError:
        chosen = set()
    return (chosen & set(all_keys)) | set(core_keys)


def save_tracked(user_id, keys) -> None:
    _ensure_table()
    row = UserPreference.query.filter_by(user_id=user_id).first()
    if row is None:
        row = UserPreference(user_id=user_id)
        db.session.add(row)
    row.tracked = json.dumps(sorted(set(keys)))
    db.session.commit()
