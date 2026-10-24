"""App tests (in-memory SQLite). Run from the project root:  python -m pytest tests/test_app.py -q

They do not need the trained ML models.
"""
from datetime import date, timedelta

import pytest

from app import create_app
from app.extensions import db
from app.habits import HABITS
from app.models import ProductivityLog, User
from app import preferences as prefs
from app.stats import dashboard_stats
from config import Config


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    WTF_CSRF_ENABLED = False
    SECRET_KEY = "test"


@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        for i in (1, 2):
            u = User(id=i, username=f"user{i}", email=f"u{i}@example.com")
            u.set_password("password123")
            db.session.add(u)
        db.session.commit()
        yield app


def login(app, user_id):
    c = app.test_client()
    with c.session_transaction() as s:
        s["_user_id"] = str(user_id)
        s["_fresh"] = True
    return c


def core_habits():
    return [h for h in HABITS if h["field"] in prefs.CORE_FIELDS]


def test_dashboard_score_uses_only_tracked_habits(app):
    today = date.today()
    db.session.add(ProductivityLog(user_id=1, log_date=today, study_hours=4,
                                   sleep_hours=8, stress_level=3))
    db.session.commit()
    stats = dashboard_stats(db.session.get(User, 1), core_habits())
    assert stats["summary"]["best_score"] == 100.0        # 3 of the 3 tracked goals
    assert len(stats["habits"]) == 3
    assert dashboard_stats(db.session.get(User, 1))["summary"]["best_score"] == 37.5   # 3 of all 8


def test_context_only_day_is_not_a_zero_percent_day(app):
    today = date.today()
    db.session.add(ProductivityLog(user_id=1, log_date=today, weather="rainy"))
    db.session.add(ProductivityLog(user_id=1, log_date=today - timedelta(days=1),
                                   study_hours=4, sleep_hours=8, stress_level=3))
    db.session.commit()
    stats = dashboard_stats(db.session.get(User, 1), core_habits())
    assert stats["summary"]["total_logs"] == 1
    assert stats["summary"]["avg_score"] == 100.0


def test_only_context_logged_gives_empty_dashboard(app):
    db.session.add(ProductivityLog(user_id=1, log_date=date.today(), weather="rainy"))
    db.session.commit()
    assert dashboard_stats(db.session.get(User, 1), core_habits()) == {"empty": True}


def test_skipped_journaling_is_filled_as_not_done(app):
    from app.insights.service import fill_for_user
    prefs.save_tracked(1, [h["key"] for h in core_habits()])
    assert fill_for_user(1).get("journaled") == 0


def test_history_lists_only_own_logs_and_delete_is_scoped(app):
    mine = ProductivityLog(user_id=1, log_date=date.today(), study_hours=2)
    theirs = ProductivityLog(user_id=2, log_date=date.today(), study_hours=5)
    db.session.add_all([mine, theirs])
    db.session.commit()
    mine_id, theirs_id = mine.id, theirs.id
    prefs.save_tracked(1, [h["key"] for h in HABITS])
    c = login(app, 1)
    assert c.get("/history").status_code == 200
    assert c.post(f"/history/{theirs_id}/delete").status_code == 404
    assert db.session.get(ProductivityLog, theirs_id) is not None
    assert c.post(f"/history/{mine_id}/delete").status_code == 302
    assert db.session.get(ProductivityLog, mine_id) is None


def test_csv_export_has_only_own_rows(app):
    db.session.add_all([
        ProductivityLog(user_id=1, log_date=date(2026, 10, 1), study_hours=2),
        ProductivityLog(user_id=2, log_date=date(2026, 10, 2), study_hours=5),
    ])
    db.session.commit()
    r = login(app, 1).get("/export.csv")
    assert r.status_code == 200 and r.mimetype == "text/csv"
    lines = r.get_data(as_text=True).strip().splitlines()
    assert lines[0].startswith("date,study_hours")
    assert len(lines) == 2 and lines[1].startswith("2026-10-01")


def test_export_requires_login(app):
    assert app.test_client().get("/export.csv").status_code == 302
