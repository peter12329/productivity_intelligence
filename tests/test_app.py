"""App tests (in-memory SQLite). Run from the project root:  python -m pytest tests/test_app.py -q

They do not need the trained ML models.
"""
import io
from datetime import date, timedelta

import pytest

from app import create_app
from app.extensions import db
from app.habits import HABITS
from app.models import ProductivityLog, User
from app import preferences as prefs
from app.notes import DayNote, get_note
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
    all_score = dashboard_stats(db.session.get(User, 1))["summary"]["best_score"]
    assert all_score == round(3 / len(HABITS) * 100, 1)   # 3 of every goal habit


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


def test_day_page_shows_own_day_and_rejects_bad_dates(app):
    prefs.save_tracked(1, [h["key"] for h in HABITS])
    db.session.add(ProductivityLog(user_id=1, log_date=date.today() - timedelta(days=1),
                                   study_hours=4, sleep_hours=8, stress_level=3))
    db.session.commit()
    c = login(app, 1)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    r = c.get(f"/day/{yesterday}")
    assert r.status_code == 200 and b"goals met" in r.data
    assert c.get("/day/not-a-date").status_code == 404
    future = (date.today() + timedelta(days=5)).isoformat()
    assert c.get(f"/day/{future}").status_code == 302           # sent to today instead


def test_day_page_does_not_show_another_users_values(app):
    day = date(2026, 10, 1)
    db.session.add(ProductivityLog(user_id=2, log_date=day, study_hours=9.5))
    db.session.commit()
    prefs.save_tracked(1, [h["key"] for h in HABITS])
    r = login(app, 1).get(f"/day/{day.isoformat()}")
    assert r.status_code == 200 and b"9.5" not in r.data


def test_day_page_requires_login(app):
    assert app.test_client().get("/day/2026-10-01").status_code == 302


def upload(client, text, **form):
    return client.post("/import", data={"file": (io.BytesIO(text.encode()), "logs.csv"), **form},
                       content_type="multipart/form-data")


def count_logs(user_id):
    return ProductivityLog.query.filter_by(user_id=user_id).count()


def test_import_adds_new_days_for_the_current_user_only(app):
    c = login(app, 1)
    r = upload(c, "date,study_hours,sleep_hours\n2026-10-01,4,8\n2026-10-02,2,6\n")
    assert r.status_code == 200 and b"Imported." in r.data
    assert count_logs(1) == 2 and count_logs(2) == 0


def test_import_with_any_error_saves_nothing(app):
    c = login(app, 1)
    r = upload(c, "date,sleep_hours\n2026-10-01,8\n2026-10-02,99\n")
    assert b"Nothing was imported" in r.data and count_logs(1) == 0


def test_check_only_does_not_save(app):
    r = upload(login(app, 1), "date,sleep_hours\n2026-10-01,8\n", dry_run="1")
    assert b"Check passed" in r.data and count_logs(1) == 0


def test_import_keeps_existing_values_unless_overwrite_is_chosen(app):
    day = date(2026, 10, 1)
    db.session.add(ProductivityLog(user_id=1, log_date=day, sleep_hours=5))
    db.session.commit()
    c = login(app, 1)
    upload(c, "date,sleep_hours,stress_level\n2026-10-01,8,3\n", mode="fill")
    log = ProductivityLog.query.filter_by(user_id=1, log_date=day).one()
    assert (log.sleep_hours, log.stress_level) == (5, 3)           # kept 5, filled the empty stress
    upload(c, "date,sleep_hours\n2026-10-01,8\n", mode="overwrite")
    db.session.refresh(log)
    assert log.sleep_hours == 8


def test_import_requires_login_and_a_file(app):
    assert app.test_client().get("/import").status_code == 302
    r = login(app, 1).post("/import", data={}, content_type="multipart/form-data")
    assert b"Choose a CSV file first" in r.data


def test_export_then_import_round_trips(app):
    db.session.add(ProductivityLog(user_id=1, log_date=date(2026, 10, 1), study_hours=4,
                                   sleep_hours=7.5, journaled=True, weather="rainy"))
    db.session.commit()
    exported = login(app, 1).get("/export.csv").get_data(as_text=True)
    ProductivityLog.query.filter_by(user_id=1).delete()
    db.session.commit()
    assert b"Imported." in upload(login(app, 1), exported).data
    log = ProductivityLog.query.filter_by(user_id=1).one()
    assert (log.study_hours, log.sleep_hours, log.journaled, log.weather) == (4, 7.5, True, "rainy")


# ------------------------------------------------------------------- notes
def test_note_can_be_saved_replaced_and_cleared(app):
    day = date(2026, 10, 1)
    c = login(app, 1)
    c.post(f"/day/{day}/note", data={"note": "exam week"})
    assert get_note(1, day) == "exam week"
    c.post(f"/day/{day}/note", data={"note": "slept badly"})
    assert get_note(1, day) == "slept badly" and DayNote.query.count() == 1
    c.post(f"/day/{day}/note", data={"note": "   "})
    assert get_note(1, day) is None


def test_note_is_optional_and_does_not_count_as_a_logged_day(app):
    day = date(2026, 10, 1)
    login(app, 1).post(f"/day/{day}/note", data={"note": "just a note"})
    assert ProductivityLog.query.filter_by(user_id=1).count() == 0        # no log was created
    assert dashboard_stats(db.session.get(User, 1), core_habits()) == {"empty": True}


def test_note_that_is_too_long_is_refused_and_notes_are_private(app):
    day = date(2026, 10, 1)
    c = login(app, 1)
    c.post(f"/day/{day}/note", data={"note": "x" * 501})
    assert get_note(1, day) is None
    c.post(f"/day/{day}/note", data={"note": "mine"})
    assert get_note(2, day) is None
    assert b"mine" not in login(app, 2).get(f"/day/{day}").data
    assert c.post("/day/2999-01-01/note", data={"note": "future"}).status_code == 404


def test_deleting_a_day_removes_its_note(app):
    day = date(2026, 10, 1)
    log = ProductivityLog(user_id=1, log_date=day, study_hours=2)
    db.session.add(log)
    db.session.commit()
    c = login(app, 1)
    c.post(f"/day/{day}/note", data={"note": "bye"})
    c.post(f"/history/{log.id}/delete")
    assert get_note(1, day) is None


# ------------------------------------------------------- heatmap habit filter
def test_heatmap_filter_shows_one_habit_and_ignores_unknown_keys(app):
    prefs.save_tracked(1, [h["key"] for h in HABITS])
    db.session.add(ProductivityLog(user_id=1, log_date=date.today(), sleep_hours=8))
    db.session.commit()
    c = login(app, 1)
    r = c.get("/?habit=sleep")
    assert r.status_code == 200 and b"Sleep 7h+ - goal met" in r.data and b"days met" in r.data
    assert c.get("/?habit=not-a-habit").status_code == 200
    assert b"goal met" not in c.get("/").data


def test_heatmap_partial_returns_only_the_card_and_needs_login(app):
    assert app.test_client().get("/heatmap").status_code == 302
    prefs.save_tracked(1, [h["key"] for h in HABITS])
    db.session.add(ProductivityLog(user_id=1, log_date=date.today(), sleep_hours=8))
    db.session.commit()
    c = login(app, 1)
    r = c.get("/heatmap?habit=sleep")
    assert r.status_code == 200
    assert b"data-heat-form" in r.data and b"goal met" in r.data
    assert b"<html" not in r.data and b"<nav" not in r.data          # just the card, not the page
    assert c.get("/heatmap?habit=nope&year=abc").status_code == 200   # bad input falls back


def test_home_page_wraps_the_heatmap_for_in_place_updates(app):
    prefs.save_tracked(1, [h["key"] for h in HABITS])
    assert b'id="heat-wrap"' in login(app, 1).get("/").data
