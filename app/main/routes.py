import random
from datetime import date, datetime, timedelta

from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import current_user, login_required

from app.extensions import db
from app.models import ProductivityLog
from app.main.forms import LogForm

main_bp = Blueprint("main", __name__)


def _sample_dashboard():
    """Placeholder data for the design. Replaced with real queries next."""
    rng = random.Random(7)
    today = date.today()
    start = today - timedelta(days=363)
    start -= timedelta(days=(start.weekday() + 1) % 7)   # columns start on Sunday

    weeks, d = [], start
    while d <= today:
        week = []
        for _ in range(7):
            if d > today:
                week.append(None)
            else:
                score = rng.choice([0, 0, 35, 50, 62, 75, 88])
                level = 0 if score == 0 else 1 if score < 40 else 2 if score < 55 else 3 if score < 70 else 4
                week.append({"date": d.isoformat(), "score": score, "level": level})
            d += timedelta(days=1)
        weeks.append(week)

    labels, prev = [], None
    for w in weeks:
        month = date.fromisoformat(w[0]["date"]).strftime("%b")
        labels.append(month if month != prev else "")
        prev = month

    habits = [
        {"name": "Study 3h+", "color": "#6f42c1"},
        {"name": "Sleep 7h+", "color": "#0d6efd"},
        {"name": "Exercise", "color": "#fd7e14"},
        {"name": "Screen under 6h", "color": "#d63384"},
        {"name": "Took breaks", "color": "#20c997"},
    ]
    for h in habits:
        h["days"] = [rng.random() < 0.6 for _ in range(7)]
        h["done"] = sum(h["days"])

    total_done = sum(h["done"] for h in habits)
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    return {
        "greeting": greeting,
        "today_label": today.strftime("%a, %b %d"),
        "weeks": weeks,
        "month_labels": labels,
        "habits": habits,
        "percent": round(total_done / (len(habits) * 7) * 100),
        "today_index": today.weekday(),
    }


@main_bp.route("/")
def index():
    data = _sample_dashboard() if current_user.is_authenticated else None
    return render_template("index.html", data=data)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@main_bp.route("/log", methods=["GET", "POST"])
@login_required
def log_day():
    form = LogForm()
    if form.validate_on_submit():
        log = ProductivityLog.query.filter_by(
            user_id=current_user.id, log_date=form.log_date.data
        ).first()
        is_new = log is None
        if is_new:
            log = ProductivityLog(user_id=current_user.id, log_date=form.log_date.data)
            db.session.add(log)
        for field in ("study_hours", "sleep_hours", "screen_time", "exercise_minutes",
                      "caffeine_mg", "stress_level", "break_count", "study_sessions",
                      "self_rating", "notes"):
            setattr(log, field, getattr(form, field).data)
        db.session.commit()
        flash(("Day saved." if is_new else "Day updated.") + f" Goals met: {log.goals_met()}/5.", "success")
        return redirect(url_for("main.log_day"))
    return render_template("log_form.html", form=form)