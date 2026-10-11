from datetime import date, datetime, timedelta

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.extensions import db
from app.habits import BY_KEY, HABITS, goals_met, is_met, level
from app.main.forms import HabitForm
from app.models import ProductivityLog

main_bp = Blueprint("main", __name__)


def build_dashboard(user):
    today = date.today()

    # Heatmap starts on a Sunday, about one year back
    start = today - timedelta(days=363)
    start -= timedelta(days=(start.weekday() + 1) % 7)

    logs = ProductivityLog.query.filter(
        ProductivityLog.user_id == user.id,
        ProductivityLog.log_date >= start,
    ).all()
    by_date = {l.log_date: l for l in logs}

    weeks, d = [], start
    while d <= today:
        week = []
        for _ in range(7):
            if d > today:
                week.append(None)
            else:
                log = by_date.get(d)
                lvl = level(log) if log else 0
                tip = (f"{d.isoformat()}: {goals_met(log)}/{len(HABITS)} goals"
                       if lvl else f"{d.isoformat()}: no data")
                week.append({
                    "date": d.isoformat(), "level": lvl, "logged": lvl > 0,
                    "today": d == today, "tip": tip,
                })
            d += timedelta(days=1)
        weeks.append(week)

    labels, prev = [], None
    for w in weeks:
        month = date.fromisoformat(w[0]["date"]).strftime("%b")
        labels.append(month if month != prev else "")
        prev = month

    # This week, Monday to Sunday
    monday = today - timedelta(days=today.weekday())
    week_dates = [monday + timedelta(days=i) for i in range(7)]

    habits = []
    for h in HABITS:
        days = []
        for wd in week_dates:
            if wd > today or wd not in by_date:
                days.append(None)                     # future or no data
            else:
                days.append(is_met(h, by_date[wd]))   # True / False / None
        habits.append({
            "key": h["key"], "name": h["name"], "color": h["color"],
            "type": h.get("type", "number"),
            "days": days, "done": sum(1 for x in days if x),
        })

    elapsed = today.weekday() + 1
    percent = round(sum(h["done"] for h in habits) / (len(HABITS) * elapsed) * 100)

    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    return {
        "greeting": greeting,
        "today_label": today.strftime("%a, %b %d"),
        "weeks": weeks,
        "month_labels": labels,
        "habits": habits,
        "percent": percent,
        "today_index": today.weekday(),
        "today_logged": today in by_date and level(by_date[today]) > 0,
        "total_logs": sum(1 for l in logs if level(l) > 0),
    }


@main_bp.route("/")
def index():
    data = build_dashboard(current_user) if current_user.is_authenticated else None
    return render_template("index.html", data=data)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@main_bp.route("/habit/<key>", methods=["GET", "POST"])
@login_required
def log_habit(key):
    habit = BY_KEY.get(key)
    if habit is None:
        abort(404)

    form = HabitForm()
    if request.method == "GET":
        existing = ProductivityLog.query.filter_by(
            user_id=current_user.id, log_date=date.today()
        ).first()
        if existing is not None and getattr(existing, habit["field"]) is not None:
            form.value.data = getattr(existing, habit["field"])

    if form.validate_on_submit():
        if not habit["min"] <= form.value.data <= habit["max"]:
            form.value.errors.append(f"Enter a value from {habit['min']} to {habit['max']}.")
        else:
            log = ProductivityLog.query.filter_by(
                user_id=current_user.id, log_date=form.log_date.data
            ).first()
            if log is None:
                log = ProductivityLog(user_id=current_user.id, log_date=form.log_date.data)
                db.session.add(log)
            setattr(log, habit["field"], habit["cast"](form.value.data))
            db.session.commit()

            met = is_met(habit, log)
            flash(f"{habit['name']}: {'goal met' if met else 'goal not met'} "
                  f"({form.value.data:g} {habit['unit']}).",
                  "success" if met else "info")
            return redirect(url_for("main.index"))

    return render_template("habit_form.html", form=form, habit=habit)


@main_bp.route("/habit/<key>/toggle", methods=["POST"])
@login_required
def toggle_habit(key):
    habit = BY_KEY.get(key)
    if habit is None or habit.get("type") != "check":
        abort(404)

    log = ProductivityLog.query.filter_by(
        user_id=current_user.id, log_date=date.today()
    ).first()
    if log is None:
        log = ProductivityLog(user_id=current_user.id, log_date=date.today())
        db.session.add(log)

    current = getattr(log, habit["field"])
    setattr(log, habit["field"], None if current else True)
    db.session.commit()
    return redirect(url_for("main.index"))