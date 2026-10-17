from datetime import date, datetime, timedelta

from flask import (Blueprint, abort, flash, jsonify, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from wtforms import DateField, FloatField, SelectField, SubmitField
from wtforms.validators import InputRequired

from app.extensions import db
from app.habits import BY_KEY, CONTEXT, HABITS, goals_met, is_met, level
from app.insights.service import latest_estimate
from app.models import ProductivityLog
from app.stats import dashboard_stats

main_bp = Blueprint("main", __name__)

# ---------------------------------------------------------------- log forms
class NumberHabitForm(FlaskForm):
    """Log page for number habits (sleep, deep work, stress, ...)."""
    log_date = DateField("Date", default=date.today, validators=[InputRequired()])
    value = FloatField("Value", validators=[InputRequired()])
    submit = SubmitField("Save")


class ChoiceHabitForm(FlaskForm):
    """Log page for dropdown habits (caffeine, junk food, day type, ...).

    The allowed choices are set per habit in the route, and WTForms rejects
    any posted value that is not one of them.
    """
    log_date = DateField("Date", default=date.today, validators=[InputRequired()])
    value = SelectField("Value", choices=[], coerce=str,
                        validators=[InputRequired()])
    submit = SubmitField("Save")


def _is_choice(habit):
    return habit.get("type") == "choice" or bool(habit.get("choices"))


def _get_log(user_id, log_date, create=False):
    log = ProductivityLog.query.filter_by(user_id=user_id, log_date=log_date).first()
    if log is None and create:
        log = ProductivityLog(user_id=user_id, log_date=log_date)
        db.session.add(log)
    return log


def _choice_label(habit, value):
    for v, label in habit.get("choices", []):
        if str(v) == str(value):
            return label
    return None


# ---------------------------------------------------------------- dashboard
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
    today_log = by_date.get(today)

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
            "value": getattr(today_log, h["field"], None) if today_log else None,
        })

    # Context inputs (no goal): today's value or None
    context = []
    for c in CONTEXT:
        raw = getattr(today_log, c["field"], None) if today_log else None
        shown = None
        if raw is not None:
            shown = _choice_label(c, raw) or str(raw)
        context.append({"key": c["key"], "name": c["name"],
                        "color": c.get("color", "#6c757d"),
                        "value": raw, "shown": shown})

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
        "context": context,
        "percent": percent,
        "today_index": today.weekday(),
        "today_logged": today_log is not None and level(today_log) > 0,
        "total_logs": sum(1 for l in logs if level(l) > 0),
    }


@main_bp.route("/")
def index():
    data = None
    if current_user.is_authenticated:
        data = build_dashboard(current_user)
        data["estimate"] = latest_estimate(current_user.id)
    return render_template("index.html", data=data)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", stats=dashboard_stats(current_user))


@main_bp.route("/api/dashboard")
@login_required
def api_dashboard():
    return jsonify(dashboard_stats(current_user))


# ---------------------------------------------------------------- log a habit
@main_bp.route("/habit/<key>", methods=["GET", "POST"])
@login_required
def log_habit(key):
    habit = BY_KEY.get(key)
    if habit is None or habit.get("type") == "check":
        abort(404)

    choice = _is_choice(habit)
    form = ChoiceHabitForm() if choice else NumberHabitForm()
    if choice:
        form.value.choices = [(str(v), label) for v, label in habit["choices"]]

    if request.method == "GET":
        existing = _get_log(current_user.id, date.today())
        current = getattr(existing, habit["field"], None) if existing else None
        if current is not None:
            form.value.data = str(current) if choice else current

    if form.validate_on_submit():
        lo, hi = habit.get("min"), habit.get("max")
        if form.log_date.data > date.today():
            form.log_date.errors.append("Date can't be in the future.")
        elif not choice and lo is not None and hi is not None \
                and not lo <= form.value.data <= hi:
            form.value.errors.append(f"Enter a value from {lo} to {hi}.")
        else:
            try:
                stored = habit["cast"](form.value.data)
            except (TypeError, ValueError):
                form.value.errors.append("That value isn't valid.")
                stored = None
            if stored is not None:
                log = _get_log(current_user.id, form.log_date.data, create=True)
                setattr(log, habit["field"], stored)
                db.session.commit()

                if choice:
                    shown = _choice_label(habit, form.value.data) or form.value.data
                else:
                    unit = habit.get("unit", "")
                    shown = f"{form.value.data:g} {unit}".strip()

                if "met" in habit:                        # goal habits
                    met = is_met(habit, log)
                    flash(f"{habit['name']}: "
                          f"{'goal met' if met else 'goal not met'} ({shown}).",
                          "success" if met else "info")
                else:                                     # context inputs
                    flash(f"{habit['name']} saved: {shown}.", "success")
                return redirect(url_for("main.index"))

    return render_template("habit_form.html", form=form, habit=habit)


@main_bp.route("/habit/<key>/toggle", methods=["POST"])
@login_required
def toggle_habit(key):
    habit = BY_KEY.get(key)
    if habit is None or habit.get("type") != "check":
        abort(404)

    log = _get_log(current_user.id, date.today(), create=True)
    current = getattr(log, habit["field"])
    setattr(log, habit["field"], None if current else True)
    db.session.commit()
    return redirect(url_for("main.index"))


@main_bp.route("/profile")
@login_required
def profile():
    total = ProductivityLog.query.filter_by(user_id=current_user.id).count()
    return render_template("profile.html", total_logs=total)
