import csv
import json
import math
import os
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from flask import (Blueprint, Response, abort, current_app, flash, jsonify, redirect,
                   render_template, request, send_from_directory, url_for)
from flask_login import current_user, login_required
from sqlalchemy import extract
from sqlalchemy.exc import SQLAlchemyError
from flask_wtf import FlaskForm
from wtforms import DateField, FloatField, SelectField, SubmitField
from wtforms.validators import InputRequired

from app import preferences as prefs
from app.csv_import import FIELDS as IMPORT_FIELDS
from app.csv_import import MAX_BYTES as IMPORT_MAX_BYTES
from app.csv_import import MAX_ROWS as IMPORT_MAX_ROWS
from app.csv_import import parse_csv
from app.extensions import db
from app.habits import BY_KEY, CONTEXT, HABITS, is_met
from app.insights.service import latest_estimate
from app.models import ProductivityLog
from app.notes import NOTE_MAX, delete_note, get_note, notes_for, save_note
from app.stats import dashboard_stats

# ------------------------------------------------------------- user themes
THEMES = {
    "midnight-purple": {
        "name": "Midnight Purple",
        "description": "Deep navy backgrounds with violet accents.",
    },
    "sage-cream": {
        "name": "Sage & Cream",
        "description": "Muted greens and warm cream surfaces.",
    },
    "modern-blue": {
        "name": "Modern Blue",
        "description": "Clean surfaces with calm blue accents.",
    },
    "dark-neon": {
        "name": "Dark Neon",
        "description": "Charcoal surfaces with neon green accents.",
    },
    "warm-minimalist": {
        "name": "Warm Minimalist",
        "description": "Soft beige surfaces and terracotta accents.",
    },
}

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
# ----------------------------------------------------- what each user tracks
# Card text for the setup page, by database field (falls back to the habit's own name).
UI_TITLES = {
    "study_hours": "Deep work", "sleep_hours": "Sleep", "stress_level": "Stress",
    "caffeine_mg": "Caffeine", "journaled": "Journaling", "junk_food": "Junk food",
    "disruptions": "Disruptions", "study_sessions": "Deep work sessions",
    "exercise_minutes": "Exercise",
    "free_hours": "Free hours", "task_difficulty": "Task difficulty",
    "day_type": "Day type", "weather": "Weather",
}
UI_BLURBS = {
    "study_hours": "Hours of focused work",
    "sleep_hours": "How long you slept",
    "stress_level": "How stressed you felt, 1 to 10",
    "caffeine_mg": "Your caffeine level, 0 to 10",
    "journaled": "One tap when you have journaled",
    "junk_food": "Whether you had junk food",
    "disruptions": "Interruptions during focus time",
    "exercise_minutes": "Minutes of exercise or movement",
    "study_sessions": "How many focused sessions",
    "free_hours": "Free time in your day",
    "task_difficulty": "How hard the day's work is",
    "day_type": "Class day, deadline, free day",
    "weather": "Sunny, rainy, and so on",
}
CONTEXT_COLORS = ["#0dcaf0", "#fd7e14", "#20c997", "#6f42c1"]


def _all_keys():
    return [h["key"] for h in HABITS] + [c["key"] for c in CONTEXT]


def _core_keys():
    return {h["key"] for h in HABITS if h["field"] in prefs.CORE_FIELDS}


def tracking_for(user):
    """(tracked keys, visible habits, visible context inputs) for this user."""
    keys = prefs.tracked_keys(user.id, _all_keys(), _core_keys())
    return keys, [h for h in HABITS if h["key"] in keys], [c for c in CONTEXT if c["key"] in keys]


def _is_logged(log, visible):
    """A day counts as logged when any habit the user tracks has a value."""
    return any(getattr(log, h["field"], None) is not None for h in visible)


def _met_count(log, visible):
    return sum(1 for h in visible if is_met(h, log) is True)


def _level(log, visible):
    """0 = nothing logged, 1-4 = share of the user's own goals met."""
    if not visible or not _is_logged(log, visible):
        return 0
    return max(1, min(4, math.ceil(4 * _met_count(log, visible) / len(visible))))


def _setup_cards(current):
    """Cards for the setup page: (group title, hint, cards)."""
    core = _core_keys()

    def card(b, i=0):
        field = b["field"]
        return {"key": b["key"], "field": field, "locked": b["key"] in core, "on": b["key"] in current,
                "title": UI_TITLES.get(field, b["name"]), "blurb": UI_BLURBS.get(field, ""),
                "color": b.get("color") or CONTEXT_COLORS[i % len(CONTEXT_COLORS)]}

    return [
        ("Always on", "The estimate needs these, so they stay on.", [card(h) for h in HABITS if h["key"] in core]),
        ("Optional habits", "Turn off anything you do not do.", [card(h) for h in HABITS if h["key"] not in core]),
        ("Quick context", "Day details that help the estimate. Skip any and typical values are used.",
         [card(c, i) for i, c in enumerate(CONTEXT)]),
    ]


def heatmap_years(user):
    """Calendar years in which the user has any log, newest first (this year always included)."""
    rows = (db.session.query(extract("year", ProductivityLog.log_date))
            .filter(ProductivityLog.user_id == user.id).distinct().all())
    years = {int(r[0]) for r in rows if r[0] is not None}
    years.add(date.today().year)
    return sorted(years, reverse=True)


def _streaks(by_date, range_start, range_end, today, is_logged):
    """(current streak, longest streak in the shown period), in consecutive logged days.

    The current streak ends today, or yesterday if today is not logged yet, so it
    does not drop to 0 every morning before you log.
    """
    logged = {dt for dt, l in by_date.items() if is_logged(l)}
    best = run = 0
    d = range_start
    while d <= range_end:
        run = run + 1 if d in logged else 0
        best = max(best, run)
        d += timedelta(days=1)
    d = today if today in logged else today - timedelta(days=1)
    current = 0
    while d in logged:
        current += 1
        d -= timedelta(days=1)
    return current, best


def build_dashboard(user, year=None, focus=None):
    """Data for the home page. `year` picks a calendar year for the heatmap;
    None means the rolling last 12 months (the past 52 weeks). `focus` is the key
    of one tracked goal habit: the heatmap then shows only that habit (met / not met)."""
    today = date.today()
    tracked, visible, visible_ctx = tracking_for(user)
    focus_habit = next((h for h in visible if h["key"] == focus), None)     # unknown key = all goals
    years = heatmap_years(user)
    if year not in years:                         # None or a year with no logs
        year = None
    roll_start = today - timedelta(days=363)
    if year:
        range_start, range_end = date(year, 1, 1), min(date(year, 12, 31), today)
    else:
        range_start, range_end = roll_start, today
    # the grid starts on the Sunday on or before the first day
    start = range_start - timedelta(days=(range_start.weekday() + 1) % 7)

    monday = today - timedelta(days=today.weekday())
    logs = ProductivityLog.query.filter(
        ProductivityLog.user_id == user.id,
        ProductivityLog.log_date >= min(start, roll_start, monday),
        ProductivityLog.log_date <= today,
    ).all()
    by_date = {l.log_date: l for l in logs}

    weeks, d = [], start
    while d <= range_end:
        week = []
        for _ in range(7):
            if d < range_start or d > range_end:
                week.append(None)                 # outside the chosen period
            else:
                log = by_date.get(d)
                if focus_habit:
                    state = is_met(focus_habit, log) if log else None       # True / False / None
                    lvl = 0 if state is None else (4 if state else 1)
                    tip = (f"{d.isoformat()}: {focus_habit['name']} - "
                           f"{'goal met' if state else 'not met'} ({_display(focus_habit, getattr(log, focus_habit['field']))})"
                           if state is not None else f"{d.isoformat()}: no data for this habit")
                else:
                    lvl = _level(log, visible) if log else 0
                    tip = (f"{d.isoformat()}: {_met_count(log, visible)}/{len(visible)} goals"
                           if lvl else f"{d.isoformat()}: no data")
                week.append({
                    "date": d.isoformat(), "level": lvl, "logged": lvl > 0,
                    "today": d == today, "tip": tip,
                })
            d += timedelta(days=1)
        weeks.append(week)

    labels, prev = [], None
    for w in weeks:
        first = next((c for c in w if c), None)   # a week can begin with blank cells
        month = date.fromisoformat(first["date"]).strftime("%b") if first else ""
        labels.append(month if month and month != prev else "")
        if month:
            prev = month

    if focus_habit:                       # a day counts when THIS goal was met
        counts = lambda l: is_met(focus_habit, l) is True
    else:                                 # a day counts when anything you track was logged
        counts = lambda l: _level(l, visible) > 0
    heat_count = sum(1 for dt, l in by_date.items() if range_start <= dt <= range_end and counts(l))
    streak_current, streak_longest = _streaks(by_date, range_start, range_end, today, counts)
    days_in_range = (range_end - range_start).days + 1
    heat_options = [{"label": "Last 12 months", "year": None}] + [
        {"label": str(y), "year": y} for y in years]

    # This week, Monday to Sunday
    week_dates = [monday + timedelta(days=i) for i in range(7)]
    today_log = by_date.get(today)

    habits = []
    for h in visible:
        days = []
        for wd in week_dates:
            if wd > today or wd not in by_date:
                days.append(None)                     # future or no data
            else:
                days.append(is_met(h, by_date[wd]))   # True / False / None
        habits.append({
            "key": h["key"], "name": h["name"], "color": h["color"], "field": h["field"],
            "type": h.get("type", "number"),
            "days": days, "done": sum(1 for x in days if x),
            "value": getattr(today_log, h["field"], None) if today_log else None,
        })

    # Context inputs (no goal): today's value or None
    context = []
    for i, c in enumerate(visible_ctx):
        raw = getattr(today_log, c["field"], None) if today_log else None
        shown = None
        if raw is not None:
            shown = _choice_label(c, raw) or str(raw)
        context.append({"key": c["key"], "name": c["name"], "field": c["field"],
                        "color": c.get("color") or CONTEXT_COLORS[i % len(CONTEXT_COLORS)],
                        "value": raw, "shown": shown})

    elapsed = today.weekday() + 1
    percent = round(sum(h["done"] for h in habits) / (max(1, len(visible)) * elapsed) * 100)

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
        "today_logged": today_log is not None and _level(today_log, visible) > 0,
        "total_logs": sum(1 for dt, l in by_date.items()
                          if roll_start <= dt <= today and _level(l, visible) > 0),   # last 12 months
        "heat_year": year,
        "streak_current": streak_current,
        "streak_longest": streak_longest,
        "heat_pct": round(heat_count / days_in_range * 100),
        "heat_title": str(year) if year else "Last 12 months",
        "heat_count": heat_count,
        "heat_options": heat_options,
        "show_year_picker": any(y < today.year for y in years),     # only once there is an earlier year
        "heat_scroll_end": year is None or year == today.year,
        "focus": focus_habit["key"] if focus_habit else None,
        "focus_options": [{"key": h["key"], "title": UI_TITLES.get(h["field"], h["name"])}
                          for h in visible],
    }


def _week_percent(user):
    """This week's goals met, as a percent of (tracked habits x days so far).

    Same number as build_dashboard()["percent"], without building the heatmap.
    """
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    visible = tracking_for(user)[1]
    logs = ProductivityLog.query.filter(
        ProductivityLog.user_id == user.id,
        ProductivityLog.log_date >= monday,
        ProductivityLog.log_date <= today).all()
    done = sum(1 for h in visible for l in logs if is_met(h, l) is True)
    return round(done / (max(1, len(visible)) * (today.weekday() + 1)) * 100)


@main_bp.route("/")
def index():
    data = None
    if current_user.is_authenticated:
        if not prefs.is_onboarded(current_user.id):         # first visit: choose what to track
            return redirect(url_for("main.setup"))
        data = build_dashboard(current_user, request.args.get("year", type=int),
                               request.args.get("habit"))
        data["estimate"] = latest_estimate(current_user.id)
    return render_template("index.html", data=data)


@main_bp.route("/heatmap")
@login_required
def heatmap_partial():
    """Only the heatmap card, for the in-page habit / year switcher (no full reload)."""
    data = build_dashboard(current_user, request.args.get("year", type=int),
                           request.args.get("habit"))
    return render_template("_heatmap.html", data=data)


@main_bp.route("/scoring")
def scoring():
    """Plain-language guide to how goals, daily scores and the weekly percent work."""
    personal = current_user.is_authenticated
    habits = tracking_for(current_user)[1] if personal else HABITS
    return render_template(
        "scoring.html", habits=habits, personal=personal, context=CONTEXT,
        core_keys=_core_keys(),
        per_habit=round(100 / len(habits), 1) if habits else 0,
    )


@main_bp.route("/setup", methods=["GET", "POST"])
@login_required
def setup():
    """Choose which habits to track (first visit, and any time from the profile)."""
    editing = prefs.is_onboarded(current_user.id)
    if request.method == "POST":
        chosen = set(request.form.getlist("track")) & set(_all_keys())
        prefs.save_tracked(current_user.id, chosen | _core_keys())
        flash("Saved. You can change what you track any time from your profile.", "success")
        return redirect(url_for("main.index"))
    current = prefs.tracked_keys(current_user.id, _all_keys(), _core_keys())
    return render_template("setup.html", groups=_setup_cards(current), editing=editing)


@main_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", stats=dashboard_stats(current_user, tracking_for(current_user)[1]))


@main_bp.route("/api/dashboard")
@login_required
def api_dashboard():
    return jsonify(dashboard_stats(current_user, tracking_for(current_user)[1]))



# ---------------------------------------------------------- log-page helpers
# Quick-pick values shown as chips on the log page, by habit key. Anything not
# listed falls back to the full grid when it is short (<= 12 values).
PRESETS = {
    "study": [0, 1, 2, 3, 4, 5, 6],
    "sleep": [5, 6, 6.5, 7, 7.5, 8, 9],
    "disruptions": [0, 1, 2, 3, 4, 5],
    "exercise": [0, 15, 30, 45, 60, 90],
    "free_hours": [0, 1, 2, 4, 6, 8, 10],
    "stress": list(range(1, 11)),
    "sessions": [0, 1, 2, 3, 4, 5, 6],
}
CHIP_LIMIT = 6            # dropdown habits with more choices keep a <select>


def _goal(habit, value):
    """True / False if `value` meets the habit's goal, None if it has no goal.

    Uses the app's own is_met(), so it works whatever shape HABITS has.
    """
    try:
        return is_met(habit, SimpleNamespace(**{habit["field"]: value}))
    except Exception:
        return None


def _date_arg():
    """?date=YYYY-MM-DD for the log page (today if missing, bad or future)."""
    raw = request.args.get("date")
    try:
        chosen = date.fromisoformat(raw) if raw else date.today()
    except ValueError:
        chosen = date.today()
    return min(chosen, date.today())


def _grid(habit):
    lo, hi = habit.get("min"), habit.get("max")
    if lo is None or hi is None:
        return []
    step = habit.get("step") or 1
    n = min(int(round((hi - lo) / step)) + 1, 400)
    return [round(lo + i * step, 6) for i in range(n)]


def _log_page_context(habit, choice, chosen, existing_value):
    """Extra data the log page needs: presets, goal table, current value."""
    ctx = {
        "chosen_iso": chosen.isoformat(),
        "today_iso": date.today().isoformat(),
        "yesterday_iso": (date.today() - timedelta(days=1)).isoformat(),
        "chosen_is_today": chosen == date.today(),
        "chip_limit": CHIP_LIMIT,
        "presets": [], "goal_points": [], "has_goal": False, "choice_ok": {},
        "current_value": existing_value, "current_label": None,
    }
    if existing_value is not None:
        if choice:
            ctx["current_label"] = _choice_label(habit, existing_value) or str(existing_value)
        else:
            ctx["current_label"] = f"{existing_value:g} {habit.get('unit', '')}".strip()

    if choice:
        ctx["choice_ok"] = {str(v): _goal(habit, v)
                            for v, _ in habit.get("choices", [])}
        ctx["has_goal"] = any(ok is not None for ok in ctx["choice_ok"].values())
        return ctx

    grid = _grid(habit)
    points = [(v, _goal(habit, v)) for v in grid]
    ctx["has_goal"] = any(ok is not None for _, ok in points)
    if ctx["has_goal"]:
        ctx["goal_points"] = [[v, bool(ok)] for v, ok in points if ok is not None]
    wanted = PRESETS.get(habit["key"]) or (grid if len(grid) <= 12 else [])
    lo, hi = habit.get("min"), habit.get("max")
    ctx["presets"] = [{"value": v, "ok": _goal(habit, v)} for v in wanted
                      if (lo is None or v >= lo) and (hi is None or v <= hi)]
    return ctx

# ---------------------------------------------------------------- log a habit
@main_bp.route("/habit/<key>", methods=["GET", "POST"])
@login_required
def log_habit(key):
    habit = BY_KEY.get(key)
    if habit is None or habit.get("type") == "check":
        abort(404)
    if habit["key"] not in tracking_for(current_user)[0]:
        flash(f"You are not tracking {UI_TITLES.get(habit['field'], habit['name'])}. Turn it on first.", "info")
        return redirect(url_for("main.setup"))

    choice = _is_choice(habit)
    form = ChoiceHabitForm() if choice else NumberHabitForm()
    if choice:
        form.value.choices = [(str(v), label) for v, label in habit["choices"]]

    chosen = _date_arg() if request.method == "GET" else (
        form.log_date.data or date.today())
    existing_value = None
    if request.method == "GET":
        form.log_date.data = chosen
        existing = _get_log(current_user.id, chosen)
        existing_value = getattr(existing, habit["field"], None) if existing else None
        if existing_value is not None:
            form.value.data = str(existing_value) if choice else existing_value
    elif form.log_date.data:
        existing = _get_log(current_user.id, form.log_date.data)
        existing_value = getattr(existing, habit["field"], None) if existing else None

    if form.validate_on_submit():
        lo, hi = habit.get("min"), habit.get("max")
        if form.log_date.data > date.today():
            form.log_date.errors.append("Date can't be in the future.")
        elif not choice and lo is not None and hi is not None \
                and not lo <= form.value.data <= hi:
            form.value.errors.append(f"Enter a value from {lo:g} to {hi:g}.")
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
                    shown = f"{form.value.data:g} {habit.get('unit', '')}".strip()

                met = is_met(habit, log) if _goal(habit, stored) is not None else None
                if met is None:                           # context inputs: no goal
                    flash(f"{habit['name']} saved: {shown}.", "success")
                else:
                    flash(f"{habit['name']}: "
                          f"{'goal met' if met else 'goal not met'} ({shown}).",
                          "success" if met else "info")
                back = request.args.get("back")
                if back == "day":
                    return redirect(url_for("main.day", iso=form.log_date.data.isoformat()))
                if back == "history":
                    return redirect(url_for("main.history"))
                return redirect(url_for("main.index"))

    ctx = _log_page_context(habit, choice, chosen, existing_value)
    return render_template("habit_form.html", form=form, habit=habit, **ctx)


@main_bp.route("/habit/<key>/toggle", methods=["POST"])
@login_required
def toggle_habit(key):
    habit = BY_KEY.get(key)
    if habit is None or habit.get("type") != "check":
        abort(404)
    if habit["key"] not in tracking_for(current_user)[0]:
        flash(f"You are not tracking {UI_TITLES.get(habit['field'], habit['name'])}. Turn it on first.", "info")
        return redirect(url_for("main.setup"))

    day = date.today()
    raw = request.form.get("date")
    if raw:
        try:
            day = min(date.fromisoformat(raw), date.today())     # never the future
        except ValueError:
            day = date.today()
    log = _get_log(current_user.id, day, create=True)
    current = getattr(log, habit["field"])
    setattr(log, habit["field"], None if current else True)
    db.session.commit()
    if request.form.get("next") == "history":
        return redirect(url_for("main.history"))
    if request.form.get("next") == "day":
        return redirect(url_for("main.day", iso=day.isoformat()))
    return redirect(url_for("main.index"))



# ------------------------------------------------------- history and export
HISTORY_PER_PAGE = 30


def _display(habit, value):
    """Full text for one stored value (shown as the hover tooltip)."""
    if value is None:
        return "-"
    if habit.get("type") == "check":
        return "Done" if value else "Not done"
    if _is_choice(habit):
        return _choice_label(habit, value) or str(value)
    return f"{value:g} {habit.get('unit', '')}".strip()


SHORT_TEXT = {"class": "Class", "free": "Free", "deadline": "Deadline",
              "sunny": "Sunny", "cloudy": "Cloudy", "rainy": "Rainy"}


def _short(habit, value):
    """Compact text for the history table cell."""
    if habit.get("key") == "junk":
        return "Yes" if value else "No"
    if habit.get("key") == "difficulty":
        return f"{value}/5"
    if habit.get("key") == "stress":
        return f"{value}/10"
    if habit.get("type") == "check":
        return ""                                   # shown as a tick
    if _is_choice(habit) and str(value) in SHORT_TEXT:
        return SHORT_TEXT[str(value)]
    if habit.get("key") in ("study", "sleep", "free_hours"):
        return f"{value:g}h"
    return f"{value:g}"


def _cell_data(habit, log, is_goal):
    """Everything a template needs to show one habit on one day."""
    value = getattr(log, habit["field"], None) if log else None
    check = habit.get("type") == "check"
    return {
        "habit": habit, "empty": value is None and not check,
        "state": is_met(habit, log) if (is_goal and log) else None,   # True met / False missed / None
        "short": "" if value is None else _short(habit, value),
        "full": _display(habit, value),
        "done": bool(value) if check else None,
    }


@main_bp.route("/history")
@login_required
def history():
    """Every log, newest first. Cells are coloured by whether the goal was met."""
    _, habits, context = tracking_for(current_user)
    goal_keys = {h["key"] for h in habits}
    columns = []
    for i, c in enumerate(habits + context):
        columns.append({
            "habit": c, "field": c["field"], "key": c["key"],
            "title": UI_TITLES.get(c["field"], c["name"]),
            "color": c["color"] if c["key"] in goal_keys
                     else CONTEXT_COLORS[i % len(CONTEXT_COLORS)],
        })
    page = ProductivityLog.query.filter_by(user_id=current_user.id).order_by(
        ProductivityLog.log_date.desc()).paginate(
        page=request.args.get("page", 1, type=int), per_page=HISTORY_PER_PAGE, error_out=False)
    today = date.today()
    notes = notes_for(current_user.id, [l.log_date for l in page.items])
    rows = []
    for log in page.items:
        met = sum(1 for h in habits if is_met(h, log) is True)
        cells = [_cell_data(col["habit"], log, col["key"] in goal_keys) for col in columns]
        pct = round(met / len(habits) * 100) if habits else 0
        rows.append({"log": log, "cells": cells, "met": met, "pct": pct,
                     "note": notes.get(log.log_date),
                     "is_today": log.log_date == today,
                     "show_year": log.log_date.year != today.year})
    return render_template("history.html", rows=rows, columns=columns, page=page,
                           n_goals=len(habits))


LEVEL_TEXT = {0: "Nothing logged", 1: "A start", 2: "Getting there", 3: "Strong day", 4: "Full day"}


@main_bp.route("/day/<iso>")
@login_required
def day(iso):
    """One day in detail: every habit's value, goal state and a link to edit it."""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        abort(404)
    today = date.today()
    if d > today:                                   # nothing to show in the future
        return redirect(url_for("main.day", iso=today.isoformat()))

    _, habits, context = tracking_for(current_user)
    log = _get_log(current_user.id, d)
    goals = [dict(_cell_data(h, log, True), title=UI_TITLES.get(h["field"], h["name"])) for h in habits]
    extras = [dict(_cell_data(c, log, False), title=UI_TITLES.get(c["field"], c["name"]),
                   color=CONTEXT_COLORS[i % len(CONTEXT_COLORS)])
              for i, c in enumerate(context)]
    met = sum(1 for g in goals if g["state"] is True)
    lvl = _level(log, habits) if log else 0
    return render_template(
        "day.html", d=d, log=log, goals=goals, extras=extras, met=met, n_goals=len(habits),
        pct=round(met / len(habits) * 100) if habits else 0,
        level=lvl, level_text=LEVEL_TEXT[lvl],
        is_today=d == today,
        note=get_note(current_user.id, d), note_max=NOTE_MAX,
        prev_iso=(d - timedelta(days=1)).isoformat(),
        next_iso=(d + timedelta(days=1)).isoformat() if d < today else None,
        iso=d.isoformat())


@main_bp.route("/day/<iso>/note", methods=["POST"])
@login_required
def save_day_note(iso):
    """Save, replace or clear the optional note for one day (empty text clears it)."""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        abort(404)
    if d > date.today():
        abort(404)
    try:
        result = save_note(current_user.id, d, request.form.get("note", ""))
        flash("Note saved." if result == "saved" else "Note removed.", "success")
    except ValueError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("main.day", iso=d.isoformat()))


@main_bp.route("/history/<int:log_id>/delete", methods=["POST"])
@login_required
def delete_log(log_id):
    log = ProductivityLog.query.filter_by(id=log_id, user_id=current_user.id).first_or_404()
    day = log.log_date.isoformat()
    delete_note(current_user.id, log.log_date)          # the day's note goes with it
    db.session.delete(log)
    db.session.commit()
    flash(f"Deleted the log for {day}.", "info")
    return redirect(url_for("main.history"))


@main_bp.route("/export.csv")
@login_required
def export_csv():
    """All of the current user's logs as CSV (their own rows only)."""
    habits = HABITS + CONTEXT
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["date"] + [h["field"] for h in habits])
    logs = ProductivityLog.query.filter_by(user_id=current_user.id).order_by(
        ProductivityLog.log_date).all()
    for log in logs:
        row = [log.log_date.isoformat()]
        for h in habits:
            v = getattr(log, h["field"], None)
            row.append("" if v is None else int(v) if isinstance(v, bool) else v)
        writer.writerow(row)
    return Response(out.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=hibi-logs.csv"})


# ------------------------------------------------------------------ import
def _rule_text(h):
    """Allowed values of one column, built from the habit definition."""
    if h.get("type") == "check":
        return "1 or 0 (true / false)"
    if h.get("type") == "choice":
        return ", ".join(str(value) for value, _ in h["choices"])
    kind = "whole numbers" if h["cast"] is int else "numbers"
    return f"{h['min']:g} to {h['max']:g} ({kind})"


def _plan_import(user_id, rows, overwrite):
    """Work out what an import would change, without touching the database.

    New days are created. For an existing day, empty values are filled in; a
    value that is already there is only replaced when `overwrite` is on.
    """
    existing = {}
    if rows:
        for log in ProductivityLog.query.filter(
                ProductivityLog.user_id == user_id,
                ProductivityLog.log_date >= min(rows),
                ProductivityLog.log_date <= max(rows)):
            existing[log.log_date] = log
    plan = {"new": 0, "updated": 0, "unchanged": 0, "values_set": 0,
            "values_kept": 0, "changes": []}
    for day, values in sorted(rows.items()):
        log = existing.get(day)
        if log is None:
            plan["new"] += 1
            plan["values_set"] += len(values)
            plan["changes"].append((day, None, values))
            continue
        changes = {}
        for name, value in values.items():
            current = getattr(log, name)
            if current is None or (overwrite and current != value):
                changes[name] = value
            elif current != value:
                plan["values_kept"] += 1            # kept your existing value
        if changes:
            plan["updated"] += 1
            plan["values_set"] += len(changes)
            plan["changes"].append((day, log, changes))
        else:
            plan["unchanged"] += 1
    return plan


@main_bp.route("/import", methods=["GET", "POST"])
@login_required
def import_csv():
    """Upload a CSV in the export format. All or nothing: any error imports nothing."""
    outcome = None
    if request.method == "POST":
        upload = request.files.get("file")
        overwrite = request.form.get("mode") == "overwrite"
        dry_run = request.form.get("dry_run") == "1"
        if not upload or not upload.filename:
            outcome = {"ok": False, "errors": ["Choose a CSV file first."], "error_total": 1}
        else:
            parsed = parse_csv(upload.stream.read(IMPORT_MAX_BYTES + 1))
            if parsed.error_total:
                outcome = {"ok": False, "errors": parsed.errors, "error_total": parsed.error_total}
            else:
                plan = _plan_import(current_user.id, parsed.rows, overwrite)
                saved = False
                if not dry_run:
                    try:
                        for day, log, changes in plan["changes"]:
                            if log is None:
                                db.session.add(ProductivityLog(
                                    user_id=current_user.id, log_date=day, **changes))
                            else:
                                for name, value in changes.items():
                                    setattr(log, name, value)
                        db.session.commit()
                        saved = True
                    except SQLAlchemyError:
                        db.session.rollback()
                        plan = None
                if plan is None:
                    outcome = {"ok": False, "error_total": 1, "errors": [
                        "The database refused the import, so nothing was saved. Try again."]}
                else:
                    outcome = {"ok": True, "saved": saved, "dry_run": dry_run,
                               "overwrite": overwrite, "blank": parsed.blank_rows,
                               **{k: v for k, v in plan.items() if k != "changes"}}
    columns = [{"field": f, "name": h.get("label") or h["name"], "rule": _rule_text(h)}
               for f, h in IMPORT_FIELDS.items()]
    return render_template("import.html", outcome=outcome, columns=columns,
                           max_rows=IMPORT_MAX_ROWS, max_mb=IMPORT_MAX_BYTES // 1_000_000)


# ------------------------------------------------------------- profile photo
# Per user, in instance/avatars/:
#   <id>.src.jpg   the uploaded photo, re-encoded (<= 1280 px), kept so the
#                  user can re-adjust later without uploading again
#   <id>.json      the framing: centre (cx, cy) and zoom, as fractions
#   <id>.jpg       the 256 px square avatar the site shows
# Uploads are never stored as sent: Pillow decodes them, which also strips
# metadata and anything hidden in the file. The browser only sends the framing
# numbers; the server does the cropping.
AVATAR_MAX_BYTES = 3 * 1024 * 1024
AVATAR_SIZE = 256
AVATAR_SOURCE_MAX = 1280
AVATAR_FORMATS = {"JPEG", "PNG", "WEBP"}
AVATAR_MAX_PIXELS = 25_000_000        # refuse decompression bombs


def _avatar_dir() -> Path:
    return Path(current_app.instance_path) / "avatars"


def _avatar_file(user_id: int, kind: str = "jpg") -> Path:
    suffix = {"jpg": ".jpg", "src": ".src.jpg", "json": ".json"}[kind]
    return _avatar_dir() / f"{int(user_id)}{suffix}"


def _read_crop(user_id: int) -> dict:
    try:
        return json.loads(_avatar_file(user_id, "json").read_text())
    except (OSError, ValueError):
        return {"cx": 0.5, "cy": 0.5, "size": 1.0}


@main_bp.app_context_processor
def _avatar_helpers():
    """avatar_version(user) -> int or None; avatar_crop(user) -> framing dict."""
    def avatar_version(user):
        try:
            return int(_avatar_file(user.id).stat().st_mtime)
        except (OSError, AttributeError, TypeError):
            return None

    def avatar_crop(user):
        return _read_crop(user.id)
    return {"avatar_version": avatar_version, "avatar_crop": avatar_crop}


def _serve_avatar(user_id, kind):
    if user_id != current_user.id:                    # photos are private for now
        abort(404)
    path = _avatar_file(user_id, kind)
    if not path.exists():
        abort(404)
    resp = send_from_directory(path.parent, path.name, mimetype="image/jpeg",
                               conditional=True, max_age=3600)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@main_bp.route("/avatar/<int:user_id>")
@login_required
def avatar(user_id):
    return _serve_avatar(user_id, "jpg")


@main_bp.route("/avatar/<int:user_id>/source")
@login_required
def avatar_source(user_id):
    return _serve_avatar(user_id, "src")


def _crop_params(form) -> dict:
    """Framing from the form, clamped. Anything missing or odd means 'centred'."""
    def num(name, default, lo, hi):
        try:
            value = float(form.get(name, default))
        except (TypeError, ValueError):
            return default
        return min(hi, max(lo, value)) if math.isfinite(value) else default
    return {"cx": num("cx", 0.5, 0.0, 1.0),
            "cy": num("cy", 0.5, 0.0, 1.0),
            "size": num("size", 1.0, 0.05, 1.0)}


def _load_source(raw: bytes):
    """Validate an uploaded image; return an RGB, upright, <=1280 px PIL image."""
    from io import BytesIO
    from PIL import Image, ImageOps, UnidentifiedImageError

    Image.MAX_IMAGE_PIXELS = AVATAR_MAX_PIXELS
    try:
        img = Image.open(BytesIO(raw))
        if img.format not in AVATAR_FORMATS:
            raise ValueError("Use a JPG, PNG or WebP image.")
        if img.width * img.height > AVATAR_MAX_PIXELS:   # Pillow only warns below 2x its limit
            raise ValueError("That image is too large. Use one under about 25 megapixels.")
        img.load()                                     # fully decode (catches truncated files)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        raise ValueError("That file isn't a valid image.")
    img = ImageOps.exif_transpose(img)                 # respect phone-camera rotation
    if img.mode in ("RGBA", "LA", "P"):                # flatten transparency onto white
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")
    img.thumbnail((AVATAR_SOURCE_MAX, AVATAR_SOURCE_MAX), Image.LANCZOS)
    if min(img.size) < 32:
        raise ValueError("That image is too small. Use one at least 32 px wide and tall.")
    return img


def _jpeg(img, quality) -> bytes:
    from io import BytesIO
    out = BytesIO()
    img.save(out, "JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _crop_avatar(img, crop: dict):
    """Square crop around (cx, cy), `size` of the shorter side, as a 256 px image."""
    from PIL import Image
    w, h = img.size
    side = max(8, round(crop["size"] * min(w, h)))
    left = max(0, min(round(crop["cx"] * w - side / 2), w - side))
    top = max(0, min(round(crop["cy"] * h - side / 2), h - side))
    return img.crop((left, top, left + side, top + side)).resize(
        (AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)                              # never a half-written file


def _save_avatar(user_id: int, source, crop: dict) -> None:
    _avatar_dir().mkdir(parents=True, exist_ok=True)
    _write_atomic(_avatar_file(user_id, "src"), _jpeg(source, 90))
    _write_atomic(_avatar_file(user_id, "json"), json.dumps(crop).encode())
    _write_atomic(_avatar_file(user_id), _jpeg(_crop_avatar(source, crop), 88))


@main_bp.route("/profile/avatar", methods=["POST"])
@login_required
def upload_avatar():
    file = request.files.get("avatar")
    if file is None or not file.filename:
        flash("Choose a photo first.", "danger")
        return redirect(url_for("main.profile"))
    raw = file.stream.read(AVATAR_MAX_BYTES + 1)
    if len(raw) > AVATAR_MAX_BYTES:
        flash("That photo is too big. The limit is 3 MB.", "danger")
        return redirect(url_for("main.profile"))
    try:
        source = _load_source(raw)
    except ImportError:
        flash("Photo upload needs Pillow: run  pip install pillow", "danger")
        return redirect(url_for("main.profile"))
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("main.profile"))

    _save_avatar(current_user.id, source, _crop_params(request.form))
    flash("Profile photo updated.", "success")
    return redirect(url_for("main.profile"))


@main_bp.route("/profile/avatar/adjust", methods=["POST"])
@login_required
def adjust_avatar():
    """Re-frame the photo already uploaded (no new upload needed)."""
    try:
        from PIL import Image
        with Image.open(_avatar_file(current_user.id, "src")) as im:
            source = im.convert("RGB")
    except (OSError, ImportError):
        flash("There is no photo to adjust yet. Upload one first.", "danger")
        return redirect(url_for("main.profile"))
    _save_avatar(current_user.id, source, _crop_params(request.form))
    flash("Photo position saved.", "success")
    return redirect(url_for("main.profile"))


@main_bp.route("/profile/avatar/remove", methods=["POST"])
@login_required
def remove_avatar():
    removed = False
    for kind in ("jpg", "src", "json"):
        try:
            _avatar_file(current_user.id, kind).unlink()
            removed = True
        except FileNotFoundError:
            pass
    if removed:
        flash("Profile photo removed.", "info")
    return redirect(url_for("main.profile"))

@main_bp.route("/profile/theme", methods=["POST"])
@login_required
def update_theme():
    selected_theme = request.form.get("theme", "").strip()

    if selected_theme not in THEMES:
        flash("Please select a valid theme.", "danger")
        return redirect(url_for("main.profile"))

    current_user.theme = selected_theme
    db.session.commit()

    flash(f"Theme updated to {THEMES[selected_theme]['name']}!", "success")
    return redirect(url_for("main.profile"))


@main_bp.route("/profile")
@login_required
def profile():
    try:
        stats = dashboard_stats(current_user, tracking_for(current_user)[1]) or {}
    except Exception:                       # the profile must open even if stats break
        current_app.logger.exception("profile stats failed")
        stats = {}
    habits = [] if stats.get("empty") else (stats.get("habits") or [])

    first = (ProductivityLog.query.filter_by(user_id=current_user.id)
             .order_by(ProductivityLog.log_date.asc()).first())
    total = ProductivityLog.query.filter_by(user_id=current_user.id).count()
    return render_template(
        "profile.html",
        summary=stats.get("summary") or {},
        habits=habits,
        top_habit=max(habits, key=lambda h: h["met_rate"], default=None),
        current_streak=max((h["current_streak"] for h in habits), default=0),
        best_streak=max((h["best_streak"] for h in habits), default=0),
        total_logs=total,
        first_log=first.log_date if first else None,
        member_since=getattr(current_user, "created_at", None),
        week_percent=_week_percent(current_user),
        tracking=[c for _, _, cards in _setup_cards(tracking_for(current_user)[0]) for c in cards if c["on"]],
        n_options=len(_all_keys()),
        estimate=latest_estimate(current_user.id),
        themes=THEMES,
        current_theme=current_user.theme or "midnight-purple",
    )
