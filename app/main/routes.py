import os
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from flask import (Blueprint, abort, current_app, flash, jsonify, redirect,
                   render_template, request, send_from_directory, url_for)
from flask_login import current_user, login_required
from sqlalchemy import extract
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
def heatmap_years(user):
    """Calendar years in which the user has any log, newest first (this year always included)."""
    rows = (db.session.query(extract("year", ProductivityLog.log_date))
            .filter(ProductivityLog.user_id == user.id).distinct().all())
    years = {int(r[0]) for r in rows if r[0] is not None}
    years.add(date.today().year)
    return sorted(years, reverse=True)


def _streaks(by_date, range_start, range_end, today):
    """(current streak, longest streak in the shown period), in consecutive logged days.

    The current streak ends today, or yesterday if today is not logged yet, so it
    does not drop to 0 every morning before you log.
    """
    logged = {dt for dt, l in by_date.items() if level(l) > 0}
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


def build_dashboard(user, year=None):
    """Data for the home page. `year` picks a calendar year for the heatmap;
    None means the rolling last 12 months (the past 52 weeks)."""
    today = date.today()
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
        first = next((c for c in w if c), None)   # a week can begin with blank cells
        month = date.fromisoformat(first["date"]).strftime("%b") if first else ""
        labels.append(month if month and month != prev else "")
        if month:
            prev = month

    heat_count = sum(1 for dt, l in by_date.items()
                     if range_start <= dt <= range_end and level(l) > 0)
    streak_current, streak_longest = _streaks(by_date, range_start, range_end, today)
    days_in_range = (range_end - range_start).days + 1
    heat_options = [{"label": "Last 12 months", "year": None}] + [
        {"label": str(y), "year": y} for y in years]

    # This week, Monday to Sunday
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
        "total_logs": sum(1 for dt, l in by_date.items()
                          if roll_start <= dt <= today and level(l) > 0),   # last 12 months
        "heat_year": year,
        "streak_current": streak_current,
        "streak_longest": streak_longest,
        "heat_pct": round(heat_count / days_in_range * 100),
        "heat_title": str(year) if year else "Last 12 months",
        "heat_count": heat_count,
        "heat_options": heat_options,
        "show_year_picker": any(y < today.year for y in years),     # only once there is an earlier year
        "heat_scroll_end": year is None or year == today.year,
    }


@main_bp.route("/")
def index():
    data = None
    if current_user.is_authenticated:
        data = build_dashboard(current_user, request.args.get("year", type=int))
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



# ---------------------------------------------------------- log-page helpers
# Quick-pick values shown as chips on the log page, by habit key. Anything not
# listed falls back to the full grid when it is short (<= 12 values).
PRESETS = {
    "study": [0, 1, 2, 3, 4, 5, 6],
    "sleep": [5, 6, 6.5, 7, 7.5, 8, 9],
    "disruptions": [0, 1, 2, 3, 4, 5],
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
                return redirect(url_for("main.index"))

    ctx = _log_page_context(habit, choice, chosen, existing_value)
    return render_template("habit_form.html", form=form, habit=habit, **ctx)


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



# ------------------------------------------------------------- profile photo
# Per user, in instance/avatars/:
#   <id>.src.jpg   the uploaded photo, re-encoded (<= 1280 px), kept so the
#                  user can re-adjust later without uploading again
#   <id>.json      the framing: centre (cx, cy) and zoom, as fractions
#   <id>.jpg       the 256 px square avatar the site shows
# Uploads are never stored as sent: Pillow decodes them, which also strips
# metadata and anything hidden in the file. The browser only sends the framing
# numbers; the server does the cropping.
import json
import math

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
    from PIL import Image
    try:
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

@main_bp.route("/profile")
@login_required
def profile():
    try:
        stats = dashboard_stats(current_user) or {}
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
        week_percent=build_dashboard(current_user)["percent"],
        estimate=latest_estimate(current_user.id),
    )
