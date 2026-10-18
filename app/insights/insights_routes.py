"""ML Insights pages.

    /insights/            "Try your own numbers": type a day, see the estimate
    /insights/log/<id>    "Details": the estimate for one of YOUR logged days

Both use the same 13 model inputs as the home-page card (ml/settings.py
FEATURES). Nothing here trains or changes a model; it only calls predict_day.
"""
from datetime import timedelta

from flask import abort, flash, render_template, request
from flask_login import current_user, login_required

# Same Blueprint object that app/insights/__init__.py creates and registers.
# (Aliased to `insights`, so both `from app.insights import bp` and
# `from app.insights.routes import insights` keep working.)
from app.insights import bp as insights
from app.insights.service import HISTORY_DAYS, LABELS, _inputs
from app.models import ProductivityLog
from ml.predict import (ModelsNotTrainedError, data_stage, model_info,
                        predict_day)
from ml.settings import FEATURES

# label, min, max, step, default for the "try your own numbers" form.
# Only used for the form's hints; predict_day does the real validation.
NUMBER_SPEC = {
    "deep_work": ("Deep work (hours)", 0, 24, 0.5, 3),
    "sleep_hours": ("Sleep (hours)", 0, 24, 0.5, 7),
    "caffeine_level": ("Caffeine level (0-10)", 0, 10, 1, 2),
    "stress_level": ("Stress (1-10)", 1, 10, 1, 5),
    "log_streak": ("Days logged in a row (0-14)", 0, 14, 1, 3),
    "disruptions": ("Disruptions", 0, 20, 1, 1),
    "task_difficulty": ("Task difficulty (1-5)", 1, 5, 1, 3),
    "deep_work_days_prev7": ("Deep-work goal days, last 7 (0-7)", 0, 7, 1, 3),
    "free_hours": ("Free hours", 0, 24, 0.5, 5),
}
CHECK_SPEC = {"journaled": "Journaled today", "junk_food": "Had junk food"}
SELECT_SPEC = {"day_type": "Day type", "weather": "Weather"}

# Friendly names for the details page (service.LABELS lacks the last three)
NAMES = {**LABELS, "journaled": "Journaling", "log_streak": "Days logged in a row",
         "deep_work_days_prev7": "Deep-work goal days (last 7)"}


def _pretty(name):
    return NAMES.get(name) or name.replace("_", " ").capitalize()


def _categories():
    """(categories dict, models_trained). Categories come from training."""
    try:
        return model_info().get("categories", {}), True
    except ModelsNotTrainedError:
        return {}, False


def _fields(values, categories):
    """Form fields in FEATURES order, pre-filled from `values`."""
    fields = []
    for name in FEATURES:
        if name in SELECT_SPEC:
            options = categories.get(name, [])
            fields.append({"name": name, "kind": "select",
                           "label": SELECT_SPEC[name], "options": options,
                           "value": values.get(name, options[0] if options else "")})
        elif name in CHECK_SPEC:
            fields.append({"name": name, "kind": "check",
                           "label": CHECK_SPEC[name],
                           "value": bool(values.get(name, 0))})
        else:
            label, lo, hi, step, default = NUMBER_SPEC.get(
                name, (_pretty(name), None, None, "any", 0))
            fields.append({"name": name, "kind": "number", "label": label,
                           "min": lo, "max": hi, "step": step,
                           "value": values.get(name, default)})
    return fields


def _read_form():
    """Raw values from the submitted form (predict_day validates them)."""
    values = {}
    for name in FEATURES:
        if name in CHECK_SPEC:
            values[name] = 1 if request.form.get(name) else 0
        else:
            values[name] = (request.form.get(name) or "").strip()
    return values


def _check_numbers(values):
    """First problem in the typed numbers as (field, message), else None.

    Friendly wording for the form. predict_day re-checks everything too.
    """
    for name in FEATURES:
        if name not in NUMBER_SPEC:
            continue
        label, lo, hi, _step, _default = NUMBER_SPEC[name]
        short = label.split(" (")[0]
        try:
            number = float(values[name])
        except (TypeError, ValueError):
            return name, f"{short}: enter a number."
        if number != number or not lo <= number <= hi:      # NaN fails too
            return name, f"{short} must be between {lo:g} and {hi:g}."
    return None


@insights.route("/", methods=["GET", "POST"])
@login_required
def index():
    categories, trained = _categories()
    values, result, error, error_field = {}, None, None, None

    if request.method == "POST":
        values = _read_form()
        problem = _check_numbers(values)
        if problem:
            error_field, error = problem
        else:
            try:
                result = predict_day(values)
            except ModelsNotTrainedError:
                trained = False
            except ValueError as exc:
                error = str(exc)

    n_logs = ProductivityLog.query.filter_by(user_id=current_user.id).count()
    return render_template(
        "insights/index.html", fields=_fields(values, categories),
        trained=trained, result=result, error=error, error_field=error_field,
        n_logs=n_logs, stage=data_stage(n_logs))


@insights.route("/log/<int:log_id>")
@login_required
def estimate_log(log_id):
    from app.habits import deep_work_days_prev7, log_streak

    # Only the owner may open a log (404 for anyone else's id)
    log = ProductivityLog.query.filter_by(
        id=log_id, user_id=current_user.id).first_or_404()

    since = log.log_date - timedelta(days=HISTORY_DAYS)
    history = ProductivityLog.query.filter(
        ProductivityLog.user_id == current_user.id,
        ProductivityLog.log_date >= since,
        ProductivityLog.log_date <= log.log_date).all()
    by_date = {r.log_date: r for r in history}

    values, missing = _inputs(log, by_date, log_streak, deep_work_days_prev7)
    ctx = {"log": log, "result": None, "message": None,
           "rows": [], "drivers": []}

    if missing:
        ctx["message"] = ("This day can't be estimated yet. Still missing: "
                          + ", ".join(_pretty(f) for f in missing) + ".")
        return render_template("insights/estimate.html", **ctx)

    try:
        ctx["result"] = predict_day(values)
        info = model_info()
    except ModelsNotTrainedError:
        ctx["message"] = "Models not trained yet. Run: python -m ml.train"
        return render_template("insights/estimate.html", **ctx)
    except ValueError as exc:
        ctx["message"] = str(exc)
        return render_template("insights/estimate.html", **ctx)

    ctx["rows"] = [(_pretty(f), values[f]) for f in FEATURES]
    importance = (info.get("feature_importance") or {}).get("regressor") or {}
    ctx["drivers"] = [(_pretty(k), round(v * 100))
                      for k, v in list(importance.items())[:5]]
    return render_template("insights/estimate.html", **ctx)
