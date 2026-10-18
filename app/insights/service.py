"""Small helper so other pages (like the home page) can show an ML estimate.

    from app.insights.service import latest_estimate
    data["estimate"] = latest_estimate(current_user.id)

Returns one of:
    None                                  models not trained, or something broke
                                          (the template then hides the card)
    {"status": "no_log"}                  no log in the last month
    {"status": "incomplete", "message"}   a log exists but inputs are missing;
                                          the message names them
    {"status": "ok", "score", "level", "error", "log_id", "log_date",
                 plus what the home card shows: "range_lo"/"range_hi" (score +-
                 typical error), "probs" (chance of each level), "low_below"/
                 "high_from" (level cut-offs), "levers" (biggest what-ifs),
                 "source" ("today's log", "yesterday's log", ...)}

How the inputs are built
    The model's inputs are named like the DATASET (ml/settings.py FEATURES),
    but the app stores them under different names, so DB_FIELDS maps one to the
    other (deep_work lives in study_hours, caffeine_level in caffeine_mg, ...).
    Two inputs are not stored at all; they are computed from the days before
    the one being estimated (app.habits):
        log_streak            consecutive logged days ending the day before
        deep_work_days_prev7  how many of the previous 7 days met the goal

How the log is chosen
    Logs are created one habit at a time, so today's row is often half empty.
    This looks at the newest 14 logs (last ~month) and uses the newest one that
    has every input. If none is complete, the message names what the NEWEST
    log is missing, so the user knows what to log next. A check habit
    (journaled) that is None means "not done", so it counts as 0. A dropdown
    habit that is None (junk food, day type, weather, ...) is really missing.

Design rule: an EXPERIMENTAL feature must never be able to break the home page,
so every failure is caught here and logged instead of raised.
"""
from datetime import date, timedelta

from flask import current_app

from ml.predict import ModelsNotTrainedError, predict_day, suggest_levers
from ml.settings import FEATURES, LEVELS, LEVEL_BINS

# dataset feature name -> ProductivityLog column
DB_FIELDS = {
    "deep_work": "study_hours",
    "sleep_hours": "sleep_hours",
    "caffeine_level": "caffeine_mg",       # stores the 0-10 LEVEL, not mg
    "stress_level": "stress_level",
    "journaled": "journaled",
    "disruptions": "disruptions",
    "junk_food": "junk_food",
    "task_difficulty": "task_difficulty",
    "day_type": "day_type",
    "weather": "weather",
    "free_hours": "free_hours",
}
COMPUTED = ("log_streak", "deep_work_days_prev7")
CHECK_FEATURES = ("journaled",)            # None means "not done"

LABELS = {
    "deep_work": "Deep work", "sleep_hours": "Sleep",
    "caffeine_level": "Caffeine", "stress_level": "Stress",
    "disruptions": "Disruptions", "junk_food": "Junk food",
    "task_difficulty": "Task difficulty", "day_type": "Day type",
    "weather": "Weather", "free_hours": "Free hours",
}

RECENT_LOGS = 14
HISTORY_DAYS = 32      # enough to compute a 14-day streak for a 14-day-old log


def _inputs(log, by_date, log_streak, deep_work_days_prev7):
    """Return (values, missing) for one log row."""
    values, missing = {}, []
    for f in FEATURES:
        if f == "log_streak":
            values[f] = log_streak(by_date, log.log_date)
        elif f == "deep_work_days_prev7":
            values[f] = deep_work_days_prev7(by_date, log.log_date)
        else:
            v = getattr(log, DB_FIELDS[f], None)
            if f in CHECK_FEATURES:
                v = 1 if v else 0
            if v is None:
                missing.append(f)
                continue
            values[f] = int(v) if isinstance(v, bool) else v
    return values, missing


def _missing_message(log, missing):
    names = ", ".join(LABELS.get(f, f) for f in missing)
    if log.log_date == date.today():
        return f"To estimate today, still log: {names}."
    return (f"Your latest log ({log.log_date.isoformat()}) is missing: "
            f"{names}. Log them to get an estimate.")


def _source(d):
    today = date.today()
    if d == today:
        return "today's log"
    if d == today - timedelta(days=1):
        return "yesterday's log"
    return f"your log from {d.isoformat()}"


def _ok(log, values, result):
    """Everything the home-page card shows for a complete day."""
    score = result["predicted_score"]
    err = result["typical_error_points"]
    d = log.log_date
    try:
        levers = suggest_levers(values)
    except Exception:                       # the extras must never break the card
        current_app.logger.exception("ML levers failed")
        levers = []
    probs = result.get("level_probabilities", {})
    return {
        "status": "ok",
        "score": score,
        "level": result["predicted_level"],
        "error": round(err, 1) if err else None,
        "range_lo": round(max(0.0, score - err), 1) if err else None,
        "range_hi": round(min(100.0, score + err), 1) if err else None,
        "probs": [{"level": lvl, "pct": round(probs.get(lvl, 0) * 100)}
                  for lvl in LEVELS],
        "low_below": LEVEL_BINS[1],
        "high_from": LEVEL_BINS[2],
        "levers": levers,
        "source": _source(d),
        "log_id": log.id,
        "log_date": d.isoformat() if hasattr(d, "isoformat") else str(d),
    }


def latest_estimate(user_id: int):
    try:
        from app.habits import deep_work_days_prev7, log_streak
        from app.models import ProductivityLog   # here to avoid import cycles

        unknown = [f for f in FEATURES
                   if f not in DB_FIELDS and f not in COMPUTED]
        if unknown:
            return {"status": "incomplete",
                    "message": "The app does not know how to build: "
                               + ", ".join(unknown) + " (see DB_FIELDS in "
                               "app/insights/service.py)."}
        absent = sorted({DB_FIELDS[f] for f in FEATURES if f in DB_FIELDS
                         and not hasattr(ProductivityLog, DB_FIELDS[f])})
        if absent:
            return {"status": "incomplete",
                    "message": "The log table has no column named: "
                               + ", ".join(absent) + "."}

        since = date.today() - timedelta(days=HISTORY_DAYS)
        rows = (ProductivityLog.query
                .filter(ProductivityLog.user_id == user_id,
                        ProductivityLog.log_date >= since)
                .order_by(ProductivityLog.log_date.desc())
                .all())
        if not rows:
            return {"status": "no_log"}
        by_date = {r.log_date: r for r in rows}

        newest_missing = None
        for log in rows[:RECENT_LOGS]:
            values, missing = _inputs(log, by_date, log_streak,
                                      deep_work_days_prev7)
            if missing:
                if newest_missing is None:
                    newest_missing = (log, missing)
                continue
            result = predict_day(values)
            return _ok(log, values, result)

        log, missing = newest_missing
        return {"status": "incomplete", "message": _missing_message(log, missing)}
    except ModelsNotTrainedError:
        return None
    except ValueError as exc:                    # value outside the model's range
        return {"status": "incomplete", "message": str(exc)}
    except Exception:                            # never break the home page
        current_app.logger.exception("ML estimate failed")
        return None
