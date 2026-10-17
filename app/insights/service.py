"""Small helper so other pages (like the home page) can show an ML estimate.

    from app.insights.service import latest_estimate
    data["estimate"] = latest_estimate(current_user.id)

Returns one of:
    None                                  models not trained, or something broke
                                          (the template then hides the card)
    {"status": "no_log"}                  the user has not logged anything yet
    {"status": "incomplete", "message"}   no recent log has the fields the model needs
    {"status": "ok", "score", "level", "error", "log_id", "log_date"}

How the log is chosen
    Logs are created one habit at a time, so today's row is often half empty.
    This looks at the user's 14 most recent log dates and uses the newest one
    that has every NUMERIC input. A check habit (journaled) that is None means
    "not done" (the toggle sets None on Undo), so it counts as 0.

Design rule: an EXPERIMENTAL feature must never be able to break the home page,
so every failure is caught here and logged instead of raised.
"""
from flask import current_app

from ml.predict import ModelsNotTrainedError, predict_day
from ml.settings import FEATURES

CHECK_FEATURES = ("journaled",)                       # None means "not done"
NUMERIC_FEATURES = [f for f in FEATURES if f not in CHECK_FEATURES]
RECENT_LOGS = 14


def latest_estimate(user_id: int):
    try:
        from app.models import ProductivityLog   # imported here to avoid import cycles

        absent = [f for f in FEATURES if not hasattr(ProductivityLog, f)]
        if absent:
            return {"status": "incomplete",
                    "message": "The log table has no column named: " + ", ".join(absent) + "."}

        logs = (ProductivityLog.query
                .filter_by(user_id=user_id)
                .order_by(ProductivityLog.log_date.desc())
                .limit(RECENT_LOGS)
                .all())
        if not logs:
            return {"status": "no_log"}

        log = next((l for l in logs
                    if all(getattr(l, f) is not None for f in NUMERIC_FEATURES)), None)
        if log is None:
            return {"status": "incomplete",
                    "message": "None of your last " + str(len(logs)) +
                               " logged days has all of: " + ", ".join(NUMERIC_FEATURES) + "."}

        values = {f: getattr(log, f) for f in NUMERIC_FEATURES}
        for f in CHECK_FEATURES:
            values[f] = 1 if getattr(log, f) else 0

        result = predict_day(values)
        d = getattr(log, "log_date", None)
        return {
            "status": "ok",
            "score": result["predicted_score"],
            "level": result["predicted_level"],
            "error": result["typical_error_points"],
            "log_id": log.id,
            "log_date": d.isoformat() if hasattr(d, "isoformat") else (str(d) if d else None),
        }
    except ModelsNotTrainedError:
        return None
    except ValueError as exc:                    # value outside the model's valid range
        return {"status": "incomplete", "message": str(exc)}
    except Exception:                            # never break the home page
        current_app.logger.exception("ML estimate failed")
        return None
