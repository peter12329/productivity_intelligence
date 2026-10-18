"""Load the saved models and make predictions.

Flask-free on purpose. From the web app:

    from ml.predict import predict_day
    result = predict_day({"deep_work": 4, "sleep_hours": 7, ..., "day_type": "class"})

predict_day needs ALL of settings.FEATURES. Numbers are range-checked; text
features (day_type, weather) must be one of the values seen in training.
The app-side code that turns a database row into these inputs lives in
app/insights/service.py.

Quick test from the project root:
    python -m ml.predict

SECURITY: the .joblib files are pickles. Only load files this project trained.
"""
import json
import math

import joblib
import pandas as pd

import ml.settings as settings
from ml.settings import ARTIFACT_DIR, FEATURES, LEVELS

# Allowed range of every numeric input. settings.FEATURE_RANGES (if present)
# overrides these, but a feature can no longer slip through unchecked just
# because settings.py is out of date.
DEFAULT_RANGES = {
    "deep_work": (0, 24),
    "sleep_hours": (0, 24),
    "caffeine_level": (0, 10),
    "stress_level": (1, 10),
    "journaled": (0, 1),
    "log_streak": (0, 14),
    "disruptions": (0, 20),
    "junk_food": (0, 1),
    "task_difficulty": (1, 5),
    "deep_work_days_prev7": (0, 7),
    "free_hours": (0, 24),
}
FEATURE_RANGES = {**DEFAULT_RANGES, **getattr(settings, "FEATURE_RANGES", {})}

DISCLAIMER = (
    "Experimental: trained on synthetic (simulated) data. This is an estimate "
    "for the day described by these inputs, not a forecast for tomorrow and "
    "not a statement about you."
)

_cache = {}


class ModelsNotTrainedError(RuntimeError):
    pass


def _load():
    """Load models once, then reuse them for every request."""
    if _cache:
        return _cache
    try:
        _cache["regressor"] = joblib.load(ARTIFACT_DIR / "regressor.joblib")
        _cache["classifier"] = joblib.load(ARTIFACT_DIR / "classifier.joblib")
        _cache["kmeans"] = joblib.load(ARTIFACT_DIR / "kmeans.joblib")
        _cache["profiles"] = pd.read_csv(
            ARTIFACT_DIR / "cluster_profiles.csv", index_col="cluster")
        with open(ARTIFACT_DIR / "metadata.json") as f:
            _cache["metadata"] = json.load(f)
    except FileNotFoundError as exc:
        _cache.clear()
        raise ModelsNotTrainedError(
            "Models not found. Run: python -m ml.train") from exc
    return _cache


def reload_models() -> None:
    """Forget cached models (call after retraining without restarting Flask)."""
    _cache.clear()


def _validate(values: dict, meta: dict) -> pd.DataFrame:
    """Check one day's inputs and return a one-row DataFrame in FEATURES order."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")

    # The models were trained on these exact columns. If settings.FEATURES
    # changed after training, stop instead of predicting from the wrong inputs.
    trained = meta.get("features")
    if trained is not None and list(trained) != list(FEATURES):
        raise ValueError(
            "settings.FEATURES differs from the trained models. "
            "Retrain: python -m ml.train")

    categories = meta.get("categories", {})
    cat = set(meta.get("categorical_features", []))
    row = {}
    for name in FEATURES:
        raw = values[name]
        if name in cat:
            text = "" if raw is None else str(raw)
            allowed = categories.get(name)
            if allowed and text not in allowed:
                raise ValueError(
                    f"'{name}' must be one of: {', '.join(allowed)}")
            row[name] = text
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
        if math.isnan(value):
            raise ValueError(f"'{name}' must be a number")
        bounds = FEATURE_RANGES.get(name)
        if bounds is not None:
            low, high = bounds
            if not low <= value <= high:
                raise ValueError(f"'{name}' must be between {low} and {high}")
        row[name] = value
    return pd.DataFrame([row], columns=FEATURES)  # fixed column order


def predict_day(values: dict) -> dict:
    """Predict score, level and behaviour cluster for one day."""
    models = _load()
    meta = models["metadata"]
    X = _validate(values, meta)

    score = float(models["regressor"].predict(X)[0])
    score = round(min(max(score, 0.0), 100.0), 1)

    clf = models["classifier"]
    class_id = int(clf.predict(X)[0])
    probs = clf.predict_proba(X)[0]
    cluster = int(models["kmeans"].predict(X)[0])

    reg_meta = meta.get("regression") or {}
    # Typical error = cross-validated MAE of the selected model (steadier than
    # a hold-out made of only a few people).
    cv_mae = next((c["cv_mae"] for c in reg_meta.get("candidates", [])
                   if c["model"] == reg_meta.get("selected")), None)
    return {
        "predicted_score": score,
        "predicted_level": LEVELS[class_id],
        "level_probabilities": {
            LEVELS[int(c)]: round(float(p), 3)
            for c, p in zip(clf.classes_, probs)
        },
        "cluster": cluster,
        "cluster_profile": models["profiles"].loc[cluster].to_dict(),
        "typical_error_points": cv_mae,
        "model_version": meta.get("model_version"),
        "experimental": True,
        "disclaimer": DISCLAIMER,
    }


# What-if levers: (feature, value to try, wording). suggest_levers() re-runs the
# regressor with one input changed and reports how much the estimate moves.
# This is what the MODEL does, not proof that the change causes anything.
LEVERS = [
    ("sleep_hours", 7.5, "Sleep about 7.5 h"),
    ("deep_work", 4, "Reach about 4 h of deep work"),
    ("stress_level", 3, "Bring stress down to 3"),
    ("disruptions", 0, "Have no disruptions"),
    ("junk_food", 0, "Skip junk food"),
    ("journaled", 1, "Journal today"),
    ("caffeine_level", 3, "Keep caffeine around level 3"),
]


def suggest_levers(values: dict, top: int = 3, min_gain: float = 0.5) -> list:
    """Biggest single changes (by the model) that would raise the score."""
    models = _load()
    meta = models["metadata"]
    base = _validate(values, meta)

    rows, info = [], []
    for feature, target, text in LEVERS:
        if feature not in FEATURES:
            continue
        try:
            if float(values[feature]) == target:
                continue
        except (TypeError, ValueError, KeyError):
            continue
        rows.append(_validate({**values, feature: target}, meta))
        info.append((feature, text))
    if not rows:
        return []

    preds = models["regressor"].predict(pd.concat([base] + rows, ignore_index=True))
    found = []
    for (feature, text), p in zip(info, preds[1:]):
        gain = float(p - preds[0])
        if gain >= min_gain:
            found.append({"feature": feature, "text": text, "gain": round(gain, 1)})
    return sorted(found, key=lambda d: -d["gain"])[:top]


def predict_for_log(log) -> dict:
    """Predict from any object whose attributes are named like FEATURES.

    This does NOT work on a raw ProductivityLog (its column names differ and
    two features are computed from past days). For the app use
    app.insights.service.latest_estimate instead.
    """
    values = {name: getattr(log, name, None) for name in FEATURES}
    missing = [name for name, v in values.items() if v is None]
    if missing:
        raise ValueError("This log is missing: " + ", ".join(missing))
    values = {name: int(v) if isinstance(v, bool) else v
              for name, v in values.items()}
    return predict_day(values)


def data_stage(n_logs: int) -> str:
    """How much of the user's own data exists decides what the app may claim.

    No stage means "a model trained on you": there is no per-user model.
    """
    if n_logs < 7:
        return "general"            # only the synthetic-trained estimate
    if n_logs < 20:
        return "descriptive"        # also show the user's own averages
    return "enough_for_trends"      # enough rows to show trends from their logs


def model_info() -> dict:
    """Metrics, split and feature importance, for an ML Insights page."""
    return _load()["metadata"]


if __name__ == "__main__":
    example = {
        "deep_work": 4, "sleep_hours": 7.5, "caffeine_level": 3,
        "stress_level": 4, "journaled": 1, "log_streak": 3,
        "disruptions": 1, "junk_food": 0, "task_difficulty": 3,
        "deep_work_days_prev7": 4, "day_type": "class", "weather": "sunny",
        "free_hours": 6,
    }
    print(json.dumps(predict_day(example), indent=2))
