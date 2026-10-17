"""Load the saved models and make predictions.

Flask-free on purpose. From the web app:

    from ml.predict import predict_day
    result = predict_day({"sleep_hours": 7, ...})

Quick test from the project root:
    python -m ml.predict

SECURITY: the .joblib files are pickles. Only load files this project trained.
"""
import json

import joblib
import pandas as pd

from ml.settings import ARTIFACT_DIR, FEATURE_RANGES, FEATURES, LEVELS

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


def _validate(values: dict) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    row = {}
    for name in FEATURES:
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
        low, high = FEATURE_RANGES[name]
        # NaN fails this comparison too, so it is rejected as well
        if not low <= value <= high:
            raise ValueError(f"'{name}' must be between {low} and {high}")
        row[name] = value
    return pd.DataFrame([row], columns=FEATURES)  # fixed column order


def predict_day(values: dict) -> dict:
    """Predict score, level and behaviour cluster for one day."""
    models = _load()
    X = _validate(values)

    score = float(models["regressor"].predict(X)[0])
    score = round(min(max(score, 0.0), 100.0), 1)

    clf = models["classifier"]
    class_id = int(clf.predict(X)[0])
    probs = clf.predict_proba(X)[0]
    cluster = int(models["kmeans"].predict(X)[0])

    meta = models["metadata"]
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


def predict_for_log(log) -> dict:
    """Predict from a ProductivityLog row (or any object with those attributes).

    NOTE: the model expects caffeine on the 0-10 scale. If your database stores
    real milligrams, convert here (mg / 60) before calling predict_day.
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
    example = {"study_hours": 4, "caffeine_mg": 3, "stress_level": 4,
               "study_sessions": 3, "journaled": 1}
    print(json.dumps(predict_day(example), indent=2))
