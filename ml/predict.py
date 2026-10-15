"""Load the saved models and make predictions.

Flask-free on purpose. From the web app, later:

    from ml.predict import predict_day
    result = predict_day({"sleep_hours": 7, ...})

Quick test from the project root:
    python -m ml.predict
"""
import json

import joblib
import pandas as pd

from ml.settings import ARTIFACT_DIR, FEATURE_RANGES, FEATURES, LEVELS

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

    return {
        "predicted_score": score,
        "predicted_level": LEVELS[class_id],
        "level_probabilities": {
            LEVELS[int(c)]: round(float(p), 3)
            for c, p in zip(clf.classes_, probs)
        },
        "cluster": cluster,
        "cluster_profile": models["profiles"].loc[cluster].to_dict(),
    }


def predict_for_log(log) -> dict:
    """Predict from a ProductivityLog row (or any object with those attributes)."""
    values = {name: getattr(log, name, None) for name in FEATURES}
    missing = [name for name, v in values.items() if v is None]
    if missing:
        raise ValueError("This log is missing: " + ", ".join(missing))
    values = {name: int(v) if isinstance(v, bool) else v
              for name, v in values.items()}
    return predict_day(values)


def data_stage(n_logs: int) -> str:
    """How much of the user's own data exists decides what the app can claim."""
    if n_logs < 7:
        return "general"       # estimate from the synthetic-trained model
    if n_logs < 20:
        return "descriptive"   # show averages, no personal model yet
    return "personal"          # enough rows to consider a per-user model


def model_info() -> dict:
    """Metrics and feature importance, for an ML Insights page."""
    return _load()["metadata"]


if __name__ == "__main__":
    example = {"study_hours": 4, "caffeine_mg": 3, "stress_level": 4,
               "study_sessions": 3, "journaled": 1}
    print(json.dumps(predict_day(example), indent=2))