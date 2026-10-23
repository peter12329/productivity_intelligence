"""Train, compare and save the three models.

Run from the project root:
    python -m ml.train

Saves to artifacts/:
    regressor.joblib, classifier.joblib, kmeans.joblib,
    metadata.json, cluster_profiles.csv

FEATURE TYPES
    FEATURES (in ml/settings.py) may mix numbers and text. Any feature whose
    column is not numeric in the data (day_type, weather) is one-hot encoded
    inside each model's pipeline; numbers are scaled for the linear models and
    clustering. Because the encoding is part of the saved pipeline, a
    prediction needs a one-row DataFrame with the FEATURES column names
    (not a plain list or array).

VALIDATION (what changed vs the first version)
    Rows from the same person are strongly related (the score carries momentum
    from the previous day). A shuffled split would put the same person in both
    train and test and flatter the metrics. So when the dataset has a
    person_id column:
        * the hold-out set is made of WHOLE PEOPLE the model never saw
          (GroupShuffleSplit), which also matches the app's cold-start case:
          a brand-new user the model has never met
        * cross-validation uses GroupKFold (same idea inside the training part)
    Without person_id (e.g. real logs later) it falls back to the old
    shuffled/stratified behaviour.

    rows < SMALL_N       : no hold-out, repeated k-fold CV only, no XGBoost
    rows >= SMALL_N      : 80/20 hold-out + 5-fold CV on the training part
    rows >= XGB_MIN_ROWS : XGBoost joins the candidates

The winner is chosen on CROSS-VALIDATION scores only. Hold-out numbers are
reported, never used to choose.
"""
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.settings import (ARTIFACT_DIR, CLASSIFICATION_TARGET, DATASET_PATH,
                         FEATURES, LEVELS, N_CLUSTERS_RANGE, RANDOM_STATE,
                         REGRESSION_TARGET)

try:  # XGBoost is optional; skip it cleanly if it is not installed.
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SMALL_N = 100        # below this: no hold-out, repeated CV only
XGB_MIN_ROWS = 300   # XGBoost only competes at or above this many rows
GROUP_COL = "person_id"
TEST_SIZE = 0.2

DISCLAIMER = ("Trained on synthetic (simulated) data. Metrics show how well the "
              "models recover the rules written in dummydataset.py, not how well "
              "they would predict a real person.")


# Columns that must never be model inputs (they are, or come from, the answer).
LEAKY = {"productivity_score", "productivity_level", "productive_day",
         "previous_day_productivity"}


# ------------------------------------------------------ feature preprocessing
def split_feature_types(df: pd.DataFrame):
    """Return (numeric, categorical) lists, in FEATURES order.

    A feature is categorical when its column is not numeric in the data
    (day_type, weather). Those are one-hot encoded; the rest are numbers.
    """
    cat = [c for c in FEATURES if not is_numeric_dtype(df[c])]
    num = [c for c in FEATURES if c not in cat]
    return num, cat


def make_prep(num, cat, scale: bool) -> ColumnTransformer:
    """Numbers are scaled (linear models / clustering) or passed through
    (trees); categoricals are one-hot encoded. Unknown categories at
    prediction time become all zeros instead of crashing."""
    parts = [("num", StandardScaler() if scale else "passthrough", num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False), cat))
    return ColumnTransformer(parts)


def _original_feature(name: str, cat) -> str:
    """'cat__day_type_class' -> 'day_type', 'num__deep_work' -> 'deep_work'."""
    name = name.split("__", 1)[-1]
    for c in cat:
        if name.startswith(c + "_"):
            return c
    return name


def feature_importance(model, cat):
    """Tree importances summed back to the original features (one-hot
    columns of day_type are added up into a single 'day_type' value)."""
    est = model.named_steps["model"]
    if not hasattr(est, "feature_importances_"):
        return None
    names = model.named_steps["prep"].get_feature_names_out()
    out = {}
    for n, v in zip(names, est.feature_importances_):
        k = _original_feature(n, cat)
        out[k] = out.get(k, 0.0) + float(v)
    return {k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def linear_coefficients(model):
    """Standardised coefficients of a linear regressor (sign = direction)."""
    est = model.named_steps["model"]
    if not hasattr(est, "coef_") or np.ndim(est.coef_) != 1:
        return None
    names = model.named_steps["prep"].get_feature_names_out()
    pairs = {n.split("__", 1)[-1]: round(float(c), 4)
             for n, c in zip(names, est.coef_)}
    return dict(sorted(pairs.items(), key=lambda kv: -abs(kv[1])))


# --------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Generate it first:  python -m ml.dummydataset --rows 10000"
        )
    leaky = sorted(LEAKY & set(FEATURES))
    if leaky:
        raise SystemExit(f"FEATURES must not contain answer columns: {leaky}")
    df = pd.read_csv(DATASET_PATH)
    needed = FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    return df.dropna(subset=needed).reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Return (train_df, test_df). test_df is None in small-data mode.

    Deterministic (fixed RANDOM_STATE) so evaluate.py rebuilds the same split.
    """
    if len(df) < SMALL_N:
        return df, None
    if GROUP_COL in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL]))
        return df.iloc[train_idx].reset_index(drop=True), \
            df.iloc[test_idx].reset_index(drop=True)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df[CLASSIFICATION_TARGET])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_description(train_df, test_df) -> dict:
    grouped = GROUP_COL in train_df.columns
    desc = {
        "method": ("hold-out of whole people (GroupShuffleSplit) + GroupKFold CV"
                   if grouped and test_df is not None else
                   "no hold-out; repeated k-fold CV" if test_df is None else
                   "stratified shuffled hold-out + KFold CV"),
        "train_rows": int(len(train_df)),
        "test_rows": 0 if test_df is None else int(len(test_df)),
    }
    if grouped:
        desc["train_people"] = int(train_df[GROUP_COL].nunique())
        desc["test_people"] = (0 if test_df is None
                               else int(test_df[GROUP_COL].nunique()))
        if test_df is not None and desc["test_people"] < 10:
            desc["warning"] = (
                f"The hold-out contains only {desc['test_people']} people, so "
                "hold-out metrics are noisy (they change a lot with a different "
                "split). Prefer the cross-validated numbers, or generate more "
                "people in dummydataset.py.")
    return desc


def use_xgb(n_rows: int) -> bool:
    return HAS_XGB and n_rows >= XGB_MIN_ROWS


def _groups(train_df):
    return train_df[GROUP_COL] if GROUP_COL in train_df.columns else None


# ---------------------------------------------------------------- regression
def regression_candidates(n_rows: int, num, cat) -> dict:
    small = n_rows < SMALL_N
    candidates = {
        "ridge": Pipeline([("prep", make_prep(num, cat, scale=True)),
                           ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([
            ("prep", make_prep(num, cat, scale=False)),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=4 if small else None,
                min_samples_leaf=5 if small else 3,
                random_state=RANDOM_STATE, n_jobs=-1))]),
    }
    if use_xgb(n_rows):
        candidates["xgboost"] = Pipeline([
            ("prep", make_prep(num, cat, scale=False)),
            ("model", XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.9, random_state=RANDOM_STATE, n_jobs=-1))])
    return candidates


def regression_metrics(y_true, pred) -> dict:
    return {"r2": round(float(r2_score(y_true, pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 3)}


def train_regression(X_train, y_train, X_test=None, y_test=None, groups=None,
                     num=(), cat=()):
    n = len(X_train) + (0 if X_test is None else len(X_test))
    if groups is not None:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
    elif n < SMALL_N:
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    else:
        cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    rows, fitted = [], {}
    for name, model in regression_candidates(n, num, cat).items():
        res = cross_validate(model, X_train, y_train, cv=cv,
                             groups=groups, scoring=scoring)
        model.fit(X_train, y_train)
        row = {"model": name,
               "cv_r2": round(float(res["test_r2"].mean()), 4),
               "cv_mae": round(float(-res["test_mae"].mean()), 3),
               "test_r2": None, "test_mae": None}
        if X_test is not None:
            m = regression_metrics(y_test, model.predict(X_test))
            row["test_r2"], row["test_mae"] = m["r2"], m["mae"]
        rows.append(row)
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("cv_r2", ascending=False)
    best = results.iloc[0]["model"]
    return fitted[best], best, results


# ------------------------------------------------------------ classification
def classification_candidates(n_rows: int, num, cat) -> dict:
    small = n_rows < SMALL_N
    candidates = {
        "logistic_regression": Pipeline([
            ("prep", make_prep(num, cat, scale=True)),
            ("model", LogisticRegression(max_iter=1000))]),
        "random_forest": Pipeline([
            ("prep", make_prep(num, cat, scale=False)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=4 if small else None,
                min_samples_leaf=5 if small else 3,
                random_state=RANDOM_STATE, n_jobs=-1))]),
    }
    if use_xgb(n_rows):
        candidates["xgboost"] = Pipeline([
            ("prep", make_prep(num, cat, scale=False)),
            ("model", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.9, random_state=RANDOM_STATE,
                eval_metric="mlogloss", n_jobs=-1))])
    return candidates


def train_classification(X_train, y_train, X_test=None, y_test=None,
                         groups=None, num=(), cat=()):
    n = len(X_train) + (0 if X_test is None else len(X_test))
    min_class = int(pd.Series(y_train).value_counts().min())
    if min_class < 2:
        raise SystemExit(
            "A level has fewer than 2 rows; cannot cross-validate the "
            "classifier. Log more days or merge levels.")
    if groups is not None:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
    else:
        n_splits = min(5, min_class)
        cv = (RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3,
                                      random_state=RANDOM_STATE)
              if n < SMALL_N else StratifiedKFold(
                  n_splits, shuffle=True, random_state=RANDOM_STATE))
    scoring = {"f1": "f1_macro", "acc": "accuracy"}

    rows, fitted = [], {}
    for name, model in classification_candidates(n, num, cat).items():
        res = cross_validate(model, X_train, y_train, cv=cv,
                             groups=groups, scoring=scoring)
        model.fit(X_train, y_train)
        row = {"model": name,
               "cv_f1_macro": round(float(res["test_f1"].mean()), 4),
               "cv_accuracy": round(float(res["test_acc"].mean()), 4),
               "test_accuracy": None, "test_f1_macro": None}
        if X_test is not None:
            pred = model.predict(X_test)
            row["test_accuracy"] = round(float(accuracy_score(y_test, pred)), 4)
            row["test_f1_macro"] = round(
                float(f1_score(y_test, pred, average="macro")), 4)
        rows.append(row)
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)
    best = results.iloc[0]["model"]
    return fitted[best], best, results


# ------------------------------------------------------------------ clusters
def train_clustering(df: pd.DataFrame, num, cat):
    """Scale numbers + one-hot categoricals, pick k by silhouette, fit K-Means.

    Unsupervised and uses FEATURES only (no productivity columns), so it is
    fitted on all rows. Silhouette is a guide: check the profiles make sense.
    """
    prep = make_prep(num, cat, scale=True)
    X_t = prep.fit_transform(df[FEATURES])

    ks = [k for k in N_CLUSTERS_RANGE if k < len(df)]
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=RANDOM_STATE).fit_predict(X_t)
        scores[k] = silhouette_score(
            X_t, labels, sample_size=min(4000, len(df)),
            random_state=RANDOM_STATE)
    best_k = max(scores, key=scores.get)

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_t)

    # Profiles: mean of each numeric feature, most common value of each
    # categorical one, and the average score (for reading, not for fitting).
    profiles = (df[num + [REGRESSION_TARGET]].assign(cluster=labels)
                .groupby("cluster").mean().round(2))
    if cat:
        modes = (df[cat].assign(cluster=labels).groupby("cluster")
                 .agg(lambda s: s.mode().iloc[0]))
        profiles = profiles.join(modes)
    profiles.insert(0, "days", pd.Series(labels).value_counts().sort_index())
    pipeline = Pipeline([("prep", prep), ("kmeans", kmeans)])
    return pipeline, best_k, scores, profiles


def typical_values(df: pd.DataFrame, num, cat) -> dict:
    """Median (numbers) or most common value (text) of every feature.

    The app uses these for habits a user chose not to track, so a skipped habit
    neither adds to nor takes from the estimate.
    """
    out = {}
    for c in num:
        col = df[c].dropna()
        med = float(col.median())
        out[c] = int(round(med)) if (col % 1 == 0).all() else round(med, 1)
    for c in cat:
        out[c] = str(df[c].mode().iloc[0])
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    df = load_data()
    n = len(df)
    num, cat = split_feature_types(df)
    print(f"Loaded {n} rows, {len(FEATURES)} features "
          f"({len(num)} numeric, {len(cat)} categorical: {cat})")
    if n < SMALL_N:
        print(f"Small-data mode (< {SMALL_N} rows): repeated CV, "
              "no hold-out, no XGBoost.\n")
    else:
        print(f"XGBoost {'ON' if use_xgb(n) else 'OFF'} "
              f"(needs >= {XGB_MIN_ROWS} rows and the package installed)\n")

    level_to_id = {name: i for i, name in enumerate(LEVELS)}
    train_df, test_df = split_data(df)
    split_info = split_description(train_df, test_df)
    print("Split:", split_info, "\n")

    X_train = train_df[FEATURES]
    yr_train = train_df[REGRESSION_TARGET]
    yc_train = train_df[CLASSIFICATION_TARGET].map(level_to_id)
    groups = _groups(train_df)
    if test_df is None:
        X_test = yr_test = yc_test = None
    else:
        X_test = test_df[FEATURES]
        yr_test = test_df[REGRESSION_TARGET]
        yc_test = test_df[CLASSIFICATION_TARGET].map(level_to_id)

    print("=== Regression: predict productivity_score ===")
    reg, reg_name, reg_results = train_regression(
        X_train, yr_train, X_test, yr_test, groups, num, cat)
    print(reg_results.to_string(index=False))
    print(f"Best (by CV R2): {reg_name}\n")

    print("=== Classification: predict productivity_level ===")
    clf, clf_name, clf_results = train_classification(
        X_train, yc_train, X_test, yc_test, groups, num, cat)
    print(clf_results.to_string(index=False))
    print(f"Best (by CV macro-F1): {clf_name}\n")

    print("=== Clustering: K-Means ===")
    km, best_k, sil_scores, profiles = train_clustering(df, num, cat)
    for k, s in sil_scores.items():
        print(f"k={k}: silhouette={s:.3f}")
    print(f"Chosen k = {best_k}\n")
    print(profiles.to_string(), "\n")

    # Hold-out metrics for the selected models (reporting only).
    reg_test = clf_test = None
    if X_test is not None:
        reg_test = regression_metrics(yr_test, reg.predict(X_test))
        pred = clf.predict(X_test)
        ids = list(range(len(LEVELS)))
        clf_test = {
            "accuracy": round(float(accuracy_score(yc_test, pred)), 4),
            "f1_macro": round(float(f1_score(yc_test, pred, average="macro")), 4),
            "confusion_matrix": confusion_matrix(
                yc_test, pred, labels=ids).tolist(),
            "labels": LEVELS,
            "per_class": classification_report(
                yc_test, pred, labels=ids, target_names=LEVELS,
                output_dict=True, zero_division=0),
        }
        print("Hold-out (selected models):", reg_test,
              {k: clf_test[k] for k in ("accuracy", "f1_macro")}, "\n")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, ARTIFACT_DIR / "regressor.joblib")
    joblib.dump(clf, ARTIFACT_DIR / "classifier.joblib")
    joblib.dump(km, ARTIFACT_DIR / "kmeans.joblib")
    profiles.to_csv(ARTIFACT_DIR / "cluster_profiles.csv")

    importance = {}
    imp = feature_importance(reg, cat)
    if imp:
        importance["regressor"] = imp
    imp = feature_importance(clf, cat)
    if imp:
        importance["classifier"] = imp
    coefficients = linear_coefficients(reg)

    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "trained_at_utc": now.isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "experimental": True,
        "disclaimer": DISCLAIMER,
        "features": FEATURES,
        "numeric_features": num,
        "categorical_features": cat,
        "categories": {c: sorted(df[c].astype(str).unique().tolist()) for c in cat},
        "typical": typical_values(df, num, cat),
        "levels": LEVELS,
        "rows_trained_on": int(n),
        "split": split_info,
        "regression": {
            "selected": reg_name,
            "selection_rule": "highest cross-validated R2 on the training part",
            "candidates": reg_results.to_dict("records"),
            "holdout": reg_test,
        },
        "classification": {
            "selected": clf_name,
            "selection_rule": "highest cross-validated macro-F1 on the training part",
            "candidates": clf_results.to_dict("records"),
            "holdout": clf_test,
        },
        "kmeans": {"k": int(best_k),
                   "silhouette": round(float(sil_scores[best_k]), 4),
                   "silhouette_by_k": {str(k): round(float(s), 4)
                                       for k, s in sil_scores.items()}},
        "feature_importance": importance,
        "regressor_coefficients": coefficients,
        "random_state": RANDOM_STATE,
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved models and metadata to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
