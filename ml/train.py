"""Train, compare and save the three models.

Run from the project root:
    python -m ml.train

Saves to artifacts/:
    regressor.joblib, classifier.joblib, kmeans.joblib,
    metadata.json, cluster_profiles.csv

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
from sklearn.cluster import KMeans
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
from sklearn.preprocessing import StandardScaler

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


# --------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Generate it first:  python -m ml.dummydataset --rows 10000"
        )
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
def regression_candidates(n_rows: int) -> dict:
    small = n_rows < SMALL_N
    candidates = {
        "ridge": Pipeline([("scale", StandardScaler()),
                           ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4 if small else None,
            min_samples_leaf=5 if small else 3,
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    if use_xgb(n_rows):
        candidates["xgboost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, random_state=RANDOM_STATE, n_jobs=-1)
    return candidates


def regression_metrics(y_true, pred) -> dict:
    return {"r2": round(float(r2_score(y_true, pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 3)}


def train_regression(X_train, y_train, X_test=None, y_test=None, groups=None):
    n = len(X_train) + (0 if X_test is None else len(X_test))
    if groups is not None:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
    elif n < SMALL_N:
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    else:
        cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    rows, fitted = [], {}
    for name, model in regression_candidates(n).items():
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
def classification_candidates(n_rows: int) -> dict:
    small = n_rows < SMALL_N
    candidates = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4 if small else None,
            min_samples_leaf=5 if small else 3,
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    if use_xgb(n_rows):
        candidates["xgboost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, random_state=RANDOM_STATE,
            eval_metric="mlogloss", n_jobs=-1)
    return candidates


def train_classification(X_train, y_train, X_test=None, y_test=None,
                         groups=None):
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
    for name, model in classification_candidates(n).items():
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
def train_clustering(df: pd.DataFrame):
    """Scale features, pick k by silhouette score, fit K-Means.

    Unsupervised and uses FEATURES only (no productivity columns), so it is
    fitted on all rows. Silhouette is a guide: check the profiles make sense.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    ks = [k for k in N_CLUSTERS_RANGE if k < len(df)]
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=RANDOM_STATE).fit_predict(X_scaled)
        scores[k] = silhouette_score(
            X_scaled, labels, sample_size=min(4000, len(df)),
            random_state=RANDOM_STATE)
    best_k = max(scores, key=scores.get)

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled)

    profiles = df[FEATURES + [REGRESSION_TARGET]].assign(cluster=labels)
    profiles = profiles.groupby("cluster").mean().round(2)
    profiles.insert(0, "days", pd.Series(labels).value_counts().sort_index())
    pipeline = Pipeline([("scale", scaler), ("kmeans", kmeans)])
    return pipeline, best_k, scores, profiles


# ---------------------------------------------------------------------- main
def main() -> None:
    df = load_data()
    n = len(df)
    print(f"Loaded {n} rows, {len(FEATURES)} features")
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
        X_train, yr_train, X_test, yr_test, groups)
    print(reg_results.to_string(index=False))
    print(f"Best (by CV R2): {reg_name}\n")

    print("=== Classification: predict productivity_level ===")
    clf, clf_name, clf_results = train_classification(
        X_train, yc_train, X_test, yc_test, groups)
    print(clf_results.to_string(index=False))
    print(f"Best (by CV macro-F1): {clf_name}\n")

    print("=== Clustering: K-Means ===")
    km, best_k, sil_scores, profiles = train_clustering(df)
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
    for label, model in (("regressor", reg), ("classifier", clf)):
        if hasattr(model, "feature_importances_"):
            importance[label] = dict(zip(
                FEATURES, np.round(model.feature_importances_, 4).tolist()))

    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "trained_at_utc": now.isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "experimental": True,
        "disclaimer": DISCLAIMER,
        "features": FEATURES,
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
        "random_state": RANDOM_STATE,
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved models and metadata to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
