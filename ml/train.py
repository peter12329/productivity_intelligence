"""Train, compare and save the three models.

Run from the project root:
    python -m ml.train

Saves to artifacts/:
    regressor.joblib, classifier.joblib, kmeans.joblib,
    metadata.json, cluster_profiles.csv

The candidate models and the validation scheme depend on the row count:
    rows < SMALL_N   : no hold-out split, repeated k-fold CV only,
                       simple / regularised models (no XGBoost)
    SMALL_N <= rows  : 80/20 hold-out + 5-fold CV on the training part
    rows >= XGB_MIN_ROWS : XGBoost is added to the candidates
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, silhouette_score)
from sklearn.model_selection import (RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, KFold, cross_validate,
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


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Generate the dataset first (your dummy dataset script)."
        )
    df = pd.read_csv(DATASET_PATH)
    needed = FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    return df.dropna(subset=needed).reset_index(drop=True)


def use_xgb(n_rows: int) -> bool:
    return HAS_XGB and n_rows >= XGB_MIN_ROWS


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


def train_regression(X_train, y_train, X_test=None, y_test=None):
    n = len(X_train) + (0 if X_test is None else len(X_test))
    cv = (RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
          if n < SMALL_N else KFold(5, shuffle=True,
                                    random_state=RANDOM_STATE))
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    rows, fitted = [], {}
    for name, model in regression_candidates(n).items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        model.fit(X_train, y_train)
        row = {"model": name,
               "cv_r2": round(res["test_r2"].mean(), 4),
               "cv_mae": round(-res["test_mae"].mean(), 3),
               "test_r2": None, "test_mae": None}
        if X_test is not None:
            pred = model.predict(X_test)
            row["test_r2"] = round(r2_score(y_test, pred), 4)
            row["test_mae"] = round(mean_absolute_error(y_test, pred), 3)
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


def train_classification(X_train, y_train, X_test=None, y_test=None):
    n = len(X_train) + (0 if X_test is None else len(X_test))
    min_class = int(pd.Series(y_train).value_counts().min())
    if min_class < 2:
        raise SystemExit(
            "A level has fewer than 2 rows; cannot cross-validate the "
            "classifier. Log more days or merge levels.")
    n_splits = min(5, min_class)
    cv = (RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3,
                                  random_state=RANDOM_STATE)
          if n < SMALL_N else StratifiedKFold(n_splits, shuffle=True,
                                              random_state=RANDOM_STATE))
    scoring = {"f1": "f1_macro", "acc": "accuracy"}

    rows, fitted = [], {}
    for name, model in classification_candidates(n).items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        model.fit(X_train, y_train)
        row = {"model": name,
               "cv_f1_macro": round(res["test_f1"].mean(), 4),
               "cv_accuracy": round(res["test_acc"].mean(), 4),
               "test_accuracy": None, "test_f1_macro": None}
        if X_test is not None:
            pred = model.predict(X_test)
            row["test_accuracy"] = round(accuracy_score(y_test, pred), 4)
            row["test_f1_macro"] = round(
                f1_score(y_test, pred, average="macro"), 4)
        rows.append(row)
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)
    best = results.iloc[0]["model"]
    return fitted[best], best, results


# ------------------------------------------------------------------ clusters
def train_clustering(df: pd.DataFrame):
    """Scale features, pick k by silhouette score, fit K-Means."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    ks = [k for k in N_CLUSTERS_RANGE if k < len(df)]
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=RANDOM_STATE).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
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

    X = df[FEATURES]
    level_to_id = {name: i for i, name in enumerate(LEVELS)}
    y_reg = df[REGRESSION_TARGET]
    y_clf = df[CLASSIFICATION_TARGET].map(level_to_id)

    if n < SMALL_N:
        X_train, X_test = X, None
        yr_train, yr_test = y_reg, None
        yc_train, yc_test = y_clf, None
    else:
        X_train, X_test, yr_train, yr_test, yc_train, yc_test = \
            train_test_split(X, y_reg, y_clf, test_size=0.2,
                             random_state=RANDOM_STATE, stratify=y_clf)

    print("=== Regression: predict productivity_score ===")
    reg, reg_name, reg_results = train_regression(
        X_train, yr_train, X_test, yr_test)
    print(reg_results.to_string(index=False))
    print(f"Best: {reg_name}\n")

    print("=== Classification: predict productivity_level ===")
    clf, clf_name, clf_results = train_classification(
        X_train, yc_train, X_test, yc_test)
    print(clf_results.to_string(index=False))
    print(f"Best: {clf_name}\n")

    print("=== Clustering: K-Means ===")
    km, best_k, sil_scores, profiles = train_clustering(df)
    for k, s in sil_scores.items():
        print(f"k={k}: silhouette={s:.3f}")
    print(f"Chosen k = {best_k}\n")
    print(profiles.to_string(), "\n")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, ARTIFACT_DIR / "regressor.joblib")
    joblib.dump(clf, ARTIFACT_DIR / "classifier.joblib")
    joblib.dump(km, ARTIFACT_DIR / "kmeans.joblib")
    profiles.to_csv(ARTIFACT_DIR / "cluster_profiles.csv")

    # Feature importance from the tree-based models, if the winner has it.
    importance = {}
    for label, model in (("regressor", reg), ("classifier", clf)):
        if hasattr(model, "feature_importances_"):
            importance[label] = dict(zip(
                FEATURES, np.round(model.feature_importances_, 4).tolist()))

    metadata = {
        "features": FEATURES,
        "levels": LEVELS,
        "kmeans": {"k": int(best_k),
                   "silhouette": round(float(sil_scores[best_k]), 4)},
        "feature_importance": importance,
        "rows_trained_on": int(n),
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved models and metadata to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()