"""Find better XGBoost settings (prints them, changes nothing).

Run from the project root:  python -m ml.tune
Scores are cross-validated on the TRAINING people only; the hold-out is not used.
"""
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from ml.settings import (CLASSIFICATION_TARGET, FEATURES, LEVELS,
                         RANDOM_STATE, REGRESSION_TARGET)
from ml.train import (GROUP_COL, _groups, classification_candidates,
                      load_data, regression_candidates, split_data,
                      split_feature_types)

N_ITER = 30
SPACE = {
    "model__n_estimators":     [200, 400, 600, 800],
    "model__learning_rate":    [0.02, 0.03, 0.05, 0.1],
    "model__max_depth":        [2, 3, 4, 5, 6],
    "model__min_child_weight": [1, 3, 5, 10],
    "model__subsample":        [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_lambda":       [1, 3, 10],
}


def search(pipe, X, y, groups, scoring):
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    s = RandomizedSearchCV(pipe, SPACE, n_iter=N_ITER, cv=cv, scoring=scoring,
                           random_state=RANDOM_STATE, n_jobs=1)
    s.fit(X, y, groups=groups)
    return s


def main():
    df = load_data()
    if GROUP_COL not in df.columns:
        raise SystemExit("Tuning here needs the person_id column.")
    num, cat = split_feature_types(df)
    train_df, _ = split_data(df)
    groups = _groups(train_df)
    X = train_df[FEATURES]
    n = len(df)

    reg = regression_candidates(n, num, cat)["xgboost"]
    s = search(reg, X, train_df[REGRESSION_TARGET], groups, "r2")
    print(f"Regression  best CV R2 = {s.best_score_:.4f}")
    print("  ", s.best_params_)

    level_to_id = {name: i for i, name in enumerate(LEVELS)}
    clf = classification_candidates(n, num, cat)["xgboost"]
    s = search(clf, X, train_df[CLASSIFICATION_TARGET].map(level_to_id),
               groups, "f1_macro")
    print(f"Classifier  best CV macro-F1 = {s.best_score_:.4f}")
    print("  ", s.best_params_)
    print("\nCompare with your current CV scores (R2 0.5246, F1 0.6206).")


if __name__ == "__main__":
    main()
