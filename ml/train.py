import sqlite3
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from xgboost import XGBRegressor

FEATURES = ["study_hours", "caffeine_mg", "stress_level", "study_sessions", "journaled"]
TARGET = "productivity_score"
SEED = 42

with sqlite3.connect("journal.db") as c:
    df = pd.read_sql("SELECT * FROM entries", c, parse_dates=["date"])

df = df.sort_values("date").dropna(subset=FEATURES + [TARGET])   # TimeSeriesSplit needs time order
X, y = df[FEATURES], df[TARGET]

def make_pipe(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                          random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                  learning_rate=0.05, subsample=0.8,
                                                  random_state=SEED),
    "XGBoost": XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=5, random_state=SEED),
}

cv = TimeSeriesSplit(n_splits=5)
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

rows = []
for name, m in models.items():
    res = cross_validate(make_pipe(m), X, y, cv=cv, scoring=scoring)
    rows.append({
        "model": name,
        "mae":  -res["test_mae"].mean(),
        "rmse": -res["test_rmse"].mean(),
        "r2":    res["test_r2"].mean(),
        "r2_std": res["test_r2"].std(),
    })

cv_table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
print(cv_table.round(3).to_string(index=False))