"""
dummydataset.py: synthetic daily productivity dataset for PPIA.

Run from the project root (as a module, never the VS Code Run button):
    python -m ml.dummydataset
    python -m ml.dummydataset --seed 7 --rows 5000

Writes datasets/productivity_logs.csv (the path set in ml/settings.py).

WHAT THIS IS
    Controlled experiment data, NOT evidence about real human behaviour.
    A model trained on this learns the artificial rules written below.

HOW IT IS BUILT
    * N_PEOPLE synthetic people, each simulated for consecutive days.
    * Each person has a personal baseline.
    * Stress follows a slow AR(1) process, so bad days cluster together.
    * Features are correlated (stress lowers sleep, study hours drive study
      sessions, caffeine rises with study and short sleep, stress lowers the
      chance of journaling, ...).
    * productivity_score is a NONLINEAR function of the features plus
      momentum from the previous day and random noise.
    * ~0.5% of rows get extreme outliers; ~3% of values are blanked in four
      input columns to mimic missing data (train.py drops those rows).

COLUMNS THE ML MODULE USES (see ml/settings.py)
    Features: study_hours, caffeine_mg, stress_level, study_sessions, journaled
    Targets : productivity_score (0-100), productivity_level (low/medium/high)

CAFFEINE SCALE
    The score formula works in real milligrams internally. The saved
    caffeine_mg column is converted to the app's 0-10 scale (mg / 60, so
    600 mg = 10) because that is what the app's habit form logs.

LEAKAGE WARNING
    previous_day_productivity is legitimate (known before the day starts) but
    is not a model feature. NEVER use productivity_score, productivity_level
    or productive_day as an input.

EXPERIMENT-ONLY COLUMNS (do not exist in the production app)
    person_id, date, and the extra columns (sleep, screen time, ...).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.settings import DATASET_PATH, LEVEL_BINS, LEVELS

# ----------------------------- defaults ------------------------------------
DEFAULT_SEED = 42
DEFAULT_ROWS = 2_000
N_PEOPLE = 20
DEFAULT_THRESHOLD = 60.0   # productive_day = score >= this
MISSING_RATE = 0.03        # per selected column
OUTLIER_RATE = 0.005       # per row
MISSING_COLUMNS = ["sleep_hours", "screen_time", "exercise_minutes", "caffeine_mg"]
MG_PER_POINT = 60          # caffeine: 60 mg = 1 point on the app's 0-10 scale

DEFAULT_OUTPUT = DATASET_PATH


def simulate_person(rng, n_days, start_date):
    """Return a DataFrame of features for one person (no score yet)."""
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()          # 0 = Monday ... 6 = Sunday
    weekend = (dow >= 5).astype(float)

    # Personal baselines
    base_sleep = rng.normal(7.0, 0.5)
    base_study = rng.normal(4.0, 0.9)
    base_screen = rng.normal(6.0, 1.2)
    base_stress = rng.normal(5.0, 0.8)
    base_journal = rng.normal(0.0, 0.12)      # some people journal more

    # Stress: AR(1) so consecutive days are related
    stress_latent = np.empty(n_days)
    stress_latent[0] = base_stress
    for t in range(1, n_days):
        stress_latent[t] = (
            base_stress + 0.8 * (stress_latent[t - 1] - base_stress) + rng.normal(0, 0.9)
        )
    stress = np.clip(np.round(stress_latent), 1, 10)

    # Study: lower on weekends, right-skewed
    study = base_study * (1 - 0.45 * weekend) + rng.gamma(2.0, 0.6, n_days) - 1.2
    study = np.clip(study, 0, 14)
    study[rng.random(n_days) < 0.06] = 0.0     # occasional zero-study days

    # Sessions relate to hours (about one session per 1.5 h) with noise
    sessions = np.round(study / 1.5 + rng.normal(0, 0.7, n_days))
    sessions = np.clip(sessions, 0, None)
    sessions[study < 0.3] = 0

    # Journaling: less likely on stressful days
    p_journal = np.clip(0.55 + base_journal - 0.05 * (stress - 5), 0.05, 0.95)
    journaled = (rng.random(n_days) < p_journal).astype(float)

    # Sleep: weekends longer, stress shortens it
    sleep = (
        base_sleep + 0.6 * weekend - 0.18 * (stress - 5) + rng.normal(0, 0.9, n_days)
    )
    sleep = np.clip(sleep, 3.0, 11.5)

    # Screen time: loosely tied to study, higher on weekends
    screen = (
        base_screen + 0.25 * (study - base_study) + 1.0 * weekend + rng.normal(0, 1.6, n_days)
    )
    screen = np.clip(screen, 0.5, 16)

    # Exercise: many zeros, otherwise right-skewed minutes
    exercise = rng.lognormal(mean=3.4, sigma=0.6, size=n_days)
    exercise[rng.random(n_days) < 0.42] = 0.0
    exercise = np.clip(exercise, 0, 150)

    # Caffeine (mg): rises with study and short sleep, some people drink none
    caffeine = 60 + 22 * study + 25 * np.clip(7.5 - sleep, 0, None) + rng.normal(0, 55, n_days)
    caffeine[rng.random(n_days) < 0.12] = 0.0
    caffeine = np.clip(caffeine, 0, 600)

    # Breaks scale with sessions
    breaks = np.round(sessions * 1.2 + rng.normal(0, 1.0, n_days))
    breaks = np.clip(breaks, 0, None)

    return pd.DataFrame(
        {
            "date": dates,
            "day_of_week": dow,
            "study_hours": study,
            "sleep_hours": sleep,
            "screen_time": screen,
            "exercise_minutes": exercise,
            "caffeine_mg": caffeine,
            "stress_level": stress,
            "break_count": breaks,
            "study_sessions": sessions,
            "journaled": journaled,
        }
    )


def add_outliers(df, rng):
    """Replace a few rows' values with extreme but still valid numbers."""
    hit = rng.random(len(df)) < OUTLIER_RATE
    n = int(hit.sum())
    if n == 0:
        return df
    choices = rng.integers(0, 4, size=n)
    idx = df.index[hit]
    for i, c in zip(idx, choices):
        if c == 0:
            df.at[i, "study_hours"] = rng.uniform(12, 18)
        elif c == 1:
            df.at[i, "screen_time"] = rng.uniform(14, 20)
        elif c == 2:
            df.at[i, "caffeine_mg"] = rng.uniform(600, 900)
        else:
            df.at[i, "sleep_hours"] = rng.uniform(1.5, 3.0)
    return df


def compute_score(df, rng, person_effect):
    """Sequentially compute the score so each day can depend on the previous one."""
    n = len(df)
    score = np.empty(n)
    prev = np.empty(n)

    sleep = df["sleep_hours"].to_numpy()
    study = df["study_hours"].to_numpy()
    screen = df["screen_time"].to_numpy()
    exercise = df["exercise_minutes"].to_numpy()
    caffeine = df["caffeine_mg"].to_numpy()       # still in mg here
    stress = df["stress_level"].to_numpy()
    breaks = df["break_count"].to_numpy()
    sessions = df["study_sessions"].to_numpy()
    journaled = df["journaled"].to_numpy()
    dow = df["day_of_week"].to_numpy()

    # Nonlinear effects (all vectorised)
    sleep_eff = 16 * np.exp(-((sleep - 7.5) / 1.6) ** 2) - 8
    study_eff = 11 * (1 - np.exp(-study / 2.5)) - 0.9 * np.clip(study - 8, 0, None) ** 2
    exercise_eff = 5 * (1 - np.exp(-exercise / 30)) - 0.0008 * np.clip(exercise - 90, 0, None) ** 2
    caffeine_eff = 3 * np.exp(-((caffeine - 150) / 120) ** 2) - 0.015 * np.clip(caffeine - 350, 0, None)
    screen_eff = -1.3 * np.clip(screen - 5, 0, None)
    stress_eff = -2.2 * (stress - 5) - 0.6 * np.clip(stress - 8, 0, None) ** 2
    break_eff = 1.0 * np.minimum(breaks, 5) - 0.8 * np.clip(breaks - sessions * 2, 0, None)
    weekend_eff = np.where(dow >= 5, -2.0, 0.0)
    journal_eff = 3.0 * journaled

    static = (
        48 + person_effect + sleep_eff + study_eff + exercise_eff + caffeine_eff
        + screen_eff + stress_eff + break_eff + weekend_eff + journal_eff
    )

    prev_score = 50.0 + person_effect          # day 0 has no real "yesterday"
    for t in range(n):
        prev[t] = prev_score
        s = static[t] + 0.25 * (prev_score - 55) + rng.normal(0, 6)
        score[t] = np.clip(s, 0, 100)
        prev_score = score[t]
    return score, prev


def generate(rows, seed, threshold):
    rng = np.random.default_rng(seed)
    days_each = int(np.ceil(rows / N_PEOPLE))
    frames = []

    for pid in range(1, N_PEOPLE + 1):
        person_df = simulate_person(rng, days_each, "2024-01-01")
        person_df = add_outliers(person_df.reset_index(drop=True), rng)

        # Keep values inside the same limits the app's forms will enforce
        person_df["study_hours"] = person_df["study_hours"].clip(0, 24)
        person_df["screen_time"] = person_df["screen_time"].clip(0, 24)
        person_df["sleep_hours"] = person_df["sleep_hours"].clip(0, 24)
        person_df["caffeine_mg"] = person_df["caffeine_mg"].clip(0, 1000)

        person_effect = rng.normal(0, 4)
        score, prev = compute_score(person_df, rng, person_effect)
        person_df["productivity_score"] = score
        person_df["previous_day_productivity"] = prev
        person_df.insert(0, "person_id", pid)
        # The first day has no real yesterday, so mark it missing
        person_df.loc[0, "previous_day_productivity"] = np.nan
        frames.append(person_df)

    df = pd.concat(frames, ignore_index=True).iloc[:rows].copy()

    # Convert caffeine from mg (used by the score formula) to the app's 0-10 scale
    df["caffeine_mg"] = (df["caffeine_mg"] / MG_PER_POINT).clip(0, 10)

    # Rounding to realistic precision
    for col in ["study_hours", "sleep_hours", "screen_time", "caffeine_mg"]:
        df[col] = df[col].round(1)
    df["exercise_minutes"] = df["exercise_minutes"].round(0)
    for col in ["stress_level", "break_count", "study_sessions", "journaled"]:
        df[col] = df[col].astype(int)
    df["productivity_score"] = df["productivity_score"].round(1)
    df["previous_day_productivity"] = df["previous_day_productivity"].round(1)

    # Labels are derived BEFORE masking so they reflect the true score
    df["productive_day"] = (df["productivity_score"] >= threshold).astype(int)
    df["productivity_level"] = pd.cut(
        df["productivity_score"], bins=LEVEL_BINS, labels=LEVELS
    ).astype(str)

    # Missing values in selected INPUT columns only (never in targets)
    for col in MISSING_COLUMNS:
        mask = rng.random(len(df)) < MISSING_RATE
        df.loc[mask, col] = np.nan

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    cols = [
        "person_id", "date", "day_of_week",
        "study_hours", "sleep_hours", "screen_time", "exercise_minutes",
        "caffeine_mg", "stress_level", "break_count", "study_sessions",
        "journaled", "previous_day_productivity",
        "productivity_score", "productivity_level", "productive_day",
    ]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic PPIA dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate(args.rows, args.seed, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Seed: {args.seed} | productive threshold: {args.threshold}")
    print(f"Productive-day share: {df['productive_day'].mean():.1%}")
    print("Level shares:")
    print(df["productivity_level"].value_counts(normalize=True)
          .reindex(LEVELS).map("{:.1%}".format).to_string())
    print("Missing values per column:")
    print(df.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()