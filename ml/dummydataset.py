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
    * N_PEOPLE synthetic people, each simulated for consecutive calendar days.
    * Each person has a personal baseline and their own logging habit: they
      keep logging with 90% probability, and resume after a miss with 40%.
      Days they did not log are dropped, so log_streak is meaningful.
    * day_type (class / free / deadline) drives free_hours, task_difficulty,
      disruptions and stress. deep_work can never exceed free_hours.
    * Stress follows a slow AR(1) process, so bad days cluster together.
    * productivity_score is a NONLINEAR function of the features plus
      momentum from the previous day and random noise.
    * ~0.5% of rows get extreme outliers; ~3% of values are blanked in four
      input columns to mimic missing data (train.py drops those rows).

MODEL FEATURES (14 inputs, see ml/settings.py)
    Numeric    : deep_work, sleep_hours, caffeine_level, stress_level,
                 journaled, log_streak, disruptions, junk_food,
                 task_difficulty, deep_work_days_prev7, free_hours
    Categorical: day_type, weather        (one-hot encode in the ML code)
    Targets    : productivity_score (0-100), productivity_level (low/medium/high)

FEATURE DEFINITIONS
    log_streak            consecutive logged days ending YESTERDAY, capped at 14.
    deep_work_days_prev7  how many of the previous 7 calendar days met the deep
                          work goal (>= 3h, and were logged). 0-7. Today is
                          excluded. It is a count, not a streak, so one missed
                          day does not reset it. For a person's first week it
                          counts only the days that exist.
    caffeine_level        0-10 scale, mg / 40, so 400 mg = 10 (the app's
                          maximum) and "under 300mg" means a level below 7.5
                          (7 or lower if the app logs whole levels).

LEAKAGE WARNING
    previous_day_productivity is legitimate (known before the day starts) but
    is not a model feature. NEVER use productivity_score, productivity_level
    or productive_day as an input.

EXPERIMENT-ONLY COLUMNS (do not exist in the production app)
    person_id, date, previous_day_productivity.
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
MISSING_COLUMNS = ["sleep_hours", "caffeine_level", "task_difficulty", "free_hours"]
MG_PER_POINT = 40          # caffeine: 40 mg = 1 point, 400 mg = 10, 300 mg = 7.5

DEEP_WORK_GOAL = 3.0       # hours; same as the app's "Deep work 3h+" habit
STREAK_CAP = 14            # log_streak is capped so it stays a bounded feature
STAY_LOGGING = 0.90        # P(log today | logged yesterday)
RESUME_LOGGING = 0.40      # P(log today | missed yesterday)
DAYS_MARGIN = 1.5          # simulate extra calendar days, missed days are dropped

BASE_SCORE = 43            # overall level of the score formula

DEFAULT_OUTPUT = DATASET_PATH


def simulate_person(rng, n_days, start_date):
    """Return a DataFrame of features for one person (no score yet).

    Every calendar day is simulated; the `logged` column says which days the
    person actually logged. Unlogged days are dropped later in generate().
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()          # 0 = Monday ... 6 = Sunday
    weekend = (dow >= 5).astype(float)

    # Personal baselines
    base_sleep = rng.normal(7.0, 0.5)
    base_deep = rng.normal(3.5, 0.8)
    base_stress = rng.normal(5.0, 0.8)
    base_journal = rng.normal(0.0, 0.12)      # some people journal more

    # Logging behaviour: who logged on which day, and the streak ending yesterday
    logged = np.empty(n_days, dtype=bool)
    logged[0] = True
    for t in range(1, n_days):
        p = STAY_LOGGING if logged[t - 1] else RESUME_LOGGING
        logged[t] = rng.random() < p
    streak_raw = np.zeros(n_days)
    for t in range(1, n_days):
        streak_raw[t] = streak_raw[t - 1] + 1 if logged[t - 1] else 0
    streak = np.minimum(streak_raw, STREAK_CAP)

    # Day type: weekends are mostly free days, weekdays mostly class days
    u = rng.random(n_days)
    day_type = np.where(
        weekend == 1,
        np.where(u < 0.10, "class", np.where(u < 0.80, "free", "deadline")),
        np.where(u < 0.60, "class", np.where(u < 0.75, "free", "deadline")),
    )
    is_class = day_type == "class"
    is_free = day_type == "free"
    is_deadline = (day_type == "deadline").astype(float)

    # Free hours available for work, set mainly by the day type
    free_mean = np.select([is_class, is_free], [5.0, 9.0], default=6.0)
    free = np.clip(free_mean + rng.normal(0, 1.5, n_days), 0, 14)

    # Task difficulty (1-5): harder on deadline days
    difficulty = np.clip(np.round(2.8 + 0.8 * is_deadline + rng.normal(0, 1.0, n_days)), 1, 5)

    # Stress: AR(1) so consecutive days are related, plus deadline and difficulty
    stress_ar = np.empty(n_days)
    stress_ar[0] = base_stress
    for t in range(1, n_days):
        stress_ar[t] = (
            base_stress + 0.8 * (stress_ar[t - 1] - base_stress) + rng.normal(0, 0.9)
        )
    stress = np.clip(np.round(stress_ar + 1.2 * is_deadline + 0.4 * (difficulty - 3)), 1, 10)

    # Disruptions (meetings, interruptions): fewer on weekends
    lam = np.select([is_class, is_free], [1.5, 0.8], default=1.2) * (1 - 0.3 * weekend)
    disruptions = np.clip(rng.poisson(lam), 0, 8).astype(float)

    # Deep work (hours): limited by free hours, cut by disruptions and hard tasks
    deep_work = (
        base_deep * (0.7 + 0.3 * free / 6)
        - 0.35 * disruptions
        - 0.25 * (difficulty - 3)
        + rng.gamma(2.0, 0.6, n_days) - 1.2
    )
    deep_work = np.clip(deep_work, 0, free)
    deep_work[rng.random(n_days) < 0.06] = 0.0     # occasional zero-work days
    deep_work = np.clip(deep_work, 0, 14)

    # Journaling: less likely on stressful days
    p_journal = np.clip(0.55 + base_journal - 0.05 * (stress - 5), 0.05, 0.95)
    journaled = (rng.random(n_days) < p_journal).astype(float)

    # Junk food: more likely when stressed and on weekends
    p_junk = np.clip(0.25 + 0.05 * (stress - 5) + 0.05 * weekend, 0.05, 0.80)
    junk_food = (rng.random(n_days) < p_junk).astype(float)

    # Sleep: weekends longer, stress shortens it
    sleep = (
        base_sleep + 0.6 * weekend - 0.18 * (stress - 5) + rng.normal(0, 0.9, n_days)
    )
    sleep = np.clip(sleep, 3.0, 11.5)

    # Caffeine (mg): rises with deep work and short sleep, some people drink none
    caffeine = 60 + 22 * deep_work + 25 * np.clip(7.5 - sleep, 0, None) + rng.normal(0, 55, n_days)
    caffeine[rng.random(n_days) < 0.12] = 0.0
    caffeine = np.clip(caffeine, 0, 600)

    # Weather: independent of everything else, only a tiny effect on the score
    weather = rng.choice(["sunny", "cloudy", "rainy"], size=n_days, p=[0.5, 0.3, 0.2])

    # Deep-work goal days among the previous 7 calendar days (today excluded).
    # A day counts only if it was logged AND met the goal.
    met = pd.Series(((deep_work >= DEEP_WORK_GOAL) & logged).astype(float))
    prev7 = met.shift(1).rolling(7, min_periods=1).sum().fillna(0).to_numpy()

    return pd.DataFrame(
        {
            "date": dates,
            "day_of_week": dow,
            "logged": logged,
            "deep_work": deep_work,
            "sleep_hours": sleep,
            "caffeine_mg": caffeine,          # renamed to caffeine_level in generate()
            "stress_level": stress,
            "journaled": journaled,
            "log_streak": streak,
            "disruptions": disruptions,
            "junk_food": junk_food,
            "task_difficulty": difficulty,
            "deep_work_days_prev7": prev7,
            "day_type": day_type,
            "weather": weather,
            "free_hours": free,
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
            df.at[i, "deep_work"] = rng.uniform(12, 18)
        elif c == 1:
            df.at[i, "caffeine_mg"] = rng.uniform(600, 900)
        elif c == 2:
            df.at[i, "sleep_hours"] = rng.uniform(1.5, 3.0)
        else:
            df.at[i, "disruptions"] = rng.integers(10, 16)
    return df


def compute_score(df, rng, person_effect):
    """Sequentially compute the score so each day can depend on the previous one."""
    n = len(df)
    score = np.empty(n)
    prev = np.empty(n)

    deep = df["deep_work"].to_numpy()
    sleep = df["sleep_hours"].to_numpy()
    caffeine = df["caffeine_mg"].to_numpy()        # still in mg here
    stress = df["stress_level"].to_numpy()
    journaled = df["journaled"].to_numpy()
    streak = df["log_streak"].to_numpy()
    disruptions = df["disruptions"].to_numpy()
    junk = df["junk_food"].to_numpy()
    difficulty = df["task_difficulty"].to_numpy()
    prev7 = df["deep_work_days_prev7"].to_numpy()
    day_type = df["day_type"].to_numpy()
    weather = df["weather"].to_numpy()
    free = df["free_hours"].to_numpy()

    # Nonlinear effects (all vectorised)
    sleep_eff = 16 * np.exp(-((sleep - 7.5) / 1.6) ** 2) - 8
    deep_eff = 11 * (1 - np.exp(-deep / 2.5)) - 0.9 * np.clip(deep - 8, 0, None) ** 2
    caffeine_eff = 3 * np.exp(-((caffeine - 150) / 120) ** 2) - 0.015 * np.clip(caffeine - 350, 0, None)
    stress_eff = -2.2 * (stress - 5) - 0.6 * np.clip(stress - 8, 0, None) ** 2
    journal_eff = 3.0 * journaled

    # New features
    use_eff = 4.0 * np.clip(deep / np.maximum(free, 1.0), 0, 1)       # used free time well
    disruption_eff = -1.5 * np.minimum(disruptions, 6)
    junk_eff = -2.0 * junk
    difficulty_eff = -1.0 * (difficulty - 3)
    streak_eff = 0.3 * streak                                          # max +4.2
    prev7_eff = 0.6 * prev7 - 0.5 * np.clip(prev7 - 5, 0, None) ** 2  # fades past 5 (burnout)
    day_type_eff = np.select(
        [day_type == "free", day_type == "deadline"], [-1.5, 1.0], default=0.0
    )
    weather_eff = np.select(
        [weather == "sunny", weather == "rainy"], [0.8, -0.8], default=0.0
    )

    static = (
        BASE_SCORE + person_effect
        + sleep_eff + deep_eff + caffeine_eff + stress_eff + journal_eff
        + use_eff + disruption_eff + junk_eff + difficulty_eff
        + streak_eff + prev7_eff + day_type_eff + weather_eff
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
    rows_per_person = int(np.ceil(rows / N_PEOPLE))
    days_each = int(np.ceil(rows_per_person * DAYS_MARGIN))
    frames = []

    for pid in range(1, N_PEOPLE + 1):
        person_df = simulate_person(rng, days_each, "2024-01-01")
        person_df = add_outliers(person_df.reset_index(drop=True), rng)

        # Keep values inside the same limits the app's forms will enforce
        person_df["deep_work"] = person_df["deep_work"].clip(0, 24)
        person_df["sleep_hours"] = person_df["sleep_hours"].clip(0, 24)
        person_df["caffeine_mg"] = person_df["caffeine_mg"].clip(0, 1000)

        person_effect = rng.normal(0, 4)
        score, prev = compute_score(person_df, rng, person_effect)
        person_df["productivity_score"] = score
        person_df["previous_day_productivity"] = prev

        # Yesterday is only known if it was a logged day
        person_df.loc[0, "previous_day_productivity"] = np.nan
        yesterday_missing = ~person_df["logged"].shift(1, fill_value=False)
        person_df.loc[yesterday_missing, "previous_day_productivity"] = np.nan

        # Drop the days this person did not log, keep a fixed number of rows
        person_df = person_df[person_df["logged"]].drop(columns="logged")
        person_df = person_df.head(rows_per_person)
        person_df.insert(0, "person_id", pid)
        frames.append(person_df)

    df = pd.concat(frames, ignore_index=True).iloc[:rows].copy()

    # Convert caffeine from mg (used by the score formula) to the app's 0-10 scale
    df["caffeine_level"] = (df["caffeine_mg"] / MG_PER_POINT).clip(0, 10)
    df = df.drop(columns="caffeine_mg")

    # Rounding to realistic precision
    for col in ["deep_work", "sleep_hours", "caffeine_level", "free_hours"]:
        df[col] = df[col].round(1)
    for col in ["stress_level", "journaled", "log_streak", "disruptions",
                "junk_food", "task_difficulty", "deep_work_days_prev7", "day_of_week"]:
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
        "deep_work", "sleep_hours", "caffeine_level", "stress_level",
        "journaled", "log_streak", "disruptions", "junk_food",
        "task_difficulty", "deep_work_days_prev7", "day_type", "weather",
        "free_hours", "previous_day_productivity",
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