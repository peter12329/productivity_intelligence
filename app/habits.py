"""Habit definitions.

Three kinds of things live here:

  HABITS     goals shown in the weekly grid, heatmap and today panel. Each has a
             "met" rule, so a day is done / not met / no data.
  CONTEXT    daily inputs with no goal (free hours, day type, ...). They are
             logged like habits but never get a grid cell or a "met" state.
  computed   log_streak() and deep_work_days_prev7() are worked out from
             existing entries, so nothing is logged for them.

An input is one of three types:
  number   free numeric input (min / max / step / cast)
  check    one-tap toggle, True = done
  choice   dropdown. "choices" is a list of (value, label) pairs.

Keys and fields of the original habits (study, study_hours, caffeine_mg, ...) are
unchanged on purpose: keys appear in URLs and fields are database columns.
"""

from datetime import timedelta

# Caffeine is logged on a 0-10 level scale: 1 level = 40 mg, so 10 = 400 mg.
# Keep this equal to MG_PER_POINT in ml/dummydataset.py.
CAFFEINE_MG_PER_LEVEL = 40
CAFFEINE_LIMIT_MG = 300
CAFFEINE_LEVELS = [(n, f"{n} ({n * CAFFEINE_MG_PER_LEVEL} mg)") for n in range(0, 11)]

DEEP_WORK_GOAL = 3        # hours, same as the "Deep work 3h+" habit
STREAK_CAP = 14           # same cap as ml/dummydataset.py

# ----------------------------- goal habits ----------------------------------
HABITS = [
    {
        "key": "study",                  # keep: used in URLs and the database
        "name": "Deep work 3h+",
        "label": "Deep work hours",
        "field": "study_hours",           # keep: database column
        "unit": "hours",
        "min": 0, "max": 24, "step": 0.5,
        "cast": float,
        "color": "#6f42c1",
        "met": lambda v: v >= DEEP_WORK_GOAL,
    },
    {
        "key": "caffeine",
        "name": "Caffeine under 300mg",
        "label": "Caffeine level",
        "field": "caffeine_mg",           # keep: database column (stores a 0-10 level)
        "type": "choice",
        "choices": CAFFEINE_LEVELS,
        "unit": "level 0-10, 1 level = 40 mg",
        "cast": int,
        "color": "#6c757d",
        # level 7 = 280 mg is met, level 8 = 320 mg is not
        "met": lambda v: v * CAFFEINE_MG_PER_LEVEL < CAFFEINE_LIMIT_MG,
    },
    {
        "key": "stress", "name": "Stress 5 or lower", "label": "Stress level",
        "field": "stress_level", "unit": "out of 10",
        "min": 1, "max": 10, "step": 1, "cast": int,
        "color": "#6c757d", "met": lambda v: v <= 5,
    },
    {
        "key": "sessions", "name": "2+ deep work sessions", "label": "Deep work sessions",
        "field": "study_sessions", "unit": "sessions",
        "min": 0, "max": 50, "step": 1, "cast": int,
        "color": "#0dcaf0", "met": lambda v: v >= 2,
    },
    {
        "key": "journal",
        "name": "Journaling",
        "type": "check",
        "field": "journaled",
        "color": "#6610f2",
        "met": lambda v: v is True,
    },
    {
        "key": "sleep", "name": "Sleep 7h+", "label": "Sleep hours",
        "field": "sleep_hours", "unit": "hours",
        "min": 0, "max": 24, "step": 0.5, "cast": float,
        "color": "#198754", "met": lambda v: v >= 7,
    },
    {
        "key": "disruptions", "name": "Disruptions 2 or fewer", "label": "Disruptions",
        "field": "disruptions", "unit": "meetings or interruptions",
        "min": 0, "max": 20, "step": 1, "cast": int,
        "color": "#fd7e14", "met": lambda v: v <= 2,
    },
    {
        "key": "junk", "name": "No junk food", "label": "Junk food",
        "field": "junk_food", "type": "choice",
        "choices": [(0, "No junk food today"), (1, "Had junk food")],
        "cast": int,
        "color": "#dc3545", "met": lambda v: v == 0,
    },
]

# ------------------------- context inputs (no goal) -------------------------
CONTEXT = [
    {
        "key": "free_hours", "name": "Free hours", "label": "Free hours available",
        "field": "free_hours", "unit": "hours free for work",
        "min": 0, "max": 24, "step": 0.5, "cast": float,
        "color": "#adb5bd",
    },
    {
        "key": "difficulty", "name": "Task difficulty", "label": "Task difficulty",
        "field": "task_difficulty", "type": "choice",
        "choices": [(1, "1 - Very easy"), (2, "2 - Easy"), (3, "3 - Medium"),
                    (4, "4 - Hard"), (5, "5 - Very hard")],
        "cast": int, "color": "#adb5bd",
    },
    {
        "key": "day_type", "name": "Day type", "label": "Day type",
        "field": "day_type", "type": "choice",
        "choices": [("class", "Class day"), ("free", "Free day"),
                    ("deadline", "Exam or deadline day")],
        "cast": str, "color": "#adb5bd",
    },
    {
        "key": "weather", "name": "Weather", "label": "Weather",
        "field": "weather", "type": "choice",
        "choices": [("sunny", "Sunny"), ("cloudy", "Cloudy"), ("rainy", "Rainy")],
        "cast": str, "color": "#adb5bd",
    },
]

# Everything the /habit/<key> log page can open.
BY_KEY = {h["key"]: h for h in HABITS + CONTEXT}

# ------------------------------ goal logic ----------------------------------
def is_met(habit, log):
    """True = goal met, False = logged but not met, None = no data."""
    value = getattr(log, habit["field"], None)
    return None if value is None else bool(habit["met"](value))


def has_data(log):
    """True if at least one goal habit was logged on this day."""
    return log is not None and any(is_met(h, log) is not None for h in HABITS)


# --------------------------- computed ML features ---------------------------
# logs_by_date: dict of date -> log row for one user.
def log_streak(logs_by_date, today, cap=STREAK_CAP):
    """Consecutive logged days ending YESTERDAY (today is excluded), capped."""
    n, day = 0, today - timedelta(days=1)
    while n < cap and has_data(logs_by_date.get(day)):
        n += 1
        day -= timedelta(days=1)
    return n


def deep_work_days_prev7(logs_by_date, today):
    """How many of the previous 7 days (today excluded) met the deep work goal."""
    study = BY_KEY["study"]
    return sum(
        1 for i in range(1, 8)
        if logs_by_date.get(today - timedelta(days=i)) is not None
        and is_met(study, logs_by_date[today - timedelta(days=i)]) is True
    )