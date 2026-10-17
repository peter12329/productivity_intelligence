from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "datasets" / "productivity_logs.csv"
ARTIFACT_DIR = ROOT / "artifacts"

RANDOM_STATE = 42

FEATURES = [
       "deep_work", "sleep_hours", "caffeine_level", "stress_level",
       "journaled", "log_streak", "disruptions", "junk_food",
       "task_difficulty", "deep_work_days_prev7", "day_type", "weather",
       "free_hours",
]

REGRESSION_TARGET = "productivity_score"
CLASSIFICATION_TARGET = "productivity_level"

LEVELS = ["low", "medium", "high"]
LEVEL_BINS = [-0.1, 45, 70, 100.1]

FEATURE_RANGES = {
    "study_hours": (0, 24),
    "caffeine_mg": (0, 10),
    "stress_level": (1, 10),
    "study_sessions": (0, 50),
    "journaled": (0, 1),
}

N_CLUSTERS_RANGE = range(2, 8)