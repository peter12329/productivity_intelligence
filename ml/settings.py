from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "datasets" / "productivity_logs.csv"
ARTIFACT_DIR = ROOT / "artifacts"

RANDOM_STATE = 42

FEATURES = [
    "study_hours",
    "caffeine_mg",
    "stress_level",
    "study_sessions",
    "journaled",
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