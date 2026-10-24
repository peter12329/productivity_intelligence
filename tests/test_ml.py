"""ML tests. Run from the project root:  python -m pytest tests/test_ml.py -q

Needs the dataset and trained models (python -m ml.dummydataset --rows 10000 && python -m ml.train).
"""
import numpy as np
import pytest

from ml.predict import data_stage, predict_day
from ml.settings import CLASSIFICATION_TARGET, FEATURES, REGRESSION_TARGET
from ml.train import GROUP_COL, load_data, split_data

# One valid value for every name in ml/settings.py FEATURES.
GOOD = {
    "deep_work": 4, "sleep_hours": 7.5, "caffeine_level": 3, "stress_level": 4,
    "journaled": 1, "log_streak": 3, "disruptions": 1, "junk_food": 0,
    "task_difficulty": 3, "deep_work_days_prev7": 3, "day_type": "class",
    "weather": "sunny", "free_hours": 6,
}


def test_good_input_covers_every_feature():
    assert set(GOOD) == set(FEATURES)


def test_targets_never_in_features():
    assert not {REGRESSION_TARGET, CLASSIFICATION_TARGET, "productive_day",
                "previous_day_productivity"} & set(FEATURES)


def test_split_holds_out_whole_people():
    train, test = split_data(load_data())
    assert not set(train[GROUP_COL]) & set(test[GROUP_COL])
    assert len(test) > 0


def test_split_is_deterministic():
    a = split_data(load_data())[1][GROUP_COL].unique()
    b = split_data(load_data())[1][GROUP_COL].unique()
    assert list(a) == list(b)


def test_predict_output_shape_and_labels():
    r = predict_day(GOOD)
    assert 0 <= r["predicted_score"] <= 100
    assert r["predicted_level"] in ("low", "medium", "high")
    assert abs(sum(r["level_probabilities"].values()) - 1) < 0.01
    assert r["experimental"] is True and "synthetic" in r["disclaimer"]


@pytest.mark.parametrize("bad", [
    {k: v for k, v in GOOD.items() if k != "deep_work"},
    {**GOOD, "deep_work": "abc"},
    {**GOOD, "stress_level": 11},
    {**GOOD, "caffeine_level": -1},
    {**GOOD, "deep_work": float("nan")},
    {**GOOD, "weather": "snowy"},
])
def test_predict_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        predict_day(bad)


def test_output_stays_finite_at_the_corners_of_the_valid_grid():
    for s in (0, 12):
        for st in (1, 10):
            assert np.isfinite(predict_day({**GOOD, "deep_work": s, "stress_level": st})["predicted_score"])


def test_data_stage_does_not_claim_a_personal_model():
    assert [data_stage(n) for n in (0, 7, 20)] == ["general", "descriptive", "enough_for_trends"]
