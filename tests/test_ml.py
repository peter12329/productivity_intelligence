"""ML tests. Run from the project root:  python -m pytest tests/test_ml.py -q

Needs the dataset and trained models (python -m ml.dummydataset --rows 10000 && python -m ml.train).
"""
import numpy as np
import pandas as pd
import pytest

from ml.predict import data_stage, predict_day
from ml.settings import CLASSIFICATION_TARGET, FEATURES, REGRESSION_TARGET
from ml.train import GROUP_COL, load_data, split_data

GOOD = {"study_hours": 4, "caffeine_mg": 3, "stress_level": 4, "study_sessions": 3, "journaled": 1}


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
    {k: v for k, v in GOOD.items() if k != "study_hours"},
    {**GOOD, "study_hours": "abc"},
    {**GOOD, "stress_level": 11},
    {**GOOD, "caffeine_mg": -1},
    {**GOOD, "study_hours": float("nan")},
])
def test_predict_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        predict_day(bad)


def test_more_study_never_gives_wildly_different_type():
    # sanity: output stays finite across the valid grid corners
    for s in (0, 12):
        for st in (1, 10):
            assert np.isfinite(predict_day({**GOOD, "study_hours": s, "stress_level": st})["predicted_score"])


def test_data_stage_does_not_claim_a_personal_model():
    assert [data_stage(n) for n in (0, 7, 20)] == ["general", "descriptive", "enough_for_trends"]
