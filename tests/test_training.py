import numpy as np
import pandas as pd
import pytest

from shared.schemas import FEATURE_COLUMNS
from training.train import prepare_split


def reviewed_frame():
    rows = []
    for event in range(8):
        for label in ("Normal", "High Risk"):
            rows.append({**dict.fromkeys(FEATURE_COLUMNS, 0.0), "risk": label,
                         "event_id": f"event-{event}", "reviewer": "test-reviewer",
                         "rights_approved": True, "permission_evidence": "synthetic test fixture only",
                         "motion_units": "pixels_per_second"})
    return pd.DataFrame(rows)


def test_event_groups_do_not_cross_split():
    frame = reviewed_frame()
    _, train, test, groups = prepare_split(frame)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    assert set(frame.risk.iloc[train]) == set(frame.risk.iloc[test])


def test_uncleared_rights_block_training():
    frame = reviewed_frame()
    frame.loc[0, "rights_approved"] = False
    with pytest.raises(ValueError, match="rights_approved"):
        prepare_split(frame)


def test_one_event_per_class_is_not_a_valid_holdout():
    frame = reviewed_frame()
    frame["event_id"] = frame.risk
    with pytest.raises(ValueError, match="independent event"):
        prepare_split(frame)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_invalid_features_block_training(value):
    frame = reviewed_frame()
    frame.loc[0, "average_velocity"] = value
    with pytest.raises(ValueError):
        prepare_split(frame)


def test_wrong_motion_units_block_training():
    frame = reviewed_frame()
    frame.loc[0, "motion_units"] = "pixels_per_frame"
    with pytest.raises(ValueError, match="consistent"):
        prepare_split(frame)
