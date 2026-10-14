import numpy as np
import pytest

from training.quality_motion_baseline import describe_content, mask_unreliable_scores


def test_no_supported_windows_means_indeterminate_not_safe_or_anomalous():
    rows = [{"quality_warning": True}]
    scores = [{"clip_id": "a", "quality_warning": True, "score": .7, "exceeds_reference_threshold": True}]
    fit = {"per_clip": {"a": {}}}
    mask_unreliable_scores(rows, scores, fit)
    assert scores[0]["score"] is None
    assert scores[0]["exceeds_reference_threshold"] is None
    assert fit["per_clip"]["a"]["threshold_exceedance_fraction"] is None
    assert fit["per_clip"]["a"]["indeterminate_windows"] == 1


def test_exceedance_fraction_uses_only_eligible_windows():
    rows = [{"quality_warning": False}, {"quality_warning": True}]
    scores = [{"clip_id": "a", "quality_warning": False, "score": .7, "exceeds_reference_threshold": True},
              {"clip_id": "a", "quality_warning": True, "score": .1, "exceeds_reference_threshold": False}]
    fit = {"per_clip": {"a": {}}}
    mask_unreliable_scores(rows, scores, fit)
    assert fit["per_clip"]["a"]["threshold_exceedance_fraction"] == 1
    assert fit["per_clip"]["a"]["eligible_windows"] == 1


def test_black_padding_does_not_change_normalised_speed():
    flow = np.ones((10, 20, 2), dtype=float)
    mask = np.ones((10, 20), dtype=bool)
    padded_flow = np.pad(flow, ((0, 0), (10, 10), (0, 0)))
    padded_mask = np.pad(mask, ((0, 0), (10, 10)))
    assert describe_content(flow, .2, mask) == pytest.approx(describe_content(padded_flow, .2, padded_mask))
