import numpy as np
import pytest

from training.motion_baseline import describe, frame_changes


def test_relative_speed_uses_time_and_image_diagonal():
    flow = np.full((8, 8, 2), [2.0, 0.0], dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    values = describe(flow, .2, mask, False)
    assert values[0] == pytest.approx(10 / np.hypot(8, 8))
    assert values[3] == 1
    assert values[4] == pytest.approx(0)


def test_residual_proxy_erases_coherent_motion_so_cannot_be_risk_truth():
    flow = np.full((8, 8, 2), [2.0, 0.0], dtype=np.float32)
    values = describe(flow, .2, np.ones((8, 8), dtype=bool), True)
    assert values == [0, 0, 0, 0, 0]


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        describe(np.zeros((8, 8, 2)), .2, np.zeros((8, 8), dtype=bool), False)


def test_moving_pixels_are_not_automatically_global_lighting_changes():
    previous = np.array([[20, 220]], dtype=np.uint8)
    gray = np.array([[220, 20]], dtype=np.uint8)
    pixel_change, lighting = frame_changes(gray, previous, np.ones_like(gray, dtype=bool))
    assert pixel_change > .7
    assert lighting == 0
