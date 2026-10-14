import pytest

from edge.motion import frame_time, velocities


def test_motion_rate_is_independent_of_sampling_interval():
    assert velocities([(2, 0)], 0.1) == velocities([(4, 0)], 0.2) == [(20, 0)]


def test_stationary_flow_is_retained():
    assert velocities([(0, 0)], 0.2) == [(0, 0)]


@pytest.mark.parametrize("delta", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_is_rejected(delta):
    with pytest.raises(ValueError):
        velocities([(1, 2)], delta)


def test_video_time_uses_media_frames():
    assert frame_time(30, 30) == 1
    assert frame_time(150, 30) == 5
