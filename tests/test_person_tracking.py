"""Tracker reuse keeps recorded-video and live edge motion units aligned."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from edge.person_tracking import PersonTracker


class Tracked:
    def __init__(self, x):
        self.xyxy = np.array([[x, 0, x + 10, 10]], dtype=float)
        self.tracker_id = np.array([7])

    def __len__(self):
        return 1


def test_tracker_velocity_uses_sample_interval(monkeypatch):
    class ByteTrack:
        def __init__(self):
            self.positions = iter([Tracked(0), Tracked(2)])

        def update_with_detections(self, detections):
            return next(self.positions)

    class Detections:
        @staticmethod
        def from_ultralytics(result):
            return result

    monkeypatch.setitem(sys.modules, "supervision", SimpleNamespace(
        ByteTrack=ByteTrack, Detections=Detections
    ))
    tracker = PersonTracker()
    count, vectors = tracker.update(object(), 0.2)
    assert count == 1 and vectors == []
    count, vectors = tracker.update(object(), 0.2)
    assert count == 1
    assert vectors[0] == pytest.approx((10.0, 0.0))
