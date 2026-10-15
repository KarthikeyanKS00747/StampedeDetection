"""The fusion contract is testable without replaying training footage."""

import numpy as np
import pytest

from classifier.combined_detector import combine_windows, process_frames
from classifier.fusion import FusionPolicy
from shared.schemas import ClassifierMode, RiskLevel


class AlwaysAbnormal:
    classes_ = np.array([0, 1])

    def predict(self, rows):
        return np.ones(len(rows), dtype=int)

    def predict_proba(self, rows):
        return np.tile([0.1, 0.9], (len(rows), 1))


def frames(count=60):
    return [dict(frame_idx=index, timestamp_s=index / 25, count=3,
                 mean_iou=0.1, motion=2.0, frame_name=f"scene-{index:05}.png")
            for index in range(count)]


def test_agreement_opens_only_after_three_abnormal_windows():
    samples = [(index, RiskLevel.HIGH) for index in range(0, 60, 15)]
    rows = combine_windows(frames(), samples, AlwaysAbnormal(), fps=25)
    assert len(rows) == 3
    assert [row["combined_candidate"] for row in rows] == [False, False, True]
    assert rows[-1]["combined_episode_started"] is True
    assert rows[-1]["combined_episode_id"] == 1
    assert rows[-1]["window_end_s"] == pytest.approx(59 / 25)


def test_missing_or_normal_baseline_cannot_create_candidate():
    rows = combine_windows(frames(), [(0, RiskLevel.NORMAL)], AlwaysAbnormal(), 25)
    assert rows[-1]["anomaly_episode_active"] is True
    assert rows[-1]["baseline_risk"] == "unavailable"
    assert not any(row["combined_candidate"] for row in rows)


def test_five_normal_windows_clear_anomaly_episode():
    policy = FusionPolicy()
    for frame in (0, 15, 30):
        decision = policy.observe(frame, True, RiskLevel.CRITICAL)
    assert decision.combined_candidate
    for frame in (45, 60, 75, 90):
        assert policy.observe(frame, False, RiskLevel.CRITICAL).combined_candidate
    decision = policy.observe(105, False, RiskLevel.CRITICAL)
    assert not decision.anomaly_episode_active
    assert not decision.combined_candidate


def test_gap_breaks_streak_and_repeated_window_is_rejected():
    policy = FusionPolicy()
    policy.observe(0, True, RiskLevel.HIGH)
    policy.observe(15, True, RiskLevel.HIGH)
    assert not policy.observe(45, True, RiskLevel.HIGH).anomaly_episode_active
    with pytest.raises(ValueError):
        policy.observe(45, True, RiskLevel.HIGH)


class FakeTensor:
    def cpu(self):
        return self

    def numpy(self):
        return np.array([[0.0, 0.0, 10.0, 10.0]])


class FakeDetector:
    def __init__(self):
        self.results = []

    def __call__(self, frame, **kwargs):
        assert kwargs == {"classes": [0], "verbose": False}
        result = type("Result", (), {"boxes": type("Boxes", (), {"xyxy": FakeTensor()})()})()
        self.results.append(result)
        return [result]


class FakeTracker:
    def __init__(self):
        self.results = []

    def update(self, result, delta_s):
        self.results.append(result)
        assert delta_s > 0
        return 1, []


class HighRisk:
    def assess(self, metrics):
        return RiskLevel.HIGH, {}, ["test high risk"], ClassifierMode.HEURISTIC


def test_single_detector_pass_feeds_both_signals_and_logs_processing():
    detector = FakeDetector()
    tracker = FakeTracker()
    source = (
        (index, np.zeros((40, 40, 3), dtype=np.uint8), f"frame-{index}", 0.0)
        for index in range(60)
    )
    rows = list(process_frames(
        source, 25, 5, 25, detector, tracker, AlwaysAbnormal(), HighRisk(),
        clock=lambda: 10.0,
    ))
    assert len(detector.results) == 60
    assert len(tracker.results) == 12
    assert all(result in detector.results for result in tracker.results)
    assert [row["combined_candidate"] for row in rows] == [False, False, True]
    assert rows[-1]["N_mean"] == 1
    assert rows[-1]["baseline_count"] == 1
    assert rows[-1]["window_processing_wall_s"] == 10.0
    assert rows[-1]["end_frame_processing_wall_s"] == 10.0
