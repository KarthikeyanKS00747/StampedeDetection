import cv2
import numpy as np
import pytest

from edge.video_quality import candidate_background, compensate, content_mask


def test_letterbox_is_excluded_without_erasing_black_clothing_inside():
    frame = np.zeros((100, 200), np.uint8)
    frame[:, 40:160] = 100
    frame[40:60, 80:120] = 0
    mask = content_mask(frame)
    assert not mask[:, :40].any()
    assert mask[50, 100]


def test_person_boxes_are_excluded_from_background_candidates():
    frame = np.full((100, 100), 100, np.uint8)
    mask = candidate_background(frame, [[40, 40, 60, 60]], (100, 100))
    assert mask[50, 50] == 0
    assert mask[10, 10] == 255


def test_known_camera_translation_is_compensated():
    rng = np.random.default_rng(42)
    previous = cv2.GaussianBlur(rng.integers(20, 230, (160, 240), dtype=np.uint8), (3, 3), 0)
    current = cv2.warpAffine(previous, np.array([[1., 0., 4.], [0., 1., 2.]]), (240, 160))
    corrected, overlap, info = compensate(previous, current, np.full(previous.shape, 255, np.uint8))
    assert info["accepted"]
    assert info["translation_px"] == pytest.approx(np.sqrt(20), abs=.3)
    central = overlap.copy()
    central[:15] = central[-15:] = False
    central[:, :15] = central[:, -15:] = False
    assert np.abs(corrected.astype(float) - previous.astype(float))[central].mean() < 2


def test_no_anchors_means_no_invented_correction():
    frame = np.full((100, 100), 100, np.uint8)
    corrected, overlap, info = compensate(frame, frame, np.zeros(frame.shape, np.uint8))
    assert corrected is None and not info["accepted"] and not overlap.any()
