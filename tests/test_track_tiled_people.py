import numpy as np

from scripts.track_tiled_people import detections


def test_detection_adapter_accepts_empty_and_scored_boxes():
    empty = detections(np.empty((0, 4)), np.empty(0))
    assert len(empty) == 0
    one = detections(np.array([[1, 2, 10, 20]]), np.array([0.8]))
    assert len(one) == 1
    assert one.confidence[0] == np.float32(0.8)
