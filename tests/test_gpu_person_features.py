import numpy as np
import pytest

from training.gpu_person_baseline import box_features, join_features, tile_regions


def test_overlapping_boxes_do_not_double_count_coverage():
    boxes = np.array([[0, 0, 50, 50], [0, 0, 50, 50]])
    coverage, local_count = box_features(boxes, (100, 100, 3))
    assert coverage == .25
    assert local_count == 2


def test_empty_boxes_have_zero_coverage():
    assert box_features(np.empty((0, 4)), (100, 100)) == (0, 0)


def test_join_rejects_missing_gpu_frames():
    with pytest.raises(ValueError, match="Missing"):
        join_features([{"clip_id": "a", "start_s": 0, "end_s": 1}], [])


def test_overlapping_tiles_cover_full_image():
    mask = np.zeros((100, 200), dtype=bool)
    for x1, y1, x2, y2 in tile_regions(mask.shape):
        mask[y1:y2, x1:x2] = True
    assert mask.all()
