import numpy as np

from scripts.compare_person_resolution import (nonmaximum_suppression,
                                               owned_interval, tile_starts)


def test_tiling_covers_right_edge_without_duplicates():
    assert tile_starts(3840, 1024, 0.2)[-1] == 3840 - 1024
    assert tile_starts(500, 1024, 0.2) == [0]


def test_nms_removes_duplicate_but_keeps_separate_person():
    boxes = np.array([[10, 10, 30, 30], [11, 11, 31, 31], [50, 50, 70, 70]],
                     dtype=np.float32)
    scores = np.array([0.9, 0.7, 0.8], dtype=np.float32)
    assert nonmaximum_suppression(boxes, scores).tolist() == [0, 2]


def test_ownership_partitions_tile_overlap():
    starts = tile_starts(1920, 1024, 0.2)
    first = owned_interval(starts, 0, 1024, 1920)
    second = owned_interval(starts, 1, 1024, 1920)
    last = owned_interval(starts, len(starts) - 1, 1024, 1920)
    assert first[0] == 0
    assert first[1] == second[0]
    assert second[1] == last[0]
    assert last[1] == 1920
