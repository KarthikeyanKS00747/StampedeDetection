import csv

import pytest

from review.head_labeler import HeadAnnotation
from scripts.evaluate_person_counts import evaluate
from test_head_labeler import annotation, make_store


def test_evaluation_needs_reviewed_labels(tmp_path, monkeypatch):
    store, _ = make_store(tmp_path, monkeypatch)
    comparison = tmp_path / "comparison.csv"
    row = {"video": "train.mp4", "frame": "0", "full640_boxes": "1",
           "full1280_boxes": "2", "tiled_boxes": "3"}
    with comparison.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="No reviewed frames"):
        evaluate(store, [comparison])
    store.save(store.frames[0]["id"], annotation([[5, 5], [15, 15]]))
    result = evaluate(store, [comparison])
    assert result[0]["reviewed_visible_heads"] == 2
    assert result[0]["full640_absolute_error"] == 1
    assert result[0]["full1280_absolute_error"] == 0
    assert result[0]["tiled_absolute_error"] == 1


def test_uncountable_is_excluded_not_zero(tmp_path, monkeypatch):
    store, _ = make_store(tmp_path, monkeypatch)
    store.save(store.frames[0]["id"], HeadAnnotation(
        expected_revision=0, status="uncountable", reviewer="reviewer",
        note="Too blurred to count"))
    comparison = tmp_path / "comparison.csv"
    row = {"video": "train.mp4", "frame": "0", "full640_boxes": "0",
           "full1280_boxes": "0", "tiled_boxes": "0"}
    with comparison.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="No reviewed frames"):
        evaluate(store, [comparison])
