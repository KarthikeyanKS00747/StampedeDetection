"""The audit never invents measurements for missing model branches."""

from types import SimpleNamespace

import pytest
import torch

from classifier.density_multisignal import FIELDS, optional_yolo, pose_count


def test_missing_optional_weights_are_not_treated_as_zero(tmp_path):
    assert optional_yolo(None, "detect") is None
    with pytest.raises(FileNotFoundError, match="weights not found"):
        optional_yolo(tmp_path / "missing.pt", "detect")
    assert "head_boxes" in FIELDS
    assert "density_estimated_people" in FIELDS


def test_pose_requires_two_confident_keypoints():
    confidences = torch.tensor([[0.8, 0.7, 0.1], [0.9, 0.2, 0.1]])
    result = SimpleNamespace(keypoints=SimpleNamespace(conf=confidences))
    assert pose_count(result, 0.5) == 1
    assert pose_count(SimpleNamespace(keypoints=None), 0.5) == 0
