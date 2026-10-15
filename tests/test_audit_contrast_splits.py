import numpy as np

from scripts.audit_contrast_splits import enhance, possible_splits


def test_enhance_preserves_image_shape_and_type():
    image = np.full((32, 40, 3), 80, dtype=np.uint8)
    result = enhance(image)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_possible_splits_require_two_smaller_children():
    parents = np.array([[0, 0, 100, 100]], dtype=float)
    children = np.array([[0, 0, 40, 40], [50, 50, 90, 90]], dtype=float)
    assert possible_splits(parents, children) == [(0, [0, 1])]
    assert possible_splits(parents, children[:1]) == []
