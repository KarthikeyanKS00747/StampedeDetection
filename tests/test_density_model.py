"""Checks for the experimental, independently supervised density model."""

import argparse
import json

import cv2
import numpy as np
import pytest
import torch

from classifier.density_model import (DensityNet, HeadPointDataset, density_target,
                                      letterbox, predict, read_manifest, train)


def test_density_target_preserves_annotated_count_at_edges():
    points = [[0, 0], [99, 49], [50, 25]]
    target = density_target(points, 100, 50, 5.12, 0, 128)
    assert target.shape == (1, 128, 128)
    assert target.sum().item() == pytest.approx(3.0, abs=1e-5)


def test_model_produces_nonnegative_density():
    model = DensityNet()
    with torch.no_grad():
        result = model(torch.zeros(1, 3, 64, 64))
    assert result.shape == (1, 1, 16, 16)
    assert torch.all(result >= 0)


def test_annotated_image_loads_and_count_is_preserved(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((50, 100, 3), dtype=np.uint8))
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(json.dumps({"image": "frame.jpg", "scene": "camera-1",
                                    "points": [[5, 5], [90, 40]]}) + "\n")
    image, target = HeadPointDataset(read_manifest(manifest))[0]
    assert image.shape == (3, 512, 512)
    assert target.sum().item() == pytest.approx(2.0, abs=1e-5)


def test_letterbox_preserves_aspect_ratio():
    canvas, scale, left, top = letterbox(np.zeros((50, 100, 3), dtype=np.uint8))
    assert canvas.shape == (512, 512, 3)
    assert scale == pytest.approx(5.12)
    assert (left, top) == (0, 128)


def test_train_rejects_same_scene(tmp_path):
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    row = json.dumps({"image": "frame.jpg", "scene": "same-video", "points": []})
    train_manifest = tmp_path / "train.jsonl"
    val_manifest = tmp_path / "val.jsonl"
    train_manifest.write_text(row + "\n")
    val_manifest.write_text(row + "\n")
    args = argparse.Namespace(train=train_manifest, val=val_manifest,
                              output=tmp_path / "model.pt", epochs=1,
                              batch_size=1, lr=1e-4, seed=42, device="cpu")
    with pytest.raises(ValueError, match="scenes overlap"):
        train(args)


def test_train_and_predict_smoke(tmp_path):
    manifests = []
    for name, count in (("train", 1), ("val", 0)):
        image_path = tmp_path / f"{name}.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
        manifest = tmp_path / f"{name}.jsonl"
        manifest.write_text(json.dumps({"image": image_path.name, "scene": name,
                                        "points": [[16, 16]] if count else []}) + "\n")
        manifests.append(manifest)
    checkpoint = tmp_path / "best.pt"
    train(argparse.Namespace(train=manifests[0], val=manifests[1],
                             output=checkpoint, epochs=1, batch_size=1,
                             lr=1e-4, seed=42, device="cpu"))
    overlay = tmp_path / "overlay.jpg"
    predict(argparse.Namespace(weights=checkpoint, image=tmp_path / "val.jpg",
                               output=overlay))
    assert checkpoint.is_file()
    assert overlay.is_file()
