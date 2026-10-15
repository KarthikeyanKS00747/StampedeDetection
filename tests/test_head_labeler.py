import csv
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from classifier.density_model import read_manifest
from review import head_labeler


def make_store(tmp_path, monkeypatch):
    monkeypatch.setattr(head_labeler, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.csv"
    rows = []
    for name in ("train", "val"):
        path = tmp_path / f"{name}.jpg"
        assert cv2.imwrite(str(path), np.zeros((30, 40, 3), dtype=np.uint8))
        rows.append({"video": f"{name}.mp4", "frame": "0", "timestamp_s": "0",
                     "detected_people": "0", "original": path.name,
                     "detected": path.name})
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    store = head_labeler.HeadStore(manifest, tmp_path / "labels.json")
    return store, manifest


def annotation(points=None, revision=0):
    return head_labeler.HeadAnnotation(
        expected_revision=revision, status="reviewed", points=points or [],
        reviewer="tester", note="Visible head centres reviewed")


def test_annotation_validation_revision_and_scene_export(tmp_path, monkeypatch):
    store, _ = make_store(tmp_path, monkeypatch)
    first, second = [row["id"] for row in store.frames]
    assert store.save(first, annotation([[10, 10]]))["revision"] == 1
    with pytest.raises(head_labeler.HTTPException) as changed:
        store.save(first, annotation([[11, 11]]))
    assert changed.value.status_code == 409
    with pytest.raises(head_labeler.HTTPException) as outside:
        store.save(second, annotation([[400, 10]]))
    assert outside.value.status_code == 422
    store.save(second, head_labeler.HeadAnnotation(expected_revision=0))
    with pytest.raises(ValueError, match="both different-video splits"):
        store.export("val.mp4", tmp_path / "export")
    store.save(second, annotation([], revision=1))
    assert store.export("val.mp4", tmp_path / "export") == (1, 1)
    train = json.loads((tmp_path / "export/head_train.jsonl").read_text())
    val = json.loads((tmp_path / "export/head_val.jsonl").read_text())
    assert train["scene"] != val["scene"]
    assert train["points"] == [[10.0, 10.0]]
    assert val["points"] == []
    assert "not independently verified" in train["permission_evidence"]
    assert len(read_manifest(tmp_path / "export/head_train.jsonl")) == 1
    assert len(read_manifest(tmp_path / "export/head_val.jsonl")) == 1


def test_local_http_routes_and_image_safety(tmp_path, monkeypatch):
    _, manifest = make_store(tmp_path, monkeypatch)
    app = head_labeler.create_app(manifest, tmp_path / "web_labels.json")
    client = TestClient(app)
    response = client.get("/api/head-frames")
    assert response.status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/static/head_labeler.js").status_code == 200
    html = client.get("/").text
    assert 'id="download"' in html
    assert 'id="rights"' not in html
    assert 'id="permission"' not in html
    frames = response.json()["frames"]
    assert len(frames) == 2
    assert client.get(f"/media/{frames[0]['id']}").status_code == 200
    assert client.get("/media/not-listed").status_code == 404
    assert client.get("/", headers={"host": "untrusted.example"}).status_code == 403
    saved = client.post(f"/api/head-labels/{frames[0]['id']}",
                        json=annotation([[5, 6]]).model_dump())
    assert saved.status_code == 200
    assert saved.json()["points"] == [[5.0, 6.0]]
    draft = client.post(f"/api/head-labels/{frames[1]['id']}",
                        json={"expected_revision": 0, "status": "draft", "points": []})
    assert draft.status_code == 200
    assert (tmp_path / "web_labels.json").is_file()
