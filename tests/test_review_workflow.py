import asyncio
import json

import httpx
import pytest

from review.app import Annotation, ReviewStore, create_app


@pytest.fixture
def package():
    return {"feature_names": ["proxy"], "clips": {}, "windows": [
        {"id": "one", "clip_id": "a", "event_group": "event-a", "start_s": 0,
         "end_s": 1, "quality_eligible": True, "features": [0.1]},
        {"id": "two", "clip_id": "a", "event_group": "event-a", "start_s": 1,
         "end_s": 2, "quality_eligible": False, "features": [0.1]}]}


def approved():
    return Annotation(expected_revision=0, status="reviewed", behaviour_label="walking",
                      reviewer="Synthetic test reviewer", note="Synthetic fixture, not source footage",
                      rights_approved=True, permission_evidence="Synthetic test fixture")


def test_review_and_permission_and_quality_are_separate_gates(tmp_path, package):
    store = ReviewStore(package, tmp_path / "annotations.json")
    draft = Annotation(expected_revision=0, behaviour_label="walking")
    store.save("one", draft)
    assert store.export()["rows"] == []
    review = approved()
    review.expected_revision = 1
    store.save("one", review)
    store.save("two", approved())
    rows = store.export()["rows"]
    assert len(rows) == 1 and rows[0]["window_id"] == "one"
    assert not store.export()["risk_labels_created"]


def test_stale_revision_cannot_overwrite(tmp_path, package):
    store = ReviewStore(package, tmp_path / "annotations.json")
    store.save("one", Annotation(expected_revision=0))
    with pytest.raises(Exception) as error:
        store.save("one", Annotation(expected_revision=0))
    assert error.value.status_code == 409


def test_review_without_observation_is_rejected(tmp_path, package):
    store = ReviewStore(package, tmp_path / "annotations.json")
    with pytest.raises(Exception) as error:
        store.save("one", Annotation(expected_revision=0, status="reviewed"))
    assert error.value.status_code == 422


def test_api_denies_cross_origin_writes_and_arbitrary_media(tmp_path, package):
    package_path = tmp_path / "package.json"
    package_path.write_text(json.dumps(package))
    app = create_app(package_path, tmp_path / "annotations.json")

    async def checks():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            assert (await client.get('/api/review')).status_code == 200
            assert (await client.get('/media/not-a-known-clip')).status_code == 404
            response = await client.post('/api/annotations/one', json={"expected_revision": 0}, headers={"Origin": "https://example.com"})
            assert response.status_code == 403
            assert (await client.get('/api/export')).json()["rows"] == []
    asyncio.run(checks())
