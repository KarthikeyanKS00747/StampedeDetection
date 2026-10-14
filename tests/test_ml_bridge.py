import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from fog.anomaly import AnomalyClassifier, NAMES
from fog.app import app, state
from fog.classifier import RiskClassifier
from scripts.replay_ml import windows
from training.train_behaviour import prepare

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/training/quality_motion_2026_10_14_run04"


@pytest.fixture
def ml(monkeypatch):
    if not (RUN / "quality_joint.joblib").exists():
        pytest.skip("Locally generated research artifact unavailable")
    monkeypatch.setattr(state, "anomaly_classifier", AnomalyClassifier(RUN / "quality_joint.joblib"))
    monkeypatch.setattr(state, "anomaly_latest", {})
    return TestClient(app)


def test_all_replay_windows_match_offline_model(ml, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Research ML cannot invoke hardware or risk alerts")
    monkeypatch.setattr(state.gpio, "set_risk", forbidden)
    monkeypatch.setattr(state.alerts, "evaluate", forbidden)
    expected = json.loads((RUN / "scores_joint.json").read_text())
    actual = []
    for window in windows(RUN / "features_quality.json"):
        response = ml.post("/api/ml/windows", json=window.model_dump())
        assert response.status_code == 200
        actual.append(response.json())
    assert len(actual) == len(expected) == 70
    for result, score in zip(actual, expected):
        assert result["hardware_alarm_enabled"] is False
        assert result["risk"] is None
        if score["score"] is None:
            assert result["score"] is None
            assert result["observation"] == "Cannot assess"
        else:
            assert result["score"] == pytest.approx(score["score"])
            assert result["exceeds_reference_threshold"] == score["exceeds_reference_threshold"]
    assert len(ml.get("/api/ml/status").json()) == 4


def test_invalid_contract_and_nonfinite_rejected(ml):
    row = next(windows(RUN / "features_quality.json")).model_dump()
    row["feature_names"] = list(reversed(NAMES))
    assert ml.post("/api/ml/windows", json=row).status_code == 422
    row["feature_names"] = NAMES
    row["end_s"] = row["start_s"]
    assert ml.post("/api/ml/windows", json=row).status_code == 422
    row["end_s"] += 1
    row["features"][0] = "NaN"
    assert ml.post("/api/ml/windows", json=row).status_code == 422


def test_disabled_ml_has_no_silent_heuristic_fallback(monkeypatch):
    monkeypatch.setattr(state, "anomaly_classifier", None)
    row = dict(zone_id="test", clip_id="test", start_s=0, end_s=1,
               feature_names=NAMES, features=[0] * 9, quality_eligible=True)
    assert TestClient(app).post("/api/ml/windows", json=row).status_code == 503


def test_risk_loader_rejects_anomaly_bundle(ml):
    with pytest.raises(ValueError, match="not four-tier"):
        RiskClassifier(str(RUN / "quality_joint.joblib"))


def test_ml_websocket(ml):
    with ml.websocket_connect("/ws") as socket:
        assert socket.receive_json()["type"] == "init"
        row = next(windows(RUN / "features_quality.json")).model_dump()
        assert ml.post("/api/ml/windows", json=row).status_code == 200
        assert socket.receive_json()["type"] == "anomaly"


def reviewed_fixture():
    # Synthetic rows test validation/splitting only, never train project weights.
    return {"schema": "reviewed_behaviour_features_v1", "feature_names": NAMES,
            "rows": [dict(window_id=f"{event}-{label}", event_id=f"event-{event}",
                          reviewer="test", review_note="test-only", permission_evidence="synthetic fixture",
                          rights_approved=True, quality_eligible=True, behaviour_label=label,
                          features=[float(event)] * 9)
                     for event in range(6) for label in ("walking", "running")]}


def test_behaviour_split_never_leaks_events():
    x, y, groups, train, test = prepare(reviewed_fixture())
    assert not set(groups[train]) & set(groups[test])
    assert set(y[train]) == set(y[test]) == {"walking", "running"}


@pytest.mark.parametrize("fault", ["empty", "rights", "duplicate", "one_event", "nan"])
def test_behaviour_rejects_unusable_data(fault):
    data = copy.deepcopy(reviewed_fixture())
    if fault == "empty":
        data["rows"] = []
    elif fault == "rights":
        data["rows"][0]["rights_approved"] = False
    elif fault == "duplicate":
        data["rows"][1]["window_id"] = data["rows"][0]["window_id"]
    elif fault == "one_event":
        for row in data["rows"]:
            row["event_id"] = "same"
    else:
        data["rows"][0]["features"][0] = float("nan")
    with pytest.raises(ValueError):
        prepare(data)
