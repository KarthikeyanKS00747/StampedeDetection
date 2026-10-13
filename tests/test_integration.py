"""Integration tests for the fog API: feature ingestion, status/history endpoints, and WebSocket."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from fog.app import app, state
from shared.schemas import RiskLevel


# ─── helpers ─────────────────────────────────────────────────────────────────

def _payload(**overrides) -> dict:
    base = dict(
        zone_id="zone-1",
        crowd_count=20,
        density=1.0,
        density_growth_rate=0.0,
        average_velocity=2.0,
        velocity_variance=1.0,
        direction_variance=0.0,
        congestion_index=0.1,
        source="simulation",
    )
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def reset_state():
    """Wipe in-memory state before each test to ensure isolation."""
    state.latest.clear()
    state.events.clear()
    state.history.clear()
    state.alerts._history.clear()
    state.alerts._active.clear()
    yield


client = TestClient(app)


# ─── Feature ingestion ────────────────────────────────────────────────────────

class TestFeatureIngestion:
    def test_valid_input_returns_200_with_assessment(self):
        r = client.post("/api/features", json=_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["zone_id"] == "zone-1"
        assert body["risk"] in (l.value for l in RiskLevel)
        assert isinstance(body["alert_active"], bool)
        assert body["classifier_mode"] in ("heuristic", "trained")

    def test_heuristic_mode_label(self):
        r = client.post("/api/features", json=_payload())
        assert r.json()["classifier_mode"] == "heuristic"

    def test_critical_input_classified_correctly(self):
        r = client.post("/api/features", json=_payload(
            density=5.0, density_growth_rate=0.7,
            direction_variance=0.8, congestion_index=0.9,
        ))
        assert r.status_code == 200
        assert r.json()["risk"] == RiskLevel.CRITICAL.value

    def test_normal_input_classified_correctly(self):
        r = client.post("/api/features", json=_payload())
        assert r.json()["risk"] == RiskLevel.NORMAL.value

    def test_invalid_negative_crowd_count_rejected(self):
        r = client.post("/api/features", json=_payload(crowd_count=-1))
        assert r.status_code == 422

    def test_invalid_density_growth_type_rejected(self):
        r = client.post("/api/features", json=_payload(density="bad"))
        assert r.status_code == 422

    def test_congestion_above_one_rejected(self):
        r = client.post("/api/features", json=_payload(congestion_index=1.5))
        assert r.status_code == 422

    def test_direction_variance_above_one_rejected(self):
        r = client.post("/api/features", json=_payload(direction_variance=1.1))
        assert r.status_code == 422

    def test_multiple_zone_ids_stored_independently(self):
        client.post("/api/features", json=_payload(zone_id="zone-1"))
        client.post("/api/features", json=_payload(zone_id="zone-2", crowd_count=50))
        r = client.get("/api/status")
        ids = {a["zone_id"] for a in r.json()}
        assert ids == {"zone-1", "zone-2"}


# ─── Status endpoint ──────────────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_empty_before_any_ingestion(self):
        r = client.get("/api/status")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_latest_per_zone(self):
        client.post("/api/features", json=_payload(crowd_count=10))
        client.post("/api/features", json=_payload(crowd_count=20))
        r = client.get("/api/status")
        assert r.status_code == 200
        zones = r.json()
        assert len(zones) == 1                      # still one zone
        assert zones[0]["metrics"]["crowd_count"] == 20   # latest value


# ─── History endpoint ─────────────────────────────────────────────────────────

class TestHistoryEndpoint:
    def test_empty_for_unknown_zone(self):
        r = client.get("/api/history/no-such-zone")
        assert r.status_code == 200
        assert r.json() == []

    def test_accumulates_assessments_in_order(self):
        for count in [10, 20, 30]:
            client.post("/api/features", json=_payload(crowd_count=count))
        r = client.get("/api/history/zone-1")
        counts = [a["metrics"]["crowd_count"] for a in r.json()]
        assert counts == [10, 20, 30]

    def test_history_scoped_to_zone(self):
        client.post("/api/features", json=_payload(zone_id="zone-A"))
        client.post("/api/features", json=_payload(zone_id="zone-B"))
        assert client.get("/api/history/zone-A").json() != []
        assert client.get("/api/history/zone-B").json() != []
        # zone-A history must not contain zone-B entries
        zones_in_a = {a["zone_id"] for a in client.get("/api/history/zone-A").json()}
        assert zones_in_a == {"zone-A"}


# ─── Alerts endpoint ──────────────────────────────────────────────────────────

class TestAlertsEndpoint:
    def test_no_alerts_initially(self):
        assert client.get("/api/alerts").json() == []

    def test_critical_input_creates_alert(self):
        client.post("/api/features", json=_payload(
            density=5.0, density_growth_rate=0.7,
            direction_variance=0.8, congestion_index=0.9,
        ))
        alerts = client.get("/api/alerts").json()
        assert len(alerts) >= 1
        assert alerts[0]["active"] is True

    def test_alert_clearance_logged(self):
        # raise then clear
        client.post("/api/features", json=_payload(
            density=5.0, density_growth_rate=0.7,
            direction_variance=0.8, congestion_index=0.9,
        ))
        client.post("/api/features", json=_payload())   # normal → clearance
        alerts = client.get("/api/alerts").json()
        active_flags = [a["active"] for a in alerts]
        assert False in active_flags                    # at least one clearance


# ─── WebSocket ────────────────────────────────────────────────────────────────

class TestWebSocket:
    def test_init_message_type_on_connect(self):
        with client.websocket_connect("/ws") as ws:
            raw = ws.receive_text()
        msg = json.loads(raw)
        assert msg["type"] == "init"
        assert isinstance(msg["data"], list)

    def test_assessment_broadcast_on_ingest(self):
        with client.websocket_connect("/ws") as ws:
            ws.receive_text()  # consume init
            client.post("/api/features", json=_payload())
            raw = ws.receive_text()
        msg = json.loads(raw)
        assert msg["type"] == "assessment"
        assert msg["data"]["zone_id"] == "zone-1"

    def test_alert_broadcast_on_critical_ingest(self):
        messages = []
        with client.websocket_connect("/ws") as ws:
            ws.receive_text()  # init
            client.post("/api/features", json=_payload(
                density=5.0, density_growth_rate=0.7,
                direction_variance=0.8, congestion_index=0.9,
            ))
            # Collect both assessment and alert messages
            for _ in range(2):
                try:
                    messages.append(json.loads(ws.receive_text()))
                except Exception:
                    break
        types = {m["type"] for m in messages}
        assert "assessment" in types
        assert "alert" in types

    def test_init_seeds_existing_zones(self):
        client.post("/api/features", json=_payload())
        with client.websocket_connect("/ws") as ws:
            raw = ws.receive_text()
        msg = json.loads(raw)
        assert msg["type"] == "init"
        assert len(msg["data"]) == 1
        assert msg["data"][0]["zone_id"] == "zone-1"
