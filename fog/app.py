from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fog.alerts import AlertEngine
from fog.classifier import RiskClassifier
from fog.gpio_alarm import GPIOAlarm
from shared.schemas import AlertEvent, FeatureVector, RiskAssessment

LOG = logging.getLogger(__name__)


class SystemState:
    def __init__(self) -> None:
        self.classifier = RiskClassifier()
        self.alerts = AlertEngine()
        self.gpio = GPIOAlarm()
        self.latest: dict[str, RiskAssessment] = {}
        self.events: deque[AlertEvent] = deque(maxlen=100)
        self.history: dict[str, deque[RiskAssessment]] = defaultdict(lambda: deque(maxlen=300))
        self.clients: set[WebSocket] = set()

    async def publish(self, assessment: RiskAssessment) -> None:
        message = assessment.model_dump_json()
        disconnected = []
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.clients.discard(client)


state = SystemState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    state.gpio.close()


app = FastAPI(title="Stampede Early Warning Fog Service", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/features", response_model=RiskAssessment)
async def ingest(features: FeatureVector) -> RiskAssessment:
    risk, probabilities, reasons = state.classifier.assess(features)
    event = state.alerts.evaluate(features.zone_id, risk, reasons)
    if event:
        state.events.appendleft(event)
    assessment = RiskAssessment(zone_id=features.zone_id, timestamp=features.timestamp, risk=risk, probabilities=probabilities, reasons=reasons, alert_active=state.alerts.is_active(features.zone_id), alert_message=event.message if event else None, metrics=features)
    state.latest[features.zone_id] = assessment
    state.history[features.zone_id].append(assessment)
    state.gpio.set_risk(risk, assessment.alert_active)
    await state.publish(assessment)
    return assessment


@app.get("/api/status")
async def status() -> list[RiskAssessment]:
    return list(state.latest.values())


@app.get("/api/history/{zone_id}")
async def history(zone_id: str) -> list[RiskAssessment]:
    return list(state.history[zone_id])


@app.get("/api/alerts")
async def alerts() -> list[AlertEvent]:
    return list(state.events)


@app.websocket("/ws")
async def websocket(client: WebSocket) -> None:
    await client.accept()
    state.clients.add(client)
    try:
        await client.send_json([item.model_dump(mode="json") for item in state.latest.values()])
        while True:
            await client.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(client)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="joblib classifier trained by training.train")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.model:
        state.classifier = RiskClassifier(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
