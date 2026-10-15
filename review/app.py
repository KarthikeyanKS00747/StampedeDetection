from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOURS = {"stationary", "walking", "running", "mixed_motion", "uncertain"}
USER_PERMISSION_STATEMENT = "User confirmed source permission for this project on 2026-10-15; not independently verified"


class Annotation(BaseModel):
    expected_revision: int = Field(ge=0)
    status: str = "draft"
    behaviour_label: str | None = None
    reviewer: str = Field(default="", max_length=100)
    note: str = Field(default="", max_length=2000)


class ReviewStore:
    def __init__(self, package, annotation_path):
        self.package = package
        self.items = {r["id"]: r for r in package["windows"]}
        self.path = Path(annotation_path)
        self.lock = threading.RLock()

    def read(self):
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))["annotations"]

    def save(self, item_id, annotation):
        if item_id not in self.items:
            raise HTTPException(404, "Unknown window")
        if annotation.status not in {"draft", "reviewed", "uncertain", "exclude"}:
            raise HTTPException(422, "Invalid review status")
        if annotation.behaviour_label is not None and annotation.behaviour_label not in BEHAVIOURS:
            raise HTTPException(422, "Invalid behaviour label")
        if annotation.status == "reviewed" and (annotation.behaviour_label in {None, "uncertain"}
                or not annotation.reviewer.strip() or not annotation.note.strip()):
            raise HTTPException(422, "Reviewed entries need a definite behaviour, reviewer and observation note")
        with self.lock:
            values = self.read()
            previous = values.get(item_id, {"revision": 0})
            if previous["revision"] != annotation.expected_revision:
                raise HTTPException(409, "Annotation changed elsewhere; reload before saving")
            values[item_id] = {**annotation.model_dump(exclude={"expected_revision"}),
                               "rights_approved": True,
                               "permission_evidence": USER_PERMISSION_STATEMENT,
                               "revision": previous["revision"] + 1,
                               "updated_at": datetime.now(timezone.utc).isoformat()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps({"annotations": values}, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
            return values[item_id]

    def export(self):
        annotations, accepted = self.read(), []
        for item_id, item in self.items.items():
            a = annotations.get(item_id, {})
            if (a.get("status") == "reviewed" and a.get("behaviour_label") in BEHAVIOURS - {"uncertain"}
                    and a.get("reviewer", "").strip() and a.get("note", "").strip()
                    and item["quality_eligible"] is True):
                accepted.append({"window_id": item_id, "clip_id": item["clip_id"],
                                 "event_id": item["event_group"], "start_s": item["start_s"], "end_s": item["end_s"],
                                 "features": item["features"], "behaviour_label": a["behaviour_label"],
                                 "reviewer": a["reviewer"], "review_note": a["note"],
                                 "rights_approved": True,
                                 "permission_evidence": USER_PERMISSION_STATEMENT,
                                 "quality_eligible": True})
        return {"schema": "reviewed_behaviour_features_v1", "feature_names": self.package["feature_names"],
                "risk_labels_created": False, "rows": accepted}


def create_app(package_path=ROOT / "data/review/annotation_package.json", annotation_path=ROOT / "data/review/behaviour_annotations.json"):
    package = json.loads(Path(package_path).read_text(encoding="utf-8"))
    store = ReviewStore(package, annotation_path)
    app = FastAPI(title="Local crowd behaviour review")
    app.state.store = store
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.headers.get("host", "").split(":")[0] not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"detail": "Local-only tool"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD"} and origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin writes denied"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; media-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/api/review")
    def read_review():
        annotations = store.read()
        return {**package, "annotations": annotations, "exportable_count": len(store.export()["rows"])}

    @app.post("/api/annotations/{item_id}")
    def save(item_id: str, annotation: Annotation):
        return store.save(item_id, annotation)

    @app.get("/api/export")
    def export():
        return JSONResponse(store.export(), headers={"Content-Disposition": 'attachment; filename="reviewed_behaviour_features.json"'})

    @app.get("/media/{clip_id}")
    def media(clip_id: str):
        clip = package["clips"].get(clip_id)
        if clip is None:
            raise HTTPException(404, "Unknown clip")
        path = Path(clip["path"])
        if not path.is_file():
            raise HTTPException(404, "Source video unavailable")
        return FileResponse(path)

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Run local-only behaviour annotation; no uploads or risk-label inference")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
