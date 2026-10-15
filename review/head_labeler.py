"""Local-only head-point annotation for independently reviewing audit frames.

Run: python -m review.head_labeler serve
Export: python -m review.head_labeler export --val-video VIDEO_NAME
No inferred YOLO boxes are used as labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/person_audit/manifest.csv"
DEFAULT_LABELS = ROOT / "data/review/head_points.json"
STATIC = Path(__file__).parent / "static"
USER_PERMISSION_STATEMENT = "User confirmed source permission for this project on 2026-10-15; not independently verified"


class HeadAnnotation(BaseModel):
    expected_revision: int = Field(ge=0)
    status: str = "draft"
    points: list[tuple[float, float]] = Field(default_factory=list, max_length=10000)
    reviewer: str = Field(default="", max_length=100)
    note: str = Field(default="", max_length=2000)


def load_frames(manifest: Path) -> list[dict]:
    frames = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            image = (ROOT / row["original"]).resolve()
            if not image.is_relative_to(ROOT) or not image.is_file():
                raise ValueError(f"Invalid image path in audit manifest: {image}")
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Unsupported audit image: {image}")
            picture = cv2.imread(str(image))
            if picture is None:
                raise ValueError(f"Cannot read audit image: {image}")
            frames.append({"id": image.stem, "video": row["video"],
                           "frame": int(row["frame"]),
                           "timestamp_s": float(row["timestamp_s"]),
                           "detected_people": int(row["detected_people"]),
                           "image": image, "width": picture.shape[1],
                           "height": picture.shape[0]})
    if not frames or len({row["id"] for row in frames}) != len(frames):
        raise ValueError("Audit manifest is empty or has duplicate image IDs")
    return frames


class HeadStore:
    def __init__(self, manifest: Path, labels: Path) -> None:
        self.frames = load_frames(manifest)
        self.by_id = {row["id"]: row for row in self.frames}
        self.labels = labels
        self.lock = threading.RLock()

    def read(self) -> dict:
        if not self.labels.is_file():
            return {}
        return json.loads(self.labels.read_text(encoding="utf-8"))["annotations"]

    def save(self, item_id: str, annotation: HeadAnnotation) -> dict:
        frame = self.by_id.get(item_id)
        if frame is None:
            raise HTTPException(404, "Unknown frame")
        if annotation.status not in {"draft", "reviewed", "uncountable"}:
            raise HTTPException(422, "Invalid status")
        if annotation.status != "draft" and (not annotation.reviewer.strip()
                                              or not annotation.note.strip()):
            raise HTTPException(422, "Reviewed/uncountable frames need reviewer and note")
        if annotation.status == "uncountable" and annotation.points:
            raise HTTPException(422, "Uncountable frames cannot have count labels")
        points = annotation.points
        if (len(set(points)) != len(points) or any(not (0 <= x < frame["width"]
                and 0 <= y < frame["height"]) for x, y in points)):
            raise HTTPException(422, "Head points must be unique and inside the image")
        with self.lock:
            labels = self.read()
            previous = labels.get(item_id, {"revision": 0})
            if annotation.expected_revision != previous["revision"]:
                raise HTTPException(409, "Annotation changed elsewhere; reload")
            saved = {**annotation.model_dump(exclude={"expected_revision"}),
                     "permission_evidence": USER_PERMISSION_STATEMENT,
                     "revision": previous["revision"] + 1}
            labels[item_id] = saved
            self.labels.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.labels.with_suffix(self.labels.suffix + ".tmp")
            temporary.write_text(json.dumps({"annotations": labels}, indent=2),
                                 encoding="utf-8")
            os.replace(temporary, self.labels)
            return saved

    def export(self, validation_video: str, output_dir: Path) -> tuple[int, int]:
        videos = {row["video"] for row in self.frames}
        if validation_video not in videos:
            raise ValueError(f"Unknown validation video {validation_video!r}; choose {sorted(videos)}")
        labels = self.read()
        partitions = {"train": [], "val": []}
        for frame in self.frames:
            annotation = labels.get(frame["id"], {})
            if (annotation.get("status") != "reviewed"
                    or not annotation.get("reviewer", "").strip()
                    or not annotation.get("note", "").strip()):
                continue
            partition = "val" if frame["video"] == validation_video else "train"
            partitions[partition].append((frame, annotation))
        if not partitions["train"] or not partitions["val"]:
            raise ValueError("Need reviewed head labels in both different-video splits")
        output_dir.mkdir(parents=True, exist_ok=True)
        for partition, rows in partitions.items():
            output = output_dir / f"head_{partition}.jsonl"
            with output.open("w", encoding="utf-8") as stream:
                for frame, annotation in rows:
                    image = os.path.relpath(frame["image"], output_dir)
                    stream.write(json.dumps({"image": image, "scene": frame["video"],
                                             "points": annotation["points"],
                                             "reviewer": annotation["reviewer"],
                                             "review_note": annotation["note"],
                                             "permission_evidence": USER_PERMISSION_STATEMENT}) + "\n")
        return len(partitions["train"]), len(partitions["val"])


def create_app(manifest: Path = DEFAULT_MANIFEST,
               labels: Path = DEFAULT_LABELS) -> FastAPI:
    store = HeadStore(manifest, labels)
    app = FastAPI(title="Local head-point annotation")
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.headers.get("host", "").split(":")[0] not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"detail": "Local-only tool"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD"} and origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin writes denied"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/")
    def home():
        return FileResponse(STATIC / "head_labeler.html")

    @app.get("/api/head-frames")
    def frames():
        return {"frames": [{k: v for k, v in row.items() if k != "image"}
                           for row in store.frames], "annotations": store.read()}

    @app.get("/media/{item_id}")
    def media(item_id: str):
        frame = store.by_id.get(item_id)
        if frame is None:
            raise HTTPException(404, "Unknown frame")
        return FileResponse(frame["image"])

    @app.post("/api/head-labels/{item_id}")
    def save(item_id: str, annotation: HeadAnnotation):
        return store.save(item_id, annotation)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8012)
    export = sub.add_parser("export")
    export.add_argument("--val-video", required=True)
    export.add_argument("--output-dir", type=Path, default=ROOT / "data/review/density_labels")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
    else:
        try:
            train_count, val_count = HeadStore(DEFAULT_MANIFEST, DEFAULT_LABELS).export(
                args.val_video, args.output_dir)
        except ValueError as exc:
            parser.exit(2, f"Cannot export: {exc}\n")
        print(f"Exported {train_count} train and {val_count} validation frames to {args.output_dir}")


if __name__ == "__main__":
    main()
