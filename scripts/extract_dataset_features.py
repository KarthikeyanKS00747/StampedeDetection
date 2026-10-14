from __future__ import annotations

"""Offline feature extraction for labelled video windows.

Output rows deliberately match `training.train` input columns. Labels are never inferred
from a dataset name: provide a reviewed intervals CSV before model training.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2

if __package__ in (None, ""):
    # Support the documented direct script command as well as python -m.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.features import FeatureExtractor
from edge.motion import frame_time, velocities
from shared.schemas import FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one crowd-feature row per video window")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--area-m2", type=float, required=True, help="Calibrated visible ground area")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Detector sampling rate; lower values speed offline extraction")
    parser.add_argument("--zone-id", default="offline-review")
    parser.add_argument("--no-yolo", action="store_true", help="Only validate optical-flow plumbing; do not use output as training data")
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.window_seconds <= 0:
        parser.error("sample-fps and window-seconds must be positive")
    if args.no_yolo:
        raise SystemExit("Refusing to write training features without people counts. Run with a YOLO model.")
    # Keep third-party settings inside the repository instead of a user profile.
    config_root = Path(".ultralytics").resolve()
    config_root.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    from ultralytics import YOLO
    import supervision as sv
    model, tracker = YOLO(args.model), sv.ByteTrack()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or args.sample_fps
    stride = max(1, round(native_fps / args.sample_fps))
    previous_gray, previous_centres, last_emit, previous_timestamp = None, {}, -float("inf"), None
    extractor = FeatureExtractor(area_m2=args.area_m2)
    started = time.monotonic()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_path", "window_start_s", "motion_units", *FEATURE_COLUMNS])
        writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = frame_time(frame_index, native_fps)
            if frame_index % stride:
                continue
            delta_s = max(timestamp - previous_timestamp, 1 / native_fps) if previous_timestamp is not None else 1 / args.sample_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = []
            if previous_gray is not None:
                dense = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                flow = velocities(dense[::20, ::20].reshape(-1, 2), delta_s)
            previous_gray = gray
            previous_timestamp = timestamp
            detections = sv.Detections.from_ultralytics(model(frame, classes=[0], verbose=False)[0])
            tracked = tracker.update_with_detections(detections)
            centres, moves = {}, []
            for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                centre = ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)
                track_id = int(track_id)
                if track_id in previous_centres:
                    old = previous_centres[track_id]
                    moves.extend(velocities([(centre[0] - old[0], centre[1] - old[1])], delta_s))
                centres[track_id] = centre
            previous_centres = centres
            if timestamp - last_emit >= args.window_seconds:
                vector = extractor.extract(args.zone_id, timestamp, len(tracked), moves, flow, source="video")
                row = vector.model_dump()
                writer.writerow({"video_path": args.input, "window_start_s": round(timestamp, 3), "motion_units": "pixels_per_second", **{field: row[field] for field in FEATURE_COLUMNS}})
                last_emit = timestamp
    cap.release()
    print(f"Wrote {output} in {time.monotonic() - started:.1f}s. Add reviewed risk labels before training.")


if __name__ == "__main__":
    main()
