from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from edge.features import FeatureExtractor
from edge.transport import FogTransport


def simulate(extractor: FeatureExtractor, transport: FogTransport, zone_id: str) -> None:
    """Safe end-to-end test signal; it does not represent people or real risk."""
    started = time.monotonic()
    while True:
        phase = int((time.monotonic() - started) / 20) % 4
        count = (8, 35, 75, 115)[phase]
        jitter = np.random.default_rng().normal(0, (2, 5, 12, 20)[phase], size=(max(count // 3, 1), 2))
        vector = extractor.extract(zone_id, time.time(), count, jitter, [], source="simulation")
        transport.send(vector)
        time.sleep(1)


def run_video(args: argparse.Namespace, extractor: FeatureExtractor, transport: FogTransport) -> None:
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    model = None
    tracker = None
    if not args.no_yolo:
        config_root = Path(".ultralytics").resolve()
        config_root.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
        from ultralytics import YOLO
        import supervision as sv
        model = YOLO(args.model)
        tracker = sv.ByteTrack()
    previous_gray = None
    previous_centres: dict[int, tuple[float, float]] = {}
    last_emit = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_vectors: list[tuple[float, float]] = []
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            sampled = flow[::20, ::20].reshape(-1, 2)
            flow_vectors = [(float(x), float(y)) for x, y in sampled if x * x + y * y > 0.04]
        previous_gray = gray
        count, track_vectors = 0, []
        if model is not None and tracker is not None:
            result = model(frame, classes=[0], verbose=False)[0]
            detections = __import__("supervision").Detections.from_ultralytics(result)
            tracked = tracker.update_with_detections(detections)
            count = len(tracked)
            current_centres = {}
            for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                centre = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                current_centres[int(track_id)] = centre
                if int(track_id) in previous_centres:
                    old = previous_centres[int(track_id)]
                    track_vectors.append((centre[0] - old[0], centre[1] - old[1]))
            previous_centres = current_centres
        if now - last_emit >= args.window_seconds:
            transport.flush()
            transport.send(extractor.extract(args.zone_id, now, count, track_vectors, flow_vectors, source="video" if not args.source.isdigit() else "camera"))
            last_emit = now
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-camera edge crowd feature extractor")
    parser.add_argument("--fog-url", required=True)
    parser.add_argument("--source", default="0", help="OpenCV camera index or video file")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--zone-id", default="zone-1")
    parser.add_argument("--area-m2", type=float, default=25.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--no-yolo", action="store_true")
    args = parser.parse_args()
    extractor = FeatureExtractor(area_m2=args.area_m2)
    transport = FogTransport(args.fog_url)
    simulate(extractor, transport, args.zone_id) if args.simulate else run_video(args, extractor, transport)


if __name__ == "__main__":
    main()
