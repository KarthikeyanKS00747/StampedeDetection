from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from edge.features import FeatureExtractor
from edge.motion import frame_time, velocities
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
    is_camera = args.source.isdigit()
    native_fps = capture.get(cv2.CAP_PROP_FPS) or args.sample_fps
    stride = max(1, round(native_fps / args.sample_fps))
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
    last_emit = -float("inf")
    previous_time = None
    frame_index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % stride:
            continue
        now = time.monotonic() if is_camera else frame_time(frame_index, native_fps)
        delta_s = now - previous_time if previous_time is not None else stride / native_fps
        delta_s = max(delta_s, 1e-6)
        previous_time = now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_vectors: list[tuple[float, float]] = []
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            sampled = flow[::20, ::20].reshape(-1, 2)
            flow_vectors = velocities(sampled, delta_s)
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
                    track_vectors.extend(velocities([(centre[0] - old[0], centre[1] - old[1])], delta_s))
            previous_centres = current_centres
        if now - last_emit >= args.window_seconds:
            transport.flush()
            transport.send(extractor.extract(args.zone_id, now, count, track_vectors, flow_vectors, source="camera" if is_camera else "video"))
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
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--no-yolo", action="store_true")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.window_seconds <= 0:
        parser.error("sample-fps and window-seconds must be positive")
    extractor = FeatureExtractor(area_m2=args.area_m2)
    transport = FogTransport(args.fog_url)
    simulate(extractor, transport, args.zone_id) if args.simulate else run_video(args, extractor, transport)


if __name__ == "__main__":
    main()
