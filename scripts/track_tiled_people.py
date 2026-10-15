"""Offline comparison of ByteTrack fed 640, 1280, or tiled YOLO detections.

Track counts are detector-dependent observations, not true crowd counts. This
does not modify the project's deployed extractor, classifier, or alert policy.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from scripts.compare_person_resolution import detect, detect_tiled


def detections(boxes: np.ndarray, scores: np.ndarray) -> sv.Detections:
    return sv.Detections(xyxy=boxes.astype(np.float32),
                         confidence=scores.astype(np.float32),
                         class_id=np.zeros(len(boxes), dtype=int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--weights", type=Path, default=Path("classifier/yolov8n.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/person_resolution/tracking.csv"))
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--tile-imgsz", type=int, default=960)
    args = parser.parse_args()
    if not args.video.is_file() or not args.weights.is_file():
        parser.error("Video and detector weights must exist")
    if (args.start_frame < 0 or args.frames < 1 or args.tile < 64
            or not 0 <= args.overlap < 0.5 or args.tile_imgsz < 64):
        parser.error("Invalid frame range or tiling settings")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError("Video FPS must be positive")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    model = YOLO(str(args.weights))
    baseline_tracker = sv.ByteTrack(frame_rate=round(fps))
    highres_tracker = sv.ByteTrack(frame_rate=round(fps))
    tiled_tracker = sv.ByteTrack(frame_rate=round(fps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for offset in range(args.frames):
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            baseline_boxes, baseline_scores = detect(model, frame, 640)
            baseline_seconds = time.perf_counter() - start
            start = time.perf_counter()
            highres_boxes, highres_scores = detect(model, frame, 1280)
            highres_seconds = time.perf_counter() - start
            start = time.perf_counter()
            tiled_boxes, tiled_scores = detect_tiled(model, frame, args.tile,
                                                      args.overlap, args.tile_imgsz)
            tiled_seconds = time.perf_counter() - start
            baseline_tracks = baseline_tracker.update_with_detections(
                detections(baseline_boxes, baseline_scores))
            highres_tracks = highres_tracker.update_with_detections(
                detections(highres_boxes, highres_scores))
            tiled_tracks = tiled_tracker.update_with_detections(
                detections(tiled_boxes, tiled_scores))
            index = args.start_frame + offset
            row = {"video": args.video.name, "frame": index,
                   "timestamp_s": round(index / fps, 3),
                   "full640_boxes": len(baseline_boxes),
                   "full640_tracks": len(baseline_tracks),
                   "full640_inference_s": round(baseline_seconds, 3),
                   "full1280_boxes": len(highres_boxes),
                   "full1280_tracks": len(highres_tracks),
                   "full1280_inference_s": round(highres_seconds, 3),
                   "tiled_boxes": len(tiled_boxes),
                   "tiled_tracks": len(tiled_tracks),
                   "tiled_inference_s": round(tiled_seconds, 3)}
            rows.append(row)
            print(row, flush=True)
    finally:
        capture.release()
    if not rows:
        raise ValueError("No frames read")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Tracking comparison: {args.output}")


if __name__ == "__main__":
    main()
