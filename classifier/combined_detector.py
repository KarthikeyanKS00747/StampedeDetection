"""Offline dry run: one YOLOv8n pass feeds both crowd signals.

Run from the repository root with ``python -m classifier.combined_detector``.
The CSV contains review candidates only; this module never calls alerts or GPIO.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import deque
import csv
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable

import cv2
import numpy as np

from classifier import extract_features as sequence_features
from classifier.fusion import FusionPolicy
from classifier.infer import FEATURE_NAMES, load_model
from edge.features import FeatureExtractor
from edge.motion import velocities
from edge.person_tracking import PersonTracker
from fog.classifier import RiskClassifier
from shared.schemas import RiskLevel


def image_frames(directory: Path) -> Iterable[tuple[int, np.ndarray, str, float]]:
    files = sorted(directory.glob("scene-*.png"))
    if not files:
        raise ValueError(f"No scene-*.png frames in {directory}")
    previous_number = None
    for index, path in enumerate(files):
        match = re.fullmatch(r"scene-(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Unexpected frame name: {path.name}")
        number = int(match.group(1))
        if previous_number is not None and number != previous_number + 1:
            raise ValueError(f"Missing or repeated frame before {path.name}")
        previous_number = number
        read_started = time.perf_counter()
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        yield index, frame, path.name, read_started


def video_frames(path: Path) -> Iterable[tuple[int, np.ndarray, str, float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        index = 0
        while True:
            read_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame, f"frame-{index:06}", read_started
            index += 1
    finally:
        capture.release()


def score_window(
    window: dict,
    baseline: tuple[int, RiskLevel, list[str], object, str] | None,
    model,
    policy: FusionPolicy,
    fps: float,
    first_read_at: float | None = None,
    end_read_at: float | None = None,
    clock=time.perf_counter,
) -> dict:
    values = np.array([[window[name] for name in FEATURE_NAMES]], dtype=np.float32)
    label = int(model.predict(values)[0])
    if label not in (0, 1):
        raise ValueError("Sequence model returned an unsupported label")
    probability = model.predict_proba(values)[0]
    class_index = list(model.classes_).index(label)
    end_frame = window["window_end_frame"]
    if baseline is not None and (end_frame - baseline[0]) / fps > 1.5:
        baseline = None
    risk = baseline[1] if baseline else None
    decision = policy.observe(window["window_start_frame"], label == 1, risk)
    decided_at = clock()
    metrics = baseline[3] if baseline else None
    return {
        "window_start_frame": window["window_start_frame"],
        "window_end_frame": end_frame,
        "window_start_s": window["window_start_ts"],
        "window_end_s": round(end_frame / fps, 4),
        "sequence_label": "abnormal" if label else "normal",
        "sequence_confidence": round(float(probability[class_index]), 4),
        "baseline_frame": baseline[0] if baseline else "",
        "baseline_risk": risk.value if risk else "unavailable",
        "baseline_mode": baseline[4] if baseline else "",
        "baseline_reasons": "; ".join(baseline[2]) if baseline else "",
        "baseline_count": metrics.crowd_count if metrics else "",
        "baseline_density": metrics.density if metrics else "",
        "baseline_growth": metrics.density_growth_rate if metrics else "",
        "baseline_velocity": metrics.average_velocity if metrics else "",
        "baseline_velocity_variance": metrics.velocity_variance if metrics else "",
        "baseline_direction_variance": metrics.direction_variance if metrics else "",
        "baseline_congestion": metrics.congestion_index if metrics else "",
        "anomaly_episode_active": decision.anomaly_episode_active,
        "combined_candidate": decision.combined_candidate,
        "combined_episode_started": decision.combined_episode_started,
        "combined_episode_id": decision.combined_episode_id or "",
        "window_processing_wall_s": (
            round(decided_at - first_read_at, 4) if first_read_at is not None else ""
        ),
        "end_frame_processing_wall_s": (
            round(decided_at - end_read_at, 4) if end_read_at is not None else ""
        ),
        **{name: round(window[name], 5) for name in FEATURE_NAMES},
    }


def combine_windows(
    per_frame: list[dict],
    samples: list[tuple[int, RiskLevel]],
    model,
    fps: float,
) -> list[dict]:
    """Pure replay helper for unit tests; production dry run streams frames."""
    if len(per_frame) < 30:
        raise ValueError("At least 30 frames are required")
    if any(b["frame_idx"] != a["frame_idx"] + 1
           for a, b in zip(per_frame, per_frame[1:])):
        raise ValueError("Missing frames: do not infer across a frame gap")
    if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError("Baseline samples must be strictly ordered")
    frames = [frame for frame, _ in samples]
    policy = FusionPolicy()
    results = []
    for window in sequence_features.window_aggregate(per_frame, 30):
        position = bisect_right(frames, window["window_end_frame"]) - 1
        baseline = None
        if position >= 0:
            frame, risk = samples[position]
            baseline = (frame, risk, [], None, "test")
        results.append(score_window(window, baseline, model, policy, fps))
    return results


def process_frames(
    frames: Iterable[tuple[int, np.ndarray, str, float]],
    fps: float,
    sample_fps: float,
    area_m2: float,
    detector,
    tracker,
    sequence_model,
    risk_classifier,
    clock=time.perf_counter,
) -> Iterable[dict]:
    """Consume one recording without a second detector pass or GPIO calls."""
    stride = max(1, round(fps / sample_fps))
    extractor = FeatureExtractor(area_m2=area_m2)
    policy = FusionPolicy()
    rows: deque[dict] = deque(maxlen=30)
    read_times: deque[float] = deque(maxlen=30)
    previous_gray = None
    previous_sample_gray = None
    previous_sample_index = None
    last_emit = -math.inf
    latest_baseline = None
    previous_index = None
    for index, frame, name, read_started in frames:
        if previous_index is not None and index != previous_index + 1:
            raise ValueError("Missing or repeated frame in input stream")
        previous_index = index
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = detector(frame, classes=[0], verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else []
        row = sequence_features.frame_observation(
            index, fps, gray, previous_gray, boxes, name
        )
        previous_gray = gray
        rows.append(row)
        read_times.append(read_started)

        if index % stride == 0:
            media_time = index / fps
            delta_s = (
                (index - previous_sample_index) / fps
                if previous_sample_index is not None else stride / fps
            )
            flow_vectors = []
            if previous_sample_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_sample_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_vectors = velocities(flow[::20, ::20].reshape(-1, 2), delta_s)
            previous_sample_gray = gray
            previous_sample_index = index
            count, track_vectors = tracker.update(result, delta_s)
            if media_time - last_emit >= 1.0:
                metrics = extractor.extract(
                    "offline", media_time, count, track_vectors, flow_vectors,
                    source="video",
                )
                risk, _, reasons, mode = risk_classifier.assess(metrics)
                latest_baseline = (index, risk, reasons, metrics, mode.value)
                last_emit = media_time

        if len(rows) == 30 and (index - 29) % 15 == 0:
            window = next(sequence_features.window_aggregate(list(rows), 30))
            yield score_window(
                window, latest_baseline, sequence_model, policy, fps,
                read_times[0], read_started, clock,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=float, required=True, help="Constant source FPS")
    parser.add_argument("--area-m2", type=float, required=True, help="Calibrated area")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--model-dir", type=Path, default=Path("classifier/model"))
    parser.add_argument("--risk-model", help="Optional seven-feature risk model")
    parser.add_argument("--max-frames", type=int, help="Bounded integration smoke run")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0
               for value in (args.fps, args.area_m2, args.sample_fps)):
        parser.error("fps, area-m2, and sample-fps must be finite and positive")
    if args.max_frames is not None and args.max_frames < 30:
        parser.error("--max-frames must be at least 30")
    selected = args.image_dir or args.video
    if args.image_dir and not args.image_dir.is_dir():
        parser.error("--image-dir must exist")
    if args.video and not args.video.is_file():
        parser.error("--video must exist")
    weights = Path(__file__).parent / "yolov8n.pt"
    if not weights.is_file():
        parser.error(f"Matching detector weights missing: {weights}")

    from ultralytics import YOLO

    detector = YOLO(str(weights))
    tracker = PersonTracker()
    sequence_model, _ = load_model(args.model_dir)
    risk_classifier = RiskClassifier(args.risk_model)
    frame_source = image_frames(args.image_dir) if args.image_dir else video_frames(args.video)
    frames = islice(frame_source, args.max_frames) if args.max_frames else frame_source
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    window_count = 0
    episode_count = 0
    last_end_frame = None
    try:
        with args.output_csv.open("w", newline="") as stream:
            writer = None
            for row in process_frames(
                frames, args.fps, args.sample_fps, args.area_m2, detector, tracker,
                sequence_model, risk_classifier,
            ):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                window_count += 1
                episode_count += int(row["combined_episode_started"])
                last_end_frame = row["window_end_frame"]
    finally:
        frame_source.close()
    if window_count == 0:
        raise ValueError("Recording has fewer than 30 usable frames")
    metadata = {
        "source": str(selected),
        "source_fps": args.fps,
        "sample_fps": args.sample_fps,
        "area_m2": args.area_m2,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "sequence_model_sha256": hashlib.sha256(
            (args.model_dir / "classifier.pkl").read_bytes()
        ).hexdigest(),
        "window_count": window_count,
        "candidate_episode_count": episode_count,
        "last_window_end_frame": last_end_frame,
        "max_frames": args.max_frames,
        "partial_smoke_run": args.max_frames is not None,
        "total_offline_runtime_s": round(time.perf_counter() - started, 4),
        "timing_scope": "offline processing wall time, not live capture-to-alert delay",
        "hardware_alarm_enabled": False,
    }
    sidecar = args.output_csv.with_suffix(args.output_csv.suffix + ".meta.json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {window_count} windows, {episode_count} review candidates "
          f"to {args.output_csv}")
    print(f"Provenance and timing scope: {sidecar}")


if __name__ == "__main__":
    main()
