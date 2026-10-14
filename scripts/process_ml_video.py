"""GPU/CPU video extraction with the same feature contract as fog research ML.

Batch video analysis, not real-time Pi inference. No fitting or labels created.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from training.quality_motion_baseline import extract_quality


def extract_person_frames(source, device, weights, sample_fps, imgsz):
    config_root = Path(".ultralytics").resolve()
    config_root.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    from ultralytics import YOLO
    import supervision as sv
    from training.gpu_person_baseline import box_features
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot open video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Constant, valid FPS is required")
    stride = max(1, round(fps / sample_fps))
    tracker = sv.ByteTrack(frame_rate=max(1, round(fps / stride)))
    model = YOLO(str(weights))
    previous, previous_gray, previous_time = {}, None, None
    index, rows = -1, []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if index % stride:
                continue
            timestamp = index / fps
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            # Conservatively reset identity across large frame changes; this is
            # not a validated scene-cut detector. The quality stage gates cuts.
            if previous_gray is not None and np.mean(cv2.absdiff(previous_gray, gray)) / 255 > .20:
                tracker = sv.ByteTrack(frame_rate=max(1, round(fps / stride)))
                previous = {}
            result = model.predict(frame, classes=[0], device=device, imgsz=imgsz,
                                   conf=.25, max_det=1000, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            tracked = tracker.update_with_detections(detections)
            current, speeds = {}, []
            dt = timestamp - previous_time if previous_time is not None else stride / fps
            for box, identity in zip(tracked.xyxy, tracked.tracker_id):
                centre = (box[:2] + box[2:]) / 2
                current[int(identity)] = centre
                if int(identity) in previous:
                    speeds.append(float(np.linalg.norm(centre - previous[int(identity)]) / dt / np.hypot(*frame.shape[:2])))
            coverage, local = box_features(detections.xyxy, frame.shape)
            rows.append(dict(clip_id="input", time_s=timestamp, detected_count=len(detections),
                             box_coverage_proxy=coverage, max_local_count_proxy=local,
                             track_speed_proxy=float(np.median(speeds)) if speeds else 0.,
                             boxes_xyxy=detections.xyxy.astype(float).tolist(),
                             frame_width=int(frame.shape[1]), frame_height=int(frame.shape[0])))
            previous, previous_time, previous_gray = current, timestamp, gray
            if len(rows) % 100 == 0:
                print(f"Processed {len(rows)} sampled frames", flush=True)
    finally:
        capture.release()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0", help="CUDA GPU index or cpu")
    parser.add_argument("--weights", default="models/yolo11s.pt")
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--imgsz", type=int, default=1024)
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.imgsz <= 0:
        parser.error("FPS and image size must be positive")
    source, weights = Path(args.source).resolve(), Path(args.weights).resolve()
    if not source.is_file() or not weights.is_file():
        parser.error("Video and trusted local detector weights must exist")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cached = extract_person_frames(source, args.device, weights, args.sample_fps, args.imgsz)
    if not cached:
        raise ValueError("No video frames extracted")
    (output / "person_frames.json").write_text(json.dumps(cached, indent=2), encoding="utf-8")
    features, quality = extract_quality(dict(id="input", path=str(source), event_group="unreviewed-input",
                                            duration_s=cached[-1]["time_s"]), cached, output)
    if not features:
        raise ValueError("No feature windows extracted")
    (output / "features_quality.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (output / "frame_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(dict(source=str(source), device=args.device,
        detector_finetuned=False, windows=len(features), eligible_windows=sum(not r["quality_warning"] for r in features),
        warning="Unreviewed input; no inferred risk ground truth or training permission. Batch inference, not realtime."), indent=2), encoding="utf-8")
    print(f"Extracted {len(features)} windows. Replay using: python -m scripts.replay_ml --features {output / 'features_quality.json'}")


if __name__ == "__main__":
    main()
