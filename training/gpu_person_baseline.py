"""GPU person/track feature audit + exploratory motion/occupancy anomaly fit.

Uses pretrained detection, not detector fine-tuning or labelled risk training.
No physical density, calibrated danger probability or alert deployment is implied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from training.motion_baseline import fit_variant

PERSON_NAMES = ["detected_count", "box_coverage_proxy", "max_local_count_proxy", "track_speed_proxy"]


def tile_regions(shape):
    height, width = shape[:2]
    tile_width, tile_height = round(width * .6), round(height * .6)
    return [(x, y, x + tile_width, y + tile_height)
            for y in (0, height - tile_height) for x in (0, width - tile_width)]


def box_features(boxes, shape):
    height, width = shape[:2]
    occupancy = np.zeros((32, 32), dtype=bool)
    local = np.zeros((4, 4), dtype=int)
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        gx1, gx2 = int(x1 / width * 32), min(32, int(np.ceil(x2 / width * 32)))
        gy1, gy2 = int(y1 / height * 32), min(32, int(np.ceil(y2 / height * 32)))
        occupancy[gy1:gy2, gx1:gx2] = True
        cx = min(3, int((x1 + x2) / 2 / width * 4))
        cy = min(3, int((y1 + y2) / 2 / height * 4))
        local[cy, cx] += 1
    return float(occupancy.mean()), int(local.max())


def join_features(motion_rows, person_rows):
    joined = []
    for row in motion_rows:
        matches = [r for r in person_rows if r["clip_id"] == row["clip_id"]
                   and row["start_s"] - 1e-6 <= r["time_s"] <= row["end_s"] + 1e-6]
        if not matches:
            raise ValueError("Missing GPU person features for a motion window")
        person_values = np.mean([[r["detected_count"], r["box_coverage_proxy"],
                                 r["max_local_count_proxy"], r["track_speed_proxy"]] for r in matches], axis=0)
        joined.append({**row, "joint": row["raw"] + person_values.tolist()})
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/review/local_video_manifest.json")
    parser.add_argument("--motion-dir", default="outputs/training/motion_baseline_2026_10_14_run02")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--tiled", action="store_true", help="Four overlapping crops, merged with NMS; audit counts, not ground truth")
    args = parser.parse_args()
    if args.sample_fps <= 0 or not 0 < args.conf < 1 or args.imgsz < 32:
        parser.error("Invalid sampling rate, confidence or image size")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
    import torch
    from ultralytics import YOLO
    import supervision as sv
    if not torch.cuda.is_available() or args.device.lower() == "cpu":
        raise RuntimeError("This audit requires CUDA; refusing a silent CPU fallback")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    started = time.monotonic()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    motion = json.loads((Path(args.motion_dir) / "features.json").read_text(encoding="utf-8"))
    selected_ids = {r["clip_id"] for r in motion}
    if selected_ids != {"L002", "L003", "L009", "L010"}:
        raise ValueError("Motion input must contain the same four authorised experimental clips")
    rows, summaries = [], {}
    for record in manifest["videos"]:
        if record["id"] not in selected_ids:
            continue
        cap = cv2.VideoCapture(record["path"])
        if not cap.isOpened():
            raise ValueError(f"Cannot open {record['path']}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, round(fps / args.sample_fps))
        tracker = sv.ByteTrack(frame_rate=max(1, round(fps / stride)))
        previous_centres, previous_time = {}, None
        previous_segment = None
        index, preview_index = -1, 0
        targets = [.2 * record["duration_s"], .5 * record["duration_s"], .8 * record["duration_s"]]
        clip_rows = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                if index % stride:
                    continue
                timestamp = index / fps
                matching_windows = [r for r in motion if r["clip_id"] == record["id"]
                                    and r["start_s"] <= timestamp <= r["end_s"]]
                segment = matching_windows[0]["segment"] if matching_windows else previous_segment
                if previous_segment is not None and segment != previous_segment:
                    tracker = sv.ByteTrack(frame_rate=max(1, round(fps / stride)))
                    previous_centres = {}
                previous_segment = segment
                torch.cuda.synchronize()
                inference_started = time.monotonic()
                if args.tiled:
                    tile_detections = []
                    for x1, y1, x2, y2 in tile_regions(frame.shape):
                        result = model.predict(frame[y1:y2, x1:x2], classes=[0], device=args.device,
                                               imgsz=args.imgsz, conf=args.conf, max_det=1000, verbose=False)[0]
                        tile_detection = sv.Detections.from_ultralytics(result)
                        tile_detection.xyxy += np.array([x1, y1, x1, y1])
                        tile_detections.append(tile_detection)
                    detections = sv.Detections.merge(tile_detections).with_nms(threshold=.5)
                else:
                    result = model.predict(frame, classes=[0], device=args.device, imgsz=args.imgsz,
                                           conf=args.conf, max_det=1000, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(result)
                torch.cuda.synchronize()
                elapsed_ms = (time.monotonic() - inference_started) * 1000
                tracked = tracker.update_with_detections(detections)
                coverage, local_count = box_features(detections.xyxy, frame.shape)
                current_centres, speeds = {}, []
                dt = timestamp - previous_time if previous_time is not None else stride / fps
                for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    track_id = int(track_id)
                    if track_id in previous_centres:
                        speeds.append(float(np.linalg.norm(centre - previous_centres[track_id]) / dt / np.hypot(*frame.shape[:2])))
                    current_centres[track_id] = centre
                previous_centres, previous_time = current_centres, timestamp
                row = {"clip_id": record["id"], "time_s": timestamp, "detected_count": len(detections),
                       "tracked_count": len(tracked), "box_coverage_proxy": coverage,
                       "max_local_count_proxy": local_count,
                       "track_speed_proxy": float(np.median(speeds)) if speeds else 0.0,
                       "valid_track_velocity_count": len(speeds), "gpu_predict_ms": elapsed_ms,
                       "boxes_xyxy": detections.xyxy.astype(float).tolist(),
                       "frame_width": int(frame.shape[1]), "frame_height": int(frame.shape[0]),
                       "confidence_mean": float(detections.confidence.mean()) if len(detections) else None}
                clip_rows.append(row)
                if preview_index < len(targets) and timestamp >= targets[preview_index]:
                    annotated = frame.copy()
                    for box in detections.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{record['id']} {timestamp:.2f}s detected={len(detections)} tiled={args.tiled}",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                    cv2.imwrite(str(output / f"{record['id']}_audit_{preview_index + 1}.jpg"), annotated)
                    preview_index += 1
                if len(clip_rows) % 25 == 0:
                    print(f"{record['id']}: {len(clip_rows)} GPU frames, latest detections={len(detections)}", flush=True)
        finally:
            cap.release()
        rows.extend(clip_rows)
        summaries[record["id"]] = {"frames": len(clip_rows),
            "detection_count_median": float(np.median([r["detected_count"] for r in clip_rows])),
            "detection_count_min": min(r["detected_count"] for r in clip_rows),
            "detection_count_max": max(r["detected_count"] for r in clip_rows),
            "gpu_predict_ms_median": float(np.median([r["gpu_predict_ms"] for r in clip_rows[1:]])),
            "gpu_predict_ms_p95": float(np.quantile([r["gpu_predict_ms"] for r in clip_rows[1:]], .95))}
        # Checkpoint after each clip; useful if a longer run is interrupted.
        (output / "person_frames.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{record['id']} complete: {summaries[record['id']]}", flush=True)
    joined = join_features(motion, rows)
    bundle, scores, summary = fit_variant(joined, "joint")
    bundle["feature_names"] = ["speed_median", "speed_p90", "speed_std", "moving_fraction", "direction_variance", *PERSON_NAMES]
    bundle["artifact_kind"] = "experimental_motion_person_anomaly_v1"
    bundle["units"] = "mixed documented image-space proxies; no physical density"
    joblib.dump(bundle, output / "motion_person_anomaly.joblib")
    (output / "features_joint.json").write_text(json.dumps(joined, indent=2), encoding="utf-8")
    (output / "scores_joint.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "model": args.model, "imgsz": args.imgsz, "conf": args.conf, "device": args.device,
              "tiled": args.tiled, "predict_timing_scope": "Synchronized GPU-assisted predict call, including preprocessing/postprocessing, not pure GPU kernel timing",
              "elapsed_seconds": time.monotonic() - started, "clips": summaries, "fit": summary,
              "detector_finetuned": False, "gpu_role": "Pretrained person detection; anomaly forest fitted on CPU",
              "deployable": False, "accuracy_available": False, "rights_verified": False,
              "warning": "No risk truth or count ground truth. Occlusion/perspective/camera/scene changes remain. Same-scene temporal calibration only."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
