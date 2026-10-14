"""Experimental reference-motion anomaly training, NOT crowd-risk prediction.

Only the four user-supplied Kaggle-origin clips are selected by default. Rights
are unresolved; no licence approvals or incident ground-truth labels are created.
The saved bundles deliberately cannot be loaded as fog risk classifiers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

MOTION_NAMES = ["speed_median", "speed_p90", "speed_std", "moving_fraction", "direction_variance"]


def frame_changes(gray, previous, mask):
    """Separate pixel changes (including moving people) from global luminance shift."""
    if not np.any(mask):
        return 1.0, 1.0
    difference = gray.astype(float)[mask] - previous.astype(float)[mask]
    return float(np.mean(np.abs(difference)) / 255), float(abs(np.mean(difference)) / 255)


def describe(flow: np.ndarray, dt: float, mask: np.ndarray, compensate: bool) -> list[float]:
    vectors = flow[mask].astype(float)
    if not len(vectors) or dt <= 0:
        raise ValueError("Empty motion mask or invalid interval")
    if compensate:
        # An ablation proxy only. Coherent crowd movement can dominate this
        # median and be subtracted too; this is NOT geometric stabilisation.
        vectors = vectors - np.median(vectors, axis=0)
    vectors /= dt * np.hypot(*flow.shape[:2])
    speed = np.linalg.norm(vectors, axis=1)
    active = speed > 0.005
    if np.count_nonzero(active) > 1:
        angles = np.arctan2(vectors[active, 1], vectors[active, 0])
        direction = 1 - float(np.hypot(np.cos(angles).mean(), np.sin(angles).mean()))
    else:
        direction = 0.0
    return [float(np.median(speed)), float(np.quantile(speed, .9)),
            float(speed.std()), float(active.mean()), float(np.clip(direction, 0, 1))]


def extract(record: dict, sample_fps: float = 5.0) -> list[dict]:
    cap = cv2.VideoCapture(record["path"])
    if not cap.isOpened():
        raise ValueError(f"Cannot open {record['path']}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Valid constant-frame-rate FPS required")
    stride = max(1, round(fps / sample_fps))
    previous, previous_t = None, None
    rows, bucket = [], []
    segment, index, bucket_start = 0, -1, None

    def emit():
        if len(bucket) < 3:
            return
        rows.append({"clip_id": record["id"], "event_group": record["event_group"],
                     "start_s": bucket[0]["time_s"], "end_s": bucket[-1]["time_s"],
                     "segment": segment, "pairs": len(bucket),
                     "raw": np.mean([r["raw"] for r in bucket], axis=0).tolist(),
                     "residual": np.mean([r["residual"] for r in bucket], axis=0).tolist(),
                     "brightness_change": float(np.mean([r["brightness"] for r in bucket])),
                     "global_flow_proxy": float(np.mean([r["global"] for r in bucket])),
                     "quality_warning": any(r["brightness"] > .12 for r in bucket)})

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index % stride:
                continue
            timestamp = index / fps
            height, width = frame.shape[:2]
            scale = min(1.0, 640 / width, 360 / height)
            gray = cv2.cvtColor(cv2.resize(frame, (round(width * scale), round(height * scale))), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                # Exclude black letterboxing; not a person ROI or overlay mask.
                mask = (gray > 8) & (previous > 8)
                valid_fraction = float(mask.mean())
                pixel_change, brightness = frame_changes(gray, previous, mask)
                hist1 = cv2.calcHist([previous], [0], mask.astype(np.uint8), [32], [0, 256])
                hist2 = cv2.calcHist([gray], [0], mask.astype(np.uint8), [32], [0, 256])
                hist_distance = cv2.compareHist(cv2.normalize(hist1, None), cv2.normalize(hist2, None), cv2.HISTCMP_BHATTACHARYYA)
                likely_cut = valid_fraction <= .05 or (pixel_change > .20 and hist_distance > .45)
                if likely_cut:
                    emit()
                    bucket, bucket_start = [], None
                    segment += 1
                else:
                    dt = timestamp - previous_t
                    flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                    if bucket_start is None:
                        bucket_start = timestamp
                    if timestamp - bucket_start >= 1.0:
                        emit()
                        bucket, bucket_start = [], timestamp
                    vectors = flow[mask]
                    bucket.append({"time_s": timestamp, "raw": describe(flow, dt, mask, False),
                                   "residual": describe(flow, dt, mask, True), "brightness": brightness,
                                   "global": float(np.linalg.norm(np.median(vectors, axis=0)) / dt / np.hypot(*gray.shape))})
            previous, previous_t = gray, timestamp
        emit()
    finally:
        cap.release()
    return rows


def fit_variant(rows: list[dict], variant: str):
    reference = [r for r in rows if r["clip_id"] in ("L002", "L003") and not r["quality_warning"]]
    # Temporal calibration with a gap; reference clips share a scene and are
    # NOT independent train/test events. No accuracy claim follows from this.
    train, calibration = [], []
    for clip_id in ("L002", "L003"):
        selected = [r for r in reference if r["clip_id"] == clip_id]
        if len(selected) < 12:
            raise ValueError(f"Too few usable reference windows in {clip_id}")
        boundary = .6 * max(r["end_s"] for r in selected)
        train.extend(r for r in selected if r["end_s"] < boundary)
        calibration.extend(r for r in selected if r["start_s"] > boundary + 2)
    if len(train) < 20 or len(calibration) < 8:
        raise ValueError("Insufficient training/calibration windows")
    model = make_pipeline(RobustScaler(), IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1))
    model.fit(np.asarray([r[variant] for r in train]))
    calibration_scores = -model.score_samples(np.asarray([r[variant] for r in calibration]))
    threshold = float(np.quantile(calibration_scores, .95))
    scores = -model.score_samples(np.asarray([r[variant] for r in rows]))
    results = [{"clip_id": r["clip_id"], "start_s": r["start_s"], "end_s": r["end_s"],
                "score": float(score), "exceeds_reference_threshold": bool(score > threshold),
                "quality_warning": r["quality_warning"]} for r, score in zip(rows, scores)]
    bundle = {"artifact_kind": "experimental_motion_anomaly_v1", "pipeline": model,
              "feature_names": MOTION_NAMES, "variant": variant, "threshold": threshold,
              "units": "frame_diagonal_per_second", "deployable": False,
              "rights_verified": False, "risk_classes": None}
    summary = {"train_windows": len(train), "calibration_windows": len(calibration),
               "threshold": threshold, "calibration_exceedance_fraction": float((calibration_scores > threshold).mean()),
               "per_clip": {}}
    for clip_id in sorted({r["clip_id"] for r in rows}):
        clip_results = [r for r in results if r["clip_id"] == clip_id]
        summary["per_clip"][clip_id] = {"windows": len(clip_results),
            "threshold_exceedance_fraction": float(np.mean([r["exceeds_reference_threshold"] for r in clip_results])),
            "quality_warning_windows": sum(r["quality_warning"] for r in clip_results)}
    return bundle, results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/review/local_video_manifest.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = [r for r in manifest["videos"] if r["id"] in ("L002", "L003", "L009", "L010")]
    if len(records) != 4:
        raise ValueError("Expected exactly L002, L003, L009 and L010")
    rows = []
    for record in records:
        clip_rows = extract(record)
        rows.extend(clip_rows)
        print(f"{record['id']}: {len(clip_rows)} windows extracted", flush=True)
    (output / "features.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_type": "Unsupervised reference-motion baseline; not four-tier risk classification",
              "licence_status": "User reports Kaggle origin and cropping; exact dataset URL/licence unresolved",
              "reference_clips": ["L002", "L003"], "probe_clips": ["L009", "L010"],
              "accuracy_available": False, "independent_event_test_available": False,
              "ground_truth_risk_labels_available": False,
              "warnings": ["Camera and scene/domain changes can dominate anomaly scores.",
                           "Threshold is exploratory, chosen from same-scene temporal calibration.",
                           "Residual-flow subtraction may suppress genuine coherent crowd movement.",
                           "No crowd density/count, physical pressure, lead-time or risk probability is measured.",
                           "No production deployment or fog risk-model loading is permitted for these bundles."],
              "source_hashes": {r["id"]: r["sha256"] for r in records}, "variants": {}}
    for variant in ("raw", "residual"):
        bundle, results, summary = fit_variant(rows, variant)
        joblib.dump(bundle, output / f"motion_{variant}.joblib")
        (output / f"scores_{variant}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        report["variants"][variant] = summary
        print(f"{variant}: fitted {summary['train_windows']} reference windows, calibrated {summary['calibration_windows']}", flush=True)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["variants"], indent=2), flush=True)


if __name__ == "__main__":
    main()
