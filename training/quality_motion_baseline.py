"""Quality-gated geometric motion experiment using cached GPU detections.

No risk labels are created. Frames without sufficient candidate camera support
are marked unreliable rather than silently treated as corrected motion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from edge.video_quality import candidate_background, compensate, content_mask
from training.gpu_person_baseline import PERSON_NAMES, box_features
from training.motion_baseline import MOTION_NAMES, describe, fit_variant, frame_changes


def describe_content(flow, dt, mask):
    """Black padding must not change the motion normalisation denominator."""
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("No visible content for motion description")
    box = (slice(y.min(), y.max() + 1), slice(x.min(), x.max() + 1))
    return describe(flow[box], dt, mask[box], False)


def mask_unreliable_scores(rows, scores, fit):
    """Unknown quality must not become an apparent Normal or anomaly decision."""
    for row, score in zip(rows, scores):
        if row["quality_warning"]:
            score["score"] = None
            score["exceeds_reference_threshold"] = None
    for clip_id in fit["per_clip"]:
        selected = [r for r in scores if r["clip_id"] == clip_id]
        eligible = [r for r in selected if not r["quality_warning"]]
        fit["per_clip"][clip_id] = {
            "windows": len(selected), "eligible_windows": len(eligible),
            "indeterminate_windows": len(selected) - len(eligible),
            "threshold_exceedance_fraction": float(np.mean([r["exceeds_reference_threshold"] for r in eligible])) if eligible else None,
        }


def extract_quality(record, cached_rows, output):
    cache = {round(r["time_s"], 6): r for r in cached_rows if r["clip_id"] == record["id"]}
    cap = cv2.VideoCapture(record["path"])
    if not cap.isOpened():
        raise ValueError(f"Cannot open {record['path']}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Valid constant-frame-rate FPS required")
    previous, previous_t, previous_boxes, previous_shape = None, None, [], None
    pairs, windows, bucket = [], [], []
    segment, guard_until, bucket_start = 0, -1., None
    preview_saved = False
    clean_saved, index = 0, -1

    def emit():
        if len(bucket) < 3:
            return
        accepted_fraction = float(np.mean([r["camera"]["accepted"] for r in bucket]))
        usable = [r for r in bucket if r["corrected"] is not None]
        corrected = np.mean([r["corrected"] for r in usable], axis=0).tolist() if usable else [0.] * 5
        persons = np.mean([r["persons"] for r in bucket], axis=0).tolist()
        windows.append({"clip_id": record["id"], "event_group": record["event_group"],
                        "start_s": bucket[0]["time_s"], "end_s": bucket[-1]["time_s"],
                        "segment": segment, "pairs": len(bucket),
                        "raw": np.mean([r["raw"] for r in bucket], axis=0).tolist(),
                        "corrected": corrected, "joint": corrected + persons,
                        "camera_accept_fraction": accepted_fraction,
                        "quality_warning": accepted_fraction < .6 or any(r["lighting_shift"] > .12 for r in bucket),
                        "quality_policy": "Candidate camera support >=60% and no large global luminance shift; not a truth guarantee"})

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            timestamp = index / fps
            person_row = cache.get(round(timestamp, 6))
            if person_row is None:
                continue
            height, width = frame.shape[:2]
            scale = min(1., 640 / width, 360 / height)
            gray = cv2.cvtColor(cv2.resize(frame, (round(width * scale), round(height * scale))), cv2.COLOR_BGR2GRAY)
            boxes = person_row["boxes_xyxy"]
            if clean_saved < 3 and timestamp >= record["duration_s"] * (.2 + clean_saved * .3):
                cv2.imwrite(str(output / f"{record['id']}_clean_{clean_saved + 1}.jpg"), frame)
                clean_saved += 1
            if previous is not None:
                mask = content_mask(previous) & content_mask(gray)
                pixel_change, lighting = frame_changes(gray, previous, mask)
                hist1 = cv2.calcHist([previous], [0], mask.astype(np.uint8), [32], [0, 256])
                hist2 = cv2.calcHist([gray], [0], mask.astype(np.uint8), [32], [0, 256])
                histogram_change = cv2.compareHist(cv2.normalize(hist1, None), cv2.normalize(hist2, None), cv2.HISTCMP_BHATTACHARYYA)
                cut = mask.mean() < .05 or (pixel_change > .20 and histogram_change > .45)
                pair = {"clip_id": record["id"], "time_s": timestamp, "cut_candidate": bool(cut),
                        "lighting_shift": lighting, "pixel_change": pixel_change, "segment": segment}
                if cut:
                    emit()
                    bucket, bucket_start = [], None
                    segment += 1
                    guard_until = timestamp + .6
                    pair.update(segment=segment, camera={"accepted": False, "reason": "cut_guard"})
                elif timestamp < guard_until:
                    pair["camera"] = {"accepted": False, "reason": "cut_guard"}
                else:
                    dt = timestamp - previous_t
                    background = candidate_background(previous, previous_boxes, previous_shape)
                    corrected, overlap, camera = compensate(previous, gray, background)
                    pair["camera"] = camera
                    flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                    raw_values = describe_content(flow, dt, mask)
                    corrected_values = None
                    if corrected is not None and overlap.mean() > .05:
                        corrected_flow = cv2.calcOpticalFlowFarneback(previous, corrected, None, .5, 3, 15, 3, 5, 1.2, 0)
                        corrected_values = describe_content(corrected_flow, dt, overlap)
                        if not preview_saved:
                            panel = np.hstack([previous, gray, corrected])
                            cv2.imwrite(str(output / f"{record['id']}_camera_comparison.jpg"), panel)
                            preview_saved = True
                    coverage, local_count = box_features(boxes, frame.shape)
                    ys, xs = np.nonzero(mask)
                    content_diagonal = np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
                    padding_ratio = np.hypot(*gray.shape) / content_diagonal
                    if bucket_start is None:
                        bucket_start = timestamp
                    if timestamp - bucket_start >= 1.:
                        emit()
                        bucket, bucket_start = [], timestamp
                    bucket.append({**pair, "raw": raw_values, "corrected": corrected_values,
                                   "persons": [person_row["detected_count"], coverage, local_count,
                                               person_row["track_speed_proxy"] * padding_ratio]})
                pairs.append(pair)
            previous, previous_t, previous_boxes, previous_shape = gray, timestamp, boxes, frame.shape
        emit()
    finally:
        cap.release()
    return windows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/review/local_video_manifest.json")
    parser.add_argument("--gpu-dir", default="outputs/training/gpu_person_2026_10_14_run04_boxes")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cached = json.loads((Path(args.gpu_dir) / "person_frames.json").read_text(encoding="utf-8"))
    if not all("boxes_xyxy" in r for r in cached):
        raise ValueError("Rerun GPU extraction with bounding-box caching first")
    records = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["videos"]
    rows, pairs, summary, review = [], [], {}, []
    for record in records:
        if record["id"] not in {r["clip_id"] for r in cached}:
            continue
        clip_windows, clip_pairs = extract_quality(record, cached, output)
        rows.extend(clip_windows)
        pairs.extend(clip_pairs)
        summary[record["id"]] = {"windows": len(clip_windows),
            "quality_eligible_windows": sum(not r["quality_warning"] for r in clip_windows),
            "cut_candidate_times_s": [r["time_s"] for r in clip_pairs if r["cut_candidate"]],
            "accepted_camera_pairs": sum(r["camera"]["accepted"] for r in clip_pairs),
            "camera_reasons": dict(Counter(r["camera"]["reason"] for r in clip_pairs))}
        for window in clip_windows:
            review.append({"clip_id": record["id"], "event_group": record["event_group"],
                           "start_s": window["start_s"], "end_s": window["end_s"], "segment": window["segment"],
                           "proposed_observation": "Walking/mixed-flow reference view; no safe-state claim" if record["id"] in ("L002", "L003") else "Entrance crowd; motion/severity uncertain" if record["id"] == "L009" else "Edited street crowd; review shot-specific motion",
                           "proposal_source": "Assistant scene review plus automated timestamps",
                           "annotation_status": "proposal_not_verified", "behaviour_label": None,
                           "risk_label": None, "reviewer": None, "training_approved": False,
                           "quality_eligible": not window["quality_warning"]})
        print(f"{record['id']}: {summary[record['id']]}", flush=True)
    (output / "features_quality.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "frame_quality.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    (output / "review_proposals.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    report = {"clips": summary, "deployable": False, "accuracy_available": False,
              "warning": "Geometric estimates use imperfect detector-excluded anchors; neither camera truth nor risk truth is established.",
              "fits": {}}
    for variant in ("corrected", "joint"):
        try:
            bundle, scores, fit = fit_variant(rows, variant)
        except ValueError as error:
            report["fits"][variant] = {"status": "not_fitted", "reason": str(error)}
            continue
        bundle.update(artifact_kind="experimental_quality_motion_anomaly_v1",
                      feature_names=MOTION_NAMES + (PERSON_NAMES if variant == "joint" else []),
                      units="visible_content_diagonal_per_second" if variant == "corrected" else "mixed image-space proxies; content-normalised motion; track speed is not camera compensated",
                      quality_gate_required=True)
        # Unreliable frames/windows receive no threshold decision. Their zero
        # corrected placeholders never appear as reliable low-risk evidence.
        mask_unreliable_scores(rows, scores, fit)
        joblib.dump(bundle, output / f"quality_{variant}.joblib")
        (output / f"scores_{variant}.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        report["fits"][variant] = {"status": "fitted_research_only", "fit": fit}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["fits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
