"""
extract_features.py
-------------------
Runs a YOLOv8n person detector over a sorted directory of PNG frames and
writes a CSV of per-frame and windowed aggregate features.

Feature vector (7 columns, frozen order — see model/feature_order.txt):
  N_mean, N_std, overlap_mean, overlap_std, motion_mean, motion_std, delta_N

Usage
-----
  python extract_features.py \
      --image_dir "path/to/scene_frames" \
      --fps 25 \
      --window 30 \
      --output_csv features.csv

Notes
-----
- Frames must follow the scene-NNNNN.png naming convention.
- --fps is REQUIRED and must match the source video FPS documented in the
  dataset README (Times Square: 29.97, Las Vegas: 15.17, Love Parade: 25,
  Italy: 25). It is NOT auto-detected.
- Person boxes are used for count and motion features only. They are NOT
  stampede labels.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


FEATURE_NAMES = [
    "N_mean",
    "N_std",
    "overlap_mean",
    "overlap_std",
    "motion_mean",
    "motion_std",
    "delta_N",
]


def _iou(boxA, boxB):
    """Axis-aligned IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def _mean_pairwise_iou(boxes):
    """Mean pairwise IoU across all person box pairs in a frame."""
    if len(boxes) < 2:
        return 0.0
    scores = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            scores.append(_iou(boxes[i], boxes[j]))
    return float(np.mean(scores))


def _frame_diff_magnitude(gray_prev, gray_curr):
    """Mean absolute pixel difference as a motion proxy."""
    if gray_prev is None:
        return 0.0
    diff = cv2.absdiff(gray_prev, gray_curr)
    return float(diff.mean())


def extract_per_frame(image_dir: Path, fps: float):
    """
    Yields dicts with per-frame raw features:
      frame_idx, timestamp_s, count, mean_iou, motion
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ERROR: ultralytics not installed. Run: pip install ultralytics"
        )

    model = YOLO("yolov8n.pt")  # downloads on first run

    frames = sorted(image_dir.glob("scene-*.png"))
    if not frames:
        sys.exit(f"ERROR: No scene-NNNNN.png files found in {image_dir}")

    print(f"Found {len(frames)} frames in {image_dir}")

    prev_gray = None
    for idx, fpath in enumerate(frames):
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            print(f"  WARNING: could not read {fpath.name}, skipping")
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        motion = _frame_diff_magnitude(prev_gray, gray)
        prev_gray = gray

        # Person detection (class 0 in COCO)
        results = model(img_bgr, classes=[0], verbose=False)
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().tolist() if results[0].boxes else []
        count = len(boxes_xyxy)
        mean_iou = _mean_pairwise_iou(boxes_xyxy)

        yield {
            "frame_idx": idx,
            "timestamp_s": round(idx / fps, 4),
            "count": count,
            "mean_iou": mean_iou,
            "motion": motion,
            "frame_name": fpath.name,
        }


def window_aggregate(per_frame_rows, window: int):
    """
    Slides a window over per-frame rows and yields one dict per window
    containing the 7 aggregate features.
    """
    counts = [r["count"] for r in per_frame_rows]
    ious = [r["mean_iou"] for r in per_frame_rows]
    motions = [r["motion"] for r in per_frame_rows]
    n = len(per_frame_rows)

    for start in range(0, n - window + 1, window // 2):  # 50 % overlap
        end = start + window
        c = counts[start:end]
        o = ious[start:end]
        m = motions[start:end]

        feat = {
            "window_start_frame": per_frame_rows[start]["frame_idx"],
            "window_end_frame": per_frame_rows[min(end - 1, n - 1)]["frame_idx"],
            "window_start_ts": per_frame_rows[start]["timestamp_s"],
            # 7 model features
            "N_mean": float(np.mean(c)),
            "N_std": float(np.std(c)),
            "overlap_mean": float(np.mean(o)),
            "overlap_std": float(np.std(o)),
            "motion_mean": float(np.mean(m)),
            "motion_std": float(np.std(m)),
            "delta_N": float(c[-1] - c[0]) if len(c) > 0 else 0.0,
        }
        yield feat


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequence features from a frame directory."
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Path to directory containing scene-NNNNN.png frames."
    )
    parser.add_argument(
        "--fps", required=True, type=float,
        help=(
            "Source video FPS. Must match README value for this scene "
            "(Times Square 29.97, Las Vegas 15.17, Love Parade 25, Italy 25)."
        ),
    )
    parser.add_argument(
        "--window", type=int, default=30,
        help="Sliding window length in frames (default: 30)."
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Output CSV file path."
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        sys.exit(f"ERROR: {image_dir} is not a directory")

    print(f"fps={args.fps}, window={args.window} frames "
          f"({args.window / args.fps:.1f}s)")

    per_frame = list(extract_per_frame(image_dir, args.fps))
    if len(per_frame) < args.window:
        sys.exit(
            f"ERROR: only {len(per_frame)} frames, need at least {args.window}"
        )

    windows = list(window_aggregate(per_frame, args.window))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["window_start_frame", "window_end_frame", "window_start_ts"]
        + FEATURE_NAMES
    )
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in windows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote {len(windows)} windows to {out}")
    # Verify no NaN
    import csv as _csv
    with open(out) as f:
        rows = list(_csv.DictReader(f))
    nan_count = sum(
        1 for r in rows for k in FEATURE_NAMES if r[k] in ("nan", "inf", "")
    )
    if nan_count:
        print(f"WARNING: {nan_count} NaN/inf values detected in output.")
    else:
        print("Validation: no NaN/inf values found.")


if __name__ == "__main__":
    main()
