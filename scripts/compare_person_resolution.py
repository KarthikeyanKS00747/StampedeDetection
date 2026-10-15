"""Compare unchanged YOLOv8n person detection at 640, 1280, and tiled input.

This is an offline detector audit only. Counts are candidate boxes, not human
ground truth and not inputs to the trained anomaly classifier.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    intervals = int(np.ceil((length - tile) / step))
    return sorted({round(i * (length - tile) / intervals)
                   for i in range(intervals + 1)})


def owned_interval(starts: list[int], index: int, tile: int,
                   length: int) -> tuple[float, float]:
    """Partition overlap midpoints so neighbouring tiles cannot claim one centre."""
    low = ((starts[index - 1] + tile + starts[index]) / 2
           if index > 0 else 0.0)
    high = ((starts[index] + tile + starts[index + 1]) / 2
            if index + 1 < len(starts) else float(length))
    return low, high


def nonmaximum_suppression(boxes: np.ndarray, scores: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
    """Return kept indices for xyxy boxes after confidence-ordered IoU NMS."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def detect(detector: YOLO, image: np.ndarray, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    result = detector(image, classes=[0], imgsz=imgsz, verbose=False)[0]
    return (result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())


def detect_tiled(detector: YOLO, image: np.ndarray, tile: int,
                 overlap: float, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    all_boxes = []
    all_scores = []
    y_starts = tile_starts(height, tile, overlap)
    x_starts = tile_starts(width, tile, overlap)
    for y_index, top in enumerate(y_starts):
        ymin, ymax = owned_interval(y_starts, y_index, tile, height)
        for x_index, left in enumerate(x_starts):
            xmin, xmax = owned_interval(x_starts, x_index, tile, width)
            crop = image[top:min(top + tile, height), left:min(left + tile, width)]
            boxes, scores = detect(detector, crop, imgsz)
            if len(boxes):
                boxes[:, [0, 2]] += left
                boxes[:, [1, 3]] += top
                centres = (boxes[:, :2] + boxes[:, 2:]) / 2
                owned = ((centres[:, 0] >= xmin) & (centres[:, 0] < xmax)
                         & (centres[:, 1] >= ymin) & (centres[:, 1] < ymax))
                if owned.any():
                    all_boxes.append(boxes[owned])
                    all_scores.append(scores[owned])
    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nonmaximum_suppression(boxes, scores)
    return boxes[keep], scores[keep]


def save_overlay(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                 path: Path) -> None:
    overlay = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(overlay, f"{score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Cannot write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, default=Path("classifier/yolov8n.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/person_resolution"))
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--tile-imgsz", type=int, default=960)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.video.is_file():
        parser.error("Video and detector weights must exist")
    if args.tile < 64 or not 0 <= args.overlap < 0.5 or args.tile_imgsz < 64:
        parser.error("Invalid tile, overlap, or tile-imgsz")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if any(index < 0 or index >= total for index in args.frames):
        capture.release()
        parser.error(f"Frame indices must be in [0, {total - 1}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(args.weights))
    rows = []
    try:
        for index in args.frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Cannot read frame {index}")
            stem = f"{args.video.stem}_frame_{index:05d}"
            row = {"video": args.video.name, "frame": index,
                   "width": image.shape[1], "height": image.shape[0]}
            for mode, size in (("full640", 640), ("full1280", 1280),
                               ("tiled", args.tile_imgsz)):
                if mode == "tiled":
                    boxes, scores = detect_tiled(detector, image, args.tile,
                                                  args.overlap, size)
                else:
                    boxes, scores = detect(detector, image, size)
                path = args.output_dir / f"{stem}_{mode}.jpg"
                save_overlay(image, boxes, scores, path)
                row[f"{mode}_boxes"] = len(boxes)
                row[f"{mode}_overlay"] = str(path)
                print(f"{stem} {mode}: {len(boxes)} candidate person boxes", flush=True)
            rows.append(row)
    finally:
        capture.release()
    frame_suffix = "_".join(f"{index:05d}" for index in args.frames)
    manifest = args.output_dir / f"{args.video.stem}_frames_{frame_suffix}_comparison.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {manifest}")


if __name__ == "__main__":
    main()
