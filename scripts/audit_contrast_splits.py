"""Offline CLAHE + zoomed-crop candidate audit; never alters the live classifier.

Example: python -m scripts.audit_contrast_splits --image outputs/person_audit/FILE_original.jpg
Extra or split boxes are hypotheses, not verified people counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.compare_person_resolution import detect, detect_tiled, save_overlay


def enhance(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = list(cv2.split(lab))
    channels[0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(channels[0])
    return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_LAB2BGR)


def possible_splits(parents: np.ndarray, children: np.ndarray) -> list[tuple[int, list[int]]]:
    """Find original boxes containing two smaller contrast candidate centres."""
    findings = []
    for index, (x1, y1, x2, y2) in enumerate(parents):
        area = (x2 - x1) * (y2 - y1)
        contained = []
        for child_index, (a, b, c, d) in enumerate(children):
            cx, cy = (a + c) / 2, (b + d) / 2
            child_area = (c - a) * (d - b)
            if x1 <= cx <= x2 and y1 <= cy <= y2 and child_area < 0.8 * area:
                contained.append(child_index)
        if len(contained) >= 2:
            findings.append((index, contained))
    return findings


def audit(image: np.ndarray, detector: YOLO, mode: str, imgsz: int,
          tile: int, overlap: float) -> dict:
    run = (lambda picture: detect_tiled(detector, picture, tile, overlap, imgsz)
           if mode == "tiled" else detect(detector, picture, imgsz))
    start = time.perf_counter()
    original_boxes, original_scores = run(image)
    original_s = time.perf_counter() - start
    contrasted = enhance(image)
    start = time.perf_counter()
    contrast_boxes, contrast_scores = run(contrasted)
    contrast_s = time.perf_counter() - start
    return {"original_boxes": original_boxes, "original_scores": original_scores,
            "contrast_boxes": contrast_boxes, "contrast_scores": contrast_scores,
            "contrast_image": contrasted,
            "possible_splits": possible_splits(original_boxes, contrast_boxes),
            "original_s": original_s, "contrast_s": contrast_s}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("classifier/yolov8n.pt"))
    parser.add_argument("--mode", choices=("full", "tiled"), default="full")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contrast_audit"))
    args = parser.parse_args()
    if not args.image.is_file() or not args.weights.is_file():
        parser.error("Image and detector weights must exist")
    if args.imgsz < 64 or args.tile < 64 or not 0 <= args.overlap < 0.5:
        parser.error("Invalid image size, tile, or overlap")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Cannot read image")
    result = audit(image, YOLO(str(args.weights)), args.mode, args.imgsz,
                   args.tile, args.overlap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image.stem}_{args.mode}"
    save_overlay(image, result["original_boxes"], result["original_scores"],
                 args.output_dir / f"{stem}_original_boxes.jpg")
    save_overlay(result["contrast_image"], result["contrast_boxes"],
                 result["contrast_scores"], args.output_dir / f"{stem}_contrast_boxes.jpg")
    path = args.output_dir / f"{stem}_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "mode", "original_boxes",
            "contrast_boxes", "possible_split_parents", "original_s", "contrast_s"])
        writer.writeheader()
        writer.writerow({"image": str(args.image), "mode": args.mode,
            "original_boxes": len(result["original_boxes"]),
            "contrast_boxes": len(result["contrast_boxes"]),
            "possible_split_parents": len(result["possible_splits"]),
            "original_s": round(result["original_s"], 3),
            "contrast_s": round(result["contrast_s"], 3)})
    print(f"Original: {len(result['original_boxes'])} candidate boxes in {result['original_s']:.3f}s")
    print(f"CLAHE: {len(result['contrast_boxes'])} candidate boxes in {result['contrast_s']:.3f}s")
    print(f"Possible split parents (unverified): {len(result['possible_splits'])}")
    print(path)


if __name__ == "__main__":
    main()
