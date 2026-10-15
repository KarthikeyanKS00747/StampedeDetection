"""Render representative video frames with detector and classifier evidence."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.tiled_detector import detect_tiled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    with args.results.open(newline="", encoding="utf-8") as stream:
        windows = list(csv.DictReader(stream))
    if not windows:
        parser.error("Classifier result CSV is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(args.weights))
    for frame_index in args.frames:
        source = args.frames_dir / f"scene-{frame_index:05d}.png"
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Cannot read {source}")
        timestamp = frame_index / args.fps
        window = min(windows, key=lambda row: abs(float(row["window_start_ts"]) - timestamp))
        boxes = detect_tiled(detector, image, conf=0.05)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)),
                          (0, 255, 255), 2)

        label = window["predicted_label"].upper()
        confidence = float(window["confidence"])
        lines = [
            f"Video time: {timestamp:.2f}s",
            f"Detected people: {len(boxes)} (YOLO conf >= 0.05)",
            f"Classifier: {label} ({confidence:.1%})",
            f"Window start: {float(window['window_start_ts']):.2f}s",
        ]
        cv2.rectangle(image, (0, 0), (image.shape[1], 112), (0, 0, 0), -1)
        for index, text in enumerate(lines):
            cv2.putText(image, text, (12, 25 + index * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        output = args.output_dir / f"evidence_{frame_index:05d}.jpg"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Cannot write {output}")
        print(output)


if __name__ == "__main__":
    main()
