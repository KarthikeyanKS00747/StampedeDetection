"""Save source frames and matching YOLO person detections for manual review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("classifier/yolov8n.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/person_audit"))
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not args.weights.is_file():
        parser.error(f"Detector weights not found: {args.weights}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(args.weights))
    rows = []

    for video in args.videos:
        if not video.is_file():
            parser.error(f"Video not found: {video}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if total < 1 or fps <= 0:
                raise ValueError(f"Invalid video metadata: {video}")
            indices = sorted({round(i * (total - 1) / max(args.samples - 1, 1))
                              for i in range(args.samples)})
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot read frame {index} of {video}")
                result = detector(frame, classes=[0], verbose=False)[0]
                stem = f"{video.stem}_frame_{index:05d}"
                original = args.output_dir / f"{stem}_original.jpg"
                detected = args.output_dir / f"{stem}_detected.jpg"
                if not cv2.imwrite(str(original), frame):
                    raise RuntimeError(f"Cannot save {original}")
                if not cv2.imwrite(str(detected), result.plot()):
                    raise RuntimeError(f"Cannot save {detected}")
                rows.append({
                    "video": video.name,
                    "frame": index,
                    "timestamp_s": round(index / fps, 3),
                    "fps": round(fps, 6),
                    "detected_people": len(result.boxes),
                    "original": str(original),
                    "detected": str(detected),
                })
        finally:
            capture.release()

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} frame pairs and {manifest}")


if __name__ == "__main__":
    main()
