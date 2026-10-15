"""Run the project's dense-crowd person detector on one image."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    detector = YOLO(str(args.weights))
    boxes = detect_tiled(detector, image, conf=args.conf)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)),
                      (0, 255, 255), 2)
    heading = f"Detected people: {len(boxes)} (YOLO conf >= {args.conf:.2f})"
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(image, heading, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Cannot write {args.output}")
    print(f"detected_people={len(boxes)}")
    print(args.output)


if __name__ == "__main__":
    main()
