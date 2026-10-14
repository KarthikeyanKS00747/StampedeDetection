from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a labelled contact sheet from a crowd video for human review")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1
    samples = np.linspace(0, max(total - 1, 0), args.frames, dtype=int)
    tiles = []
    for index in samples:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        scale = min(320 / width, 180 / height)
        resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        top, left = (180 - resized.shape[0]) // 2, (320 - resized.shape[1]) // 2
        frame[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.putText(frame, f"{index / fps:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        tiles.append(frame)
    capture.release()
    if not tiles:
        raise SystemExit("No frames extracted")
    columns = 3
    rows = ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet = np.vstack([np.hstack(tiles[n:n + columns]) for n in range(0, len(tiles), columns)])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, sheet)
    print(f"Saved {args.output}; source: {total} frames at {fps:.2f} FPS")


if __name__ == "__main__":
    main()
