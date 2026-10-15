"""
tiled_detector.py
-----------------
Tiled YOLO inference for dense crowd scenes.

Why tiling?
  At imgsz=640, a 1920x1080 frame is shrunk so each person becomes ~5-15px tall.
  YOLO struggles to detect partially occluded people at that scale.
  Tiling cuts the frame into overlapping 640px crops before detection, so each
  person appears at a larger apparent scale. Duplicate detections from overlapping
  tiles are merged with NMS.

  This is the single biggest detection improvement for dense crowds where YOLO
  with default settings misses the majority of people.
"""
from __future__ import annotations

import cv2
import numpy as np


# Detection defaults tuned for dense crowd footage
CONF_THRESHOLD = 0.05    # vs ultralytics default 0.25 — keeps occluded people
IOU_NMS        = 0.45    # NMS IoU for merging overlapping tile detections
MAX_DET        = 1000    # vs default 300 — needed for very dense scenes
TILE_SIZE      = 640     # crop size fed to YOLO (matches training resolution)
TILE_OVERLAP   = 0.25    # fraction of tile overlap between adjacent crops


def detect_tiled(
    model,
    frame_bgr: np.ndarray,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_NMS,
    max_det: int = MAX_DET,
    tile: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
) -> np.ndarray:
    """
    Run tiled YOLO person detection on a full frame.

    Returns:
        np.ndarray of shape (N, 4) with [x1,y1,x2,y2] boxes in full-frame pixel
        coordinates, after NMS merge of duplicate detections from overlapping tiles.
        Returns empty (0,4) array if no people detected.
    """
    H, W = frame_bgr.shape[:2]
    step = max(1, int(tile * (1 - overlap)))

    raw_boxes: list[list[float]] = []
    raw_scores: list[float] = []

    # slide tiles across the frame
    ys = list(range(0, max(H - tile, 0) + step, step))
    xs = list(range(0, max(W - tile, 0) + step, step))

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            crop = frame_bgr[y0:y1, x0:x1]
            result = model.predict(
                crop,
                conf=conf,
                iou=iou,
                imgsz=tile,
                max_det=max_det,
                classes=[0],   # person only
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box, score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                # translate crop coordinates back to full-frame coordinates
                raw_boxes.append([
                    float(box[0]) + x0,
                    float(box[1]) + y0,
                    float(box[2]) + x0,
                    float(box[3]) + y0,
                ])
                raw_scores.append(float(score))

    if not raw_boxes:
        return np.empty((0, 4), dtype=np.float32)

    # NMS to remove duplicates from overlapping tiles
    # cv2.dnn.NMSBoxes expects [x, y, w, h] format
    nms_rects = [
        [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
        for b in raw_boxes
    ]
    indices = cv2.dnn.NMSBoxes(nms_rects, raw_scores, conf, iou)
    if len(indices) == 0:
        return np.empty((0, 4), dtype=np.float32)

    idx = np.array(indices).flatten()
    return np.array([raw_boxes[i] for i in idx], dtype=np.float32)


def detect_simple(
    model,
    frame_bgr: np.ndarray,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_NMS,
    max_det: int = MAX_DET,
    imgsz: int = 1280,
) -> np.ndarray:
    """
    Single-pass detection at higher resolution (no tiling).
    Faster than tiling but less effective on very dense footage.
    Use as fallback when speed matters more than count accuracy.
    """
    result = model.predict(
        frame_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        classes=[0],
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return result.boxes.xyxy.cpu().numpy().astype(np.float32)
