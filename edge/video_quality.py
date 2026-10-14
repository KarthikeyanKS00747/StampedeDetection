"""Experimental quality checks and geometric camera-motion compensation.

Detector-excluded anchors are only candidate background points; missed people
can still contaminate them. Acceptance is a numerical check, not ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np


def content_mask(gray):
    visible = gray > 12
    rows = np.flatnonzero(visible.mean(axis=1) > .05)
    columns = np.flatnonzero(visible.mean(axis=0) > .05)
    mask = np.zeros(gray.shape, dtype=bool)
    if len(rows) and len(columns):
        mask[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1] = True
    return mask


def candidate_background(gray, boxes, source_shape):
    mask = content_mask(gray).astype(np.uint8) * 255
    height, width = gray.shape
    source_h, source_w = source_shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        x1, x2 = (np.array([x1, x2]) * width / source_w).astype(int)
        y1, y2 = (np.array([y1, y2]) * height / source_h).astype(int)
        margin = 4
        mask[max(0, y1 - margin):min(height, y2 + margin), max(0, x1 - margin):min(width, x2 + margin)] = 0
    return mask


def estimate_camera(previous, current, background):
    info = {"accepted": False, "reason": "insufficient_anchors", "anchors": 0,
            "inlier_fraction": 0.0, "quadrants": 0, "translation_px": 0.0, "scale": 1.0}
    points = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=.02,
                                   minDistance=6, mask=background)
    if points is None or len(points) < 16:
        return None, info
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                   winSize=(21, 21), maxLevel=3)
    if next_points is None:
        return None, info
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, next_points, None,
                                                         winSize=(21, 21), maxLevel=3)
    if back_points is None:
        return None, info
    valid = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    valid &= np.linalg.norm(points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1) < 1.0
    a, b = points.reshape(-1, 2)[valid], next_points.reshape(-1, 2)[valid]
    info["anchors"] = len(a)
    if len(a) < 16:
        return None, info
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                ransacReprojThreshold=1.5, maxIters=2000,
                                                confidence=.99)
    if matrix is None or inliers is None:
        info["reason"] = "affine_fit_failed"
        return None, info
    good = inliers.ravel().astype(bool)
    height, width = previous.shape
    quadrants = {(int(x >= width / 2), int(y >= height / 2)) for x, y in a[good]}
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    translation = float(np.linalg.norm(matrix[:, 2]))
    info.update(inlier_fraction=float(good.mean()), quadrants=len(quadrants),
                translation_px=translation, scale=scale)
    if good.mean() < .6 or len(quadrants) < 3:
        info["reason"] = "poor_or_localised_support"
    elif not .96 <= scale <= 1.04 or rotation > 3 or translation > .1 * np.hypot(height, width):
        info["reason"] = "implausible_camera_step"
    else:
        info.update(accepted=True, reason="numerically_supported_candidate")
        return matrix, info
    return None, info


def compensate(previous, current, background):
    matrix, info = estimate_camera(previous, current, background)
    if matrix is None:
        return None, np.zeros(previous.shape, dtype=bool), info
    height, width = previous.shape
    corrected = cv2.warpAffine(current, matrix, (width, height),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    overlap = cv2.warpAffine(content_mask(current).astype(np.uint8), matrix, (width, height),
                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
    return corrected, content_mask(previous) & overlap, info
