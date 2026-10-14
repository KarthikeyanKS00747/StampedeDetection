"""Shared motion units for offline and online feature extraction."""
from __future__ import annotations

import numpy as np


def velocities(displacements, delta_s: float) -> list[tuple[float, float]]:
    """Convert pixel displacement vectors to pixels/second, retaining zero flow."""
    if not np.isfinite(delta_s) or delta_s <= 0:
        raise ValueError("delta_s must be finite and positive")
    array = np.asarray(displacements, dtype=float).reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError("Motion vectors must be finite")
    return [tuple(vector) for vector in array / delta_s]


def frame_time(frame_index: int, fps: float) -> float:
    """Constant-frame-rate media time; not time spent executing the detector."""
    if frame_index < 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonnegative frame_index and finite positive fps required")
    return frame_index / fps
