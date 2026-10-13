from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import Iterable

import numpy as np

from shared.schemas import FeatureVector


def circular_variance(vectors: Iterable[tuple[float, float]]) -> float:
    angles = [atan2(y, x) for x, y in vectors if x or y]
    if len(angles) < 2:
        return 0.0
    resultant = (sum(cos(a) for a in angles) ** 2 + sum(sin(a) for a in angles) ** 2) ** 0.5 / len(angles)
    return float(max(0.0, min(1.0, 1.0 - resultant)))


@dataclass
class FeatureExtractor:
    area_m2: float = 25.0
    history_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.area_m2 <= 0:
            raise ValueError("area_m2 must be positive")
        self._density_history: deque[tuple[float, float]] = deque()

    def extract(
        self,
        zone_id: str,
        timestamp_s: float,
        count: int,
        track_vectors: Iterable[tuple[float, float]],
        flow_vectors: Iterable[tuple[float, float]],
        source: str = "camera",
    ) -> FeatureVector:
        density = count / self.area_m2
        self._density_history.append((timestamp_s, density))
        while self._density_history and timestamp_s - self._density_history[0][0] > self.history_seconds:
            self._density_history.popleft()
        oldest_t, oldest_density = self._density_history[0]
        elapsed = max(timestamp_s - oldest_t, 1e-6)
        growth = (density - oldest_density) / elapsed
        vectors = list(track_vectors) or list(flow_vectors)
        speeds = [float(np.hypot(x, y)) for x, y in vectors]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        variance = float(np.var(speeds)) if len(speeds) > 1 else 0.0
        direction = circular_variance(vectors)
        # Bounded, explainable blend; calibrate weights with labelled footage before deployment.
        congestion = min(1.0, 0.35 * min(density / 4.0, 1.0) + 0.25 * min(max(growth, 0) / 0.5, 1.0) + 0.2 * direction + 0.2 * min(variance / 400.0, 1.0))
        return FeatureVector(
            zone_id=zone_id,
            crowd_count=count,
            density=density,
            density_growth_rate=growth,
            average_velocity=mean_speed,
            velocity_variance=variance,
            direction_variance=direction,
            congestion_index=congestion,
            source=source,  # type: ignore[arg-type]
        )
