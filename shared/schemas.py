from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    NORMAL = "Normal"
    MODERATE = "Moderate"
    HIGH = "High Risk"
    CRITICAL = "Critical"


FEATURE_COLUMNS = [
    "crowd_count",
    "density",
    "density_growth_rate",
    "average_velocity",
    "velocity_variance",
    "direction_variance",
    "congestion_index",
]


class FeatureVector(BaseModel):
    zone_id: str = Field(default="zone-1", min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    crowd_count: int = Field(ge=0)
    density: float = Field(ge=0, description="Persons per configured square metre")
    density_growth_rate: float = Field(description="Persons/m²/s")
    average_velocity: float = Field(ge=0, description="Pixels/s or calibrated metres/s")
    velocity_variance: float = Field(ge=0)
    direction_variance: float = Field(ge=0, le=1, description="Circular variance 0..1")
    congestion_index: float = Field(ge=0, le=1)
    source: Literal["camera", "video", "simulation"] = "camera"


class RiskAssessment(BaseModel):
    zone_id: str
    timestamp: datetime
    risk: RiskLevel
    probabilities: dict[RiskLevel, float]
    reasons: list[str]
    alert_active: bool
    alert_message: str | None = None
    metrics: FeatureVector


class AlertEvent(BaseModel):
    timestamp: datetime
    zone_id: str
    risk: RiskLevel
    message: str
    active: bool
