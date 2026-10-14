"""Research-only ML adapter. Never translates anomaly scores into risk classes."""
from datetime import datetime, timezone
import math

from pydantic import BaseModel, Field, ConfigDict

NAMES = ["speed_median", "speed_p90", "speed_std", "moving_fraction", "direction_variance",
         "detected_count", "box_coverage_proxy", "max_local_count_proxy", "track_speed_proxy"]


class AnomalyWindow(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    zone_id: str = Field(min_length=1, max_length=64)
    clip_id: str = Field(min_length=1, max_length=100)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    feature_names: list[str]
    features: list[float]
    quality_eligible: bool
    quality_reason: str = Field(default="", max_length=1000)


class AnomalyClassifier:
    def __init__(self, path):
        import joblib
        # joblib uses pickle: load only locally generated, trusted artifacts.
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("artifact_kind") != "experimental_quality_motion_anomaly_v1":
            raise ValueError("Expected a quality-gated experimental anomaly bundle")
        if bundle.get("variant") != "joint" or bundle.get("feature_names") != NAMES:
            raise ValueError("Anomaly feature contract mismatch")
        if bundle.get("quality_gate_required") is not True or bundle.get("deployable") is not False:
            raise ValueError("Expected research-only quality gating")
        self.threshold = float(bundle["threshold"])
        if not math.isfinite(self.threshold):
            raise ValueError("Nonfinite threshold")
        self.pipeline = bundle["pipeline"]

    def assess(self, window: AnomalyWindow):
        if window.end_s <= window.start_s:
            raise ValueError("Window end must follow start")
        if window.feature_names != NAMES or len(window.features) != len(NAMES):
            raise ValueError("Input feature contract mismatch")
        if not all(math.isfinite(value) for value in window.features):
            raise ValueError("Features must be finite")
        score = None
        if window.quality_eligible:
            score = float(-self.pipeline.score_samples([window.features])[0])
            if not math.isfinite(score):
                raise ValueError("Model returned nonfinite score")
        exceeded = None if score is None else score > self.threshold
        return {**window.model_dump(exclude={"features", "feature_names"}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classifier_mode": "experimental_ml", "score": score,
                "threshold": self.threshold, "exceeds_reference_threshold": exceeded,
                "observation": "Cannot assess" if score is None else "Review unusual motion" if exceeded else "Within reference distribution",
                "risk": None, "risk_probabilities": None, "hardware_alarm_enabled": False,
                "warning": "Research model: scene change can cause anomalies; this is not a safe/danger classification."}
