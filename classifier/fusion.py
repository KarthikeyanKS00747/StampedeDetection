"""Experimental agreement gate for the crowd baseline and sequence classifier.

This is a decision rule, not a trained or validated stampede model. It never
changes the operational risk tier or drives GPIO.
"""

from dataclasses import dataclass

from shared.schemas import RiskLevel


@dataclass(frozen=True)
class FusionDecision:
    anomaly_episode_active: bool
    combined_candidate: bool
    combined_episode_started: bool
    combined_episode_id: int | None


class FusionPolicy:
    """Apply the draft 3-positive/5-normal policy to 15-frame strides."""

    def __init__(self) -> None:
        self.last_start_frame: int | None = None
        self.positive_streak = 0
        self.normal_streak = 0
        self.anomaly_episode_active = False
        self.combined_candidate = False
        self.combined_episode_count = 0

    def observe(
        self, start_frame: int, abnormal: bool, baseline_risk: RiskLevel | None
    ) -> FusionDecision:
        if start_frame < 0:
            raise ValueError("start_frame must be nonnegative")
        if self.last_start_frame is not None:
            if start_frame <= self.last_start_frame:
                raise ValueError("windows must be in increasing frame order")
            if start_frame != self.last_start_frame + 15:
                # A missing window breaks both streaks. Never bridge a gap.
                self.positive_streak = 0
                self.normal_streak = 0
                self.anomaly_episode_active = False
                self.combined_candidate = False
        self.last_start_frame = start_frame

        if abnormal:
            self.positive_streak += 1
            self.normal_streak = 0
            if self.positive_streak >= 3:
                self.anomaly_episode_active = True
        else:
            self.positive_streak = 0
            if self.anomaly_episode_active:
                self.normal_streak += 1
                if self.normal_streak >= 5:
                    self.anomaly_episode_active = False
                    self.normal_streak = 0

        candidate = self.anomaly_episode_active and baseline_risk in (
            RiskLevel.HIGH, RiskLevel.CRITICAL
        )
        started = candidate and not self.combined_candidate
        if started:
            self.combined_episode_count += 1
        self.combined_candidate = candidate
        return FusionDecision(
            anomaly_episode_active=self.anomaly_episode_active,
            combined_candidate=candidate,
            combined_episode_started=started,
            combined_episode_id=self.combined_episode_count if candidate else None,
        )
