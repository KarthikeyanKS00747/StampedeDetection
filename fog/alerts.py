from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

from shared.schemas import AlertEvent, RiskLevel

_SEVERITY = {RiskLevel.NORMAL: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}


class AlertEngine:
    def __init__(self) -> None:
        self._history: dict[str, deque[RiskLevel]] = defaultdict(lambda: deque(maxlen=3))
        self._active: dict[str, bool] = defaultdict(bool)

    def evaluate(self, zone_id: str, risk: RiskLevel, reasons: list[str]) -> AlertEvent | None:
        history = self._history[zone_id]
        history.append(risk)
        active = risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        # A sustained Moderate condition escalates to an operational alert.
        if len(history) == 3 and all(level == RiskLevel.MODERATE for level in history):
            active = True
        changed = active != self._active[zone_id]
        self._active[zone_id] = active
        if not changed and not (active and risk == RiskLevel.CRITICAL):
            return None
        action = "ALERT" if active else "CLEARED"
        detail = "; ".join(reasons)
        return AlertEvent(timestamp=datetime.now(timezone.utc), zone_id=zone_id, risk=risk, active=active, message=f"{action}: {risk.value}. {detail}")

    def is_active(self, zone_id: str) -> bool:
        return self._active[zone_id]
