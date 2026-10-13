from __future__ import annotations

import logging
from pathlib import Path

from shared.schemas import RiskLevel

LOG = logging.getLogger(__name__)


class GPIOAlarm:
    """BCM pin controller; automatically becomes a log-only alarm outside Raspberry Pi OS."""

    GREEN_PIN, RED_PIN, BUZZER_PIN = 17, 27, 22

    def __init__(self) -> None:
        # Avoid gpiozero's slow pin-factory probing on laptops and CI.
        if not Path("/proc/device-tree/model").exists():
            LOG.info("GPIO alarm in simulation mode (not Raspberry Pi OS)")
            self.available = False
            return
        try:
            from gpiozero import Buzzer, LED
            self.green, self.red, self.buzzer = LED(self.GREEN_PIN), LED(self.RED_PIN), Buzzer(self.BUZZER_PIN)
            self.available = True
        except (ImportError, RuntimeError, OSError) as error:
            LOG.warning("GPIO alarm in simulation mode: %s", error)
            self.available = False

    def set_risk(self, risk: RiskLevel, alert_active: bool) -> None:
        if not self.available:
            LOG.info("GPIO simulated: %s alert=%s", risk.value, alert_active)
            return
        self.green.value = risk in (RiskLevel.NORMAL, RiskLevel.MODERATE)
        self.red.value = risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        self.buzzer.value = alert_active

    def close(self) -> None:
        if self.available:
            self.green.close(); self.red.close(); self.buzzer.close()
