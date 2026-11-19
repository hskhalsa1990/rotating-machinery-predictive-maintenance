"""ISO 10816-style vibration severity zones and alarm logic.

IMPORTANT: the zone limits below are ILLUSTRATIVE values in the style of ISO 10816-1/-3
(velocity RMS, mm/s). ISO 10816 has since been superseded by ISO 20816. Always take
limits from the current standard / machine manufacturer for real decisions.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# (A/B limit, B/C limit, C/D limit) in mm/s RMS
THRESHOLDS = {
    "small_class_I": (0.71, 1.8, 4.5),
    "medium_rigid": (1.4, 2.8, 4.5),
    "large_rigid": (2.3, 4.5, 7.1),
}
ZONE_MEANING = {"A": "new machine condition", "B": "acceptable, unrestricted operation",
                "C": "restricted operation, plan maintenance", "D": "damage possible, act now"}
_STATE_FOR_ZONE = {"A": "NORMAL", "B": "NORMAL", "C": "ALERT", "D": "DANGER"}
_RANK = {"NORMAL": 0, "ALERT": 1, "DANGER": 2}


def zone(vel_rms: float, machine_class: str = "medium_rigid") -> str:
    ab, bc, cd = THRESHOLDS[machine_class]
    if vel_rms < ab:
        return "A"
    if vel_rms < bc:
        return "B"
    if vel_rms < cd:
        return "C"
    return "D"


@dataclass
class AlarmLogic:
    """Zone alarm with persistence (avoid single-spike false alarms) and hysteresis,
    plus a simple trend flag (recent mean far above the previous window)."""
    machine_class: str = "medium_rigid"
    confirm_n: int = 3      # consecutive readings needed to escalate
    clear_n: int = 3        # consecutive readings needed to de-escalate
    trend_window: int = 5
    trend_ratio: float = 1.5

    def run(self, vel) -> pd.DataFrame:
        vel = np.asarray(vel, float)
        zones = [zone(v, self.machine_class) for v in vel]
        state, up, down, states = "NORMAL", 0, 0, []
        for z in zones:
            target = _STATE_FOR_ZONE[z]
            if _RANK[target] > _RANK[state]:
                up, down = up + 1, 0
                if up >= self.confirm_n:
                    state, up = target, 0
            elif _RANK[target] < _RANK[state]:
                down, up = down + 1, 0
                if down >= self.clear_n:
                    state, down = target, 0
            else:
                up = down = 0
            states.append(state)
        w = self.trend_window
        trend = [False] * len(vel)
        for i in range(2 * w - 1, len(vel)):
            recent, prev = vel[i - w + 1:i + 1].mean(), vel[i - 2 * w + 1:i - w + 1].mean()
            trend[i] = bool(recent > self.trend_ratio * max(prev, 1e-9))
        return pd.DataFrame({"vel_rms": vel, "zone": zones, "state": states, "trend_flag": trend})
