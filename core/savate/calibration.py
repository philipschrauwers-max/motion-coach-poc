from dataclasses import dataclass, asdict
from typing import Optional, Dict
import time
import numpy as np

from .signals import FrameSignals


@dataclass
class Calibration:
    started_t: float
    duration_s: float
    # Baselines
    guard_left: float
    guard_right: float
    stance_width: float
    head_over_hips_x: float
    torso_rotation_proxy: float

    # Simple dispersions (helps for robust tolerances)
    guard_left_std: float
    guard_right_std: float
    stance_width_std: float
    head_over_hips_x_std: float

    def to_dict(self) -> Dict:
        return asdict(self)


class CalibrationManager:
    """
    Collects FrameSignals for N seconds while user is in guard/stance,
    then computes stable baselines.
    """
    def __init__(self, duration_s: float = 2.0):
        self.duration_s = float(duration_s)
        self._collecting = False
        self._start_t = 0.0
        self._buf = []

    def start(self):
        self._collecting = True
        self._start_t = time.time()
        self._buf = []

    def cancel(self):
        self._collecting = False
        self._buf = []

    @property
    def collecting(self) -> bool:
        return self._collecting

    def progress(self) -> float:
        if not self._collecting:
            return 0.0
        return min(1.0, max(0.0, (time.time() - self._start_t) / self.duration_s))

    def update(self, sig: FrameSignals) -> Optional[Calibration]:
        if not self._collecting:
            return None

        # Collect
        self._buf.append(sig)

        # Finish?
        if (time.time() - self._start_t) >= self.duration_s and len(self._buf) >= 10:
            cal = self._compute()
            self._collecting = False
            self._buf = []
            return cal

        return None

    def _compute(self) -> Calibration:
        guards_l = np.array([s.guard_left for s in self._buf], dtype=np.float32)
        guards_r = np.array([s.guard_right for s in self._buf], dtype=np.float32)
        stance = np.array([s.stance_width for s in self._buf], dtype=np.float32)
        headx = np.array([s.head_over_hips_x for s in self._buf], dtype=np.float32)
        torso = np.array([s.torso_rotation_proxy for s in self._buf], dtype=np.float32)

        # Robust mean (median is stable under jitter)
        guard_left = float(np.median(guards_l))
        guard_right = float(np.median(guards_r))
        stance_width = float(np.median(stance))
        head_over_hips_x = float(np.median(headx))
        torso_rotation_proxy = float(np.median(torso))

        return Calibration(
            started_t=self._start_t,
            duration_s=self.duration_s,
            guard_left=guard_left,
            guard_right=guard_right,
            stance_width=stance_width,
            head_over_hips_x=head_over_hips_x,
            torso_rotation_proxy=torso_rotation_proxy,
            guard_left_std=float(np.std(guards_l)),
            guard_right_std=float(np.std(guards_r)),
            stance_width_std=float(np.std(stance)),
            head_over_hips_x_std=float(np.std(headx)),
        )
