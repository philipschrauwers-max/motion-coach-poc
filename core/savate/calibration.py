# core/savate/calibration.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Calibration:
    # Guard baseline
    guard_left: float
    guard_right: float
    guard_left_std: float
    guard_right_std: float

    # Balance / stance baselines (keep for your existing summary/logic)
    head_over_hips_x: float
    head_over_hips_x_std: float
    stance_width: float
    stance_width_std: float

    # Auto thresholds for punch detection (NEW)
    punch_idle_p95: float
    punch_peak_median: float
    punch_speed_start: float
    punch_speed_end: float


class CalibrationManager:
    """
    Two-phase calibration:
      - GUARD: stand in guard for duration_guard_s
      - TEST: throw N punches; detect wrist speed peaks and derive thresholds
    """

    def __init__(self, duration_guard_s: float = 2.0, max_test_s: float = 6.0, target_peaks: int = 5):
        self.duration_guard_s = duration_guard_s
        self.max_test_s = max_test_s
        self.target_peaks = target_peaks

        self.collecting = False
        self.phase = "IDLE"  # IDLE | GUARD | TEST

        self._t0 = 0.0
        self._buf_guard: List = []
        self._buf_test: List = []
        self._peaks: List[float] = []

        # rolling window for streaming peak detection
        self._last_speeds: List[float] = []

        # derived from guard phase
        self._idle_p95 = 0.0
        self._idle_mean = 0.0

    def start(self):
        self.collecting = True
        self.phase = "GUARD"
        self._t0 = 0.0
        self._buf_guard = []
        self._buf_test = []
        self._peaks = []
        self._last_speeds = []
        self._idle_p95 = 0.0
        self._idle_mean = 0.0

    def status_text(self) -> str:
        if not self.collecting:
            return "Press K to calibrate"
        if self.phase == "GUARD":
            return "Calibrating: hold guard still…"
        if self.phase == "TEST":
            return f"Calibration: throw {self.target_peaks} straight punches (60%)… ({len(self._peaks)}/{self.target_peaks})"
        return "Calibrating…"

    def progress(self) -> float:
        """0..1 across both phases"""
        if not self.collecting:
            return 0.0
        if self.phase == "GUARD":
            # guard phase is first half
            return 0.5 * min(1.0, max(0.0, (self._elapsed() / self.duration_guard_s)))
        if self.phase == "TEST":
            # second half depends on peaks collected
            return 0.5 + 0.5 * (len(self._peaks) / float(self.target_peaks))
        return 0.0

    def _elapsed(self) -> float:
        if not self._buf_guard and not self._buf_test:
            return 0.0
        # we store first sample timestamp as start for each phase
        if self.phase == "GUARD" and self._buf_guard:
            return self._buf_guard[-1].t - self._buf_guard[0].t
        if self.phase == "TEST" and self._buf_test:
            return self._buf_test[-1].t - self._buf_test[0].t
        return 0.0

    def update(self, sig) -> Optional[Calibration]:
        if not self.collecting:
            return None

        if self.phase == "GUARD":
            self._buf_guard.append(sig)
            # Start timer once we have the first sample
            if len(self._buf_guard) >= 2 and self._elapsed() >= self.duration_guard_s:
                # derive idle wrist noise from GUARD
                speeds = np.array(
                    [max(s.l_wrist_speed, s.r_wrist_speed) for s in self._buf_guard],
                    dtype=np.float32
                )
                self._idle_p95 = float(np.percentile(speeds, 95))
                self._idle_mean = float(np.mean(speeds))

                # move to TEST phase
                self.phase = "TEST"
                self._buf_test = []
                self._last_speeds = []
                return None

            return None

        if self.phase == "TEST":
            self._buf_test.append(sig)

            # track max wrist speed this frame
            sp = float(max(sig.l_wrist_speed, sig.r_wrist_speed))
            self._last_speeds.append(sp)
            if len(self._last_speeds) > 3:
                self._last_speeds.pop(0)

            # simple peak detection using a 3-sample window:
            # if middle sample is a local maximum AND above a dynamic floor, count it
            if len(self._last_speeds) == 3:
                a, b, c = self._last_speeds
                # floor is idle noise + small margin
                floor = self._idle_p95 + max(0.008, 0.75 * (self._idle_p95 - self._idle_mean))
                if b > a and b > c and b > floor:
                    # debounce: avoid counting the same punch twice
                    if not self._peaks or (b > (self._peaks[-1] * 0.65)):
                        self._peaks.append(b)

            # stop conditions:
            test_elapsed = self._elapsed()
            if len(self._peaks) >= self.target_peaks or test_elapsed >= self.max_test_s:
                cal = self._compute()
                self.collecting = False
                self.phase = "IDLE"
                return cal

            return None

        return None

    def _compute(self) -> Calibration:
        # --- Guard/balance/stance baselines ---
        gL = np.array([s.guard_left for s in self._buf_guard], dtype=np.float32)
        gR = np.array([s.guard_right for s in self._buf_guard], dtype=np.float32)
        hoh = np.array([s.head_over_hips_x for s in self._buf_guard], dtype=np.float32)
        stance = np.array([s.stance_width for s in self._buf_guard], dtype=np.float32)

        guard_left = float(np.mean(gL))
        guard_right = float(np.mean(gR))
        guard_left_std = float(np.std(gL) + 1e-6)
        guard_right_std = float(np.std(gR) + 1e-6)

        head_over_hips_x = float(np.mean(hoh))
        head_over_hips_x_std = float(np.std(hoh) + 1e-6)

        stance_width = float(np.mean(stance))
        stance_width_std = float(np.std(stance) + 1e-6)

        # --- Auto punch thresholds ---
        if self._peaks:
            peak_med = float(np.median(np.array(self._peaks, dtype=np.float32)))
        else:
            # fallback if user didn't throw punches / detection missed
            peak_med = max(self._idle_p95 + 0.02, self._idle_p95 * 2.5)

        idle = float(self._idle_p95)

        # thresholds are fractions between idle noise and typical strike peak
        # (these numbers are easy to tune but work well across devices)
        speed_start = idle + 0.28 * (peak_med - idle)
        speed_end = idle + 0.12 * (peak_med - idle)

        # sanity bounds
        speed_start = float(max(speed_start, idle + 0.01))
        speed_end = float(max(min(speed_end, speed_start * 0.85), idle + 0.006))

        return Calibration(
            guard_left=guard_left,
            guard_right=guard_right,
            guard_left_std=guard_left_std,
            guard_right_std=guard_right_std,

            head_over_hips_x=head_over_hips_x,
            head_over_hips_x_std=head_over_hips_x_std,
            stance_width=stance_width,
            stance_width_std=stance_width_std,

            punch_idle_p95=idle,
            punch_peak_median=peak_med,
            punch_speed_start=speed_start,
            punch_speed_end=speed_end,
        )
