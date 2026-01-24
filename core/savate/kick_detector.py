from dataclasses import dataclass
from typing import Optional, Dict, List

from .signals import FrameSignals
from .config import SavateConfig
from .calibration import Calibration
from .punch_detector import RepResult


@dataclass
class KickDetector:
    cfg: SavateConfig
    mode: str = "auto"          # "fouette" or "auto"
    calibration: Optional[Calibration] = None

    state: str = "IDLE"
    active_leg: Optional[str] = None  # "left" or "right"

    start_t: float = 0.0
    peak_ankle_speed: float = 0.0
    peak_knee_ang: float = 0.0
    peak_rot_delta: float = 0.0
    min_stance_width: float = 999.0
    start_rot: float = 0.0

    chamber_ok: bool = False

    # --------------------
    # Public API
    # --------------------

    def set_mode(self, mode: str):
        self.mode = mode
        self.reset()

    def set_calibration(self, cal: Optional[Calibration]):
        self.calibration = cal

    def reset(self):
        self.state = "IDLE"
        self.active_leg = None
        self.start_t = 0.0
        self.peak_ankle_speed = 0.0
        self.peak_knee_ang = 0.0
        self.peak_rot_delta = 0.0
        self.min_stance_width = 999.0
        self.start_rot = 0.0
        self.chamber_ok = False

    # --------------------
    # Main update
    # --------------------

    def update(self, sig: FrameSignals) -> Optional[RepResult]:
        # Determine which leg is moving more
        leg = "left" if sig.l_ankle_speed >= sig.r_ankle_speed else "right"
        ankle_speed = sig.l_ankle_speed if leg == "left" else sig.r_ankle_speed
        knee_ang = sig.l_knee_ang if leg == "left" else sig.r_knee_ang

        # Chamber proxy: knee is folded (bent)
        chamber_like = knee_ang < 140.0

        # Track peaks while active
        if self.state != "IDLE":
            self.peak_ankle_speed = max(self.peak_ankle_speed, ankle_speed)
            self.peak_knee_ang = max(self.peak_knee_ang, knee_ang)
            self.min_stance_width = min(self.min_stance_width, sig.stance_width)
            self.peak_rot_delta = max(
                self.peak_rot_delta,
                abs(sig.torso_rotation_proxy - self.start_rot),
            )
            if chamber_like:
                self.chamber_ok = True

        # --------------------
        # IDLE → CHAMBER
        # --------------------
        if self.state == "IDLE":
            if ankle_speed > self.cfg.ankle_speed_start:
                self.state = "CHAMBER"
                self.active_leg = leg
                self.start_t = sig.t
                self.start_rot = sig.torso_rotation_proxy
                self.peak_ankle_speed = ankle_speed
                self.peak_knee_ang = knee_ang
                self.min_stance_width = sig.stance_width
                self.chamber_ok = chamber_like
            return None

        # Lock onto active leg
        active = self.active_leg or leg
        ankle_speed = sig.l_ankle_speed if active == "left" else sig.r_ankle_speed
        knee_ang = sig.l_knee_ang if active == "left" else sig.r_knee_ang

        # --------------------
        # CHAMBER → EXTEND
        # --------------------
        if self.state == "CHAMBER":
            if knee_ang > 150.0 and ankle_speed > self.cfg.ankle_speed_start:
                self.state = "EXTEND"

            # Abort weak kicks
            if ankle_speed < self.cfg.ankle_speed_end and (sig.t - self.start_t) > 0.4:
                self.reset()
            return None

        # --------------------
        # EXTEND → RECOIL
        # --------------------
        if self.state == "EXTEND":
            if knee_ang >= self.cfg.knee_extended_deg or ankle_speed < self.cfg.ankle_speed_start:
                self.state = "RECOIL"
            return None

        # --------------------
        # RECOIL → DONE
        # --------------------
        if self.state == "RECOIL":
            # Calibration-aware stance return
            if self.calibration:
                baseline = self.calibration.stance_width
                returned = abs(sig.stance_width - baseline) <= (
                    self.cfg.stance_return_allow * baseline
                )
            else:
                returned = sig.stance_width > (self.min_stance_width * 1.05)

            if ankle_speed < self.cfg.ankle_speed_end and returned:
                duration = sig.t - self.start_t
                score, feedback, meta = self._score(sig, duration)
                rep = RepResult(
                    kind="fouette",
                    side=active,
                    score=score,
                    feedback=feedback,
                    meta=meta,
                )
                self.reset()
                return rep

        return None

    # --------------------
    # Scoring
    # --------------------

    def _score(self, sig: FrameSignals, dur: float):
        # Chamber
        chamber_score = 1.0 if self.chamber_ok else 0.5

        # Speed (whip)
        sp = self.peak_ankle_speed
        sp_score = max(0.0, min(1.0, (sp - 0.03) / 0.10))

        # Extension
        ext = self.peak_knee_ang
        ext_score = max(0.0, min(1.0, (ext - 140.0) / 30.0))  # 140→170

        # Pivot / rotation proxy
        rot = self.peak_rot_delta
        rot_score = max(0.0, min(1.0, rot / 25.0))

        # Stability (head over hips)
        if self.calibration:
            tol = max(
                self.cfg.head_over_hips_allow,
                self.calibration.head_over_hips_x
                + 2.0 * self.calibration.head_over_hips_x_std
                + 0.01,
            )
        else:
            tol = self.cfg.head_over_hips_tol_x

        stab_score = 1.0 - max(0.0, min(1.0, sig.head_over_hips_x / tol))

        # Final weighted score
        total = (
            0.25 * chamber_score
            + 0.25 * sp_score
            + 0.20 * ext_score
            + 0.20 * rot_score
            + 0.10 * stab_score
        )
        score = round(total * 100.0, 1)

        # Feedback
        feedback: List[str] = []
        if not self.chamber_ok:
            feedback.append("Chamber wasn't clear - lift and fold the knee first.")
        if rot_score < 0.5:
            feedback.append("Add more hip pivot through the kick.")
        if ext_score < 0.6:
            feedback.append("Whip the leg out more at peak extension.")
        if stab_score < 0.6:
            feedback.append("Balance drift - stay tall and centered.")
        if dur > 0.9:
            feedback.append("Kick felt slow - aim for a snappier recoil.")

        meta: Dict = {
            "duration_s": dur,
            "peak_ankle_speed": self.peak_ankle_speed,
            "peak_knee_ang": self.peak_knee_ang,
            "rot_delta": self.peak_rot_delta,
        }

        return score, feedback[:3], meta
