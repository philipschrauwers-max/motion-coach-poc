from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .signals import FrameSignals
from .config import SavateConfig
from .calibration import Calibration

@dataclass
class RepResult:
    kind: str
    side: str
    score: float
    feedback: List[str]
    meta: Dict

@dataclass
class PunchDetector:
    cfg: SavateConfig
    mode: str = "auto"  # "jab", "cross", "auto"
    calibration: Optional[Calibration] = None
    state: str = "IDLE"
    active_side: Optional[str] = None  # "left" or "right"
    start_t: float = 0.0
    peak_speed: float = 0.0
    peak_elbow: float = 0.0
    min_guard_other: float = 999.0
    min_guard_strike: float = 999.0
    start_rot: float = 0.0
    peak_rot_delta: float = 0.0

    def set_calibration(self, cal: Optional[Calibration]):
        self.calibration = cal

    def set_mode(self, mode: str):
        self.mode = mode
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.active_side = None
        self.peak_speed = 0.0
        self.peak_elbow = 0.0
        self.min_guard_other = 999.0
        self.min_guard_strike = 999.0
        self.start_rot = 0.0
        self.peak_rot_delta = 0.0

    def _side_metrics(self, sig: FrameSignals, side: str):
        if side == "left":
            return sig.l_wrist_speed, sig.l_elbow_ang, sig.guard_left, sig.guard_right
        else:
            return sig.r_wrist_speed, sig.r_elbow_ang, sig.guard_right, sig.guard_left

    def update(self, sig: FrameSignals) -> Optional[RepResult]:
        # pick which side we’re looking for
        if self.mode == "jab":
            target_side = "left"   # orthodox jab
        elif self.mode == "cross":
            target_side = "right"  # orthodox cross
        else:
            # auto: whichever wrist is moving more
            target_side = "left" if sig.l_wrist_speed >= sig.r_wrist_speed else "right"

        speed, elbow, guard_strike, guard_other = self._side_metrics(sig, target_side)

        # track guard mins while active
        if self.state != "IDLE":
            self.min_guard_strike = min(self.min_guard_strike, guard_strike)
            self.min_guard_other = min(self.min_guard_other, guard_other)
            self.peak_speed = max(self.peak_speed, speed)
            self.peak_elbow = max(self.peak_elbow, elbow)
            self.peak_rot_delta = max(self.peak_rot_delta, abs(sig.torso_rotation_proxy - self.start_rot))

        if self.state == "IDLE":
            if speed > self.cfg.wrist_speed_start:
                self.state = "EXTEND"
                self.active_side = target_side
                self.start_t = sig.t
                self.start_rot = sig.torso_rotation_proxy
                self.peak_speed = speed
                self.peak_elbow = elbow
                self.min_guard_strike = guard_strike
                self.min_guard_other = guard_other
                self.peak_rot_delta = 0.0
            return None

        # If target changed mid-rep (auto mode), stick to active side
        active = self.active_side or target_side
        speed, elbow, guard_strike, guard_other = self._side_metrics(sig, active)

        if self.state == "EXTEND":
            if elbow >= self.cfg.elbow_extended_deg or speed < self.cfg.wrist_speed_start:
                self.state = "RETURN"
            return None

        if self.state == "RETURN":
            # finish when hand returns to guard-ish and slows down
            # guard_strike: nose_y - wrist_y ; we want it to be >= -guard_tol-ish (near nose)
            in_guard = guard_strike >= -self.cfg.guard_tol
            if speed < self.cfg.wrist_speed_end and in_guard:
                dur = sig.t - self.start_t
                kind = "jab" if active == "left" else "cross"
                score, fb, meta = self._score(kind, sig, dur)
                out = RepResult(kind=kind, side=active, score=score, feedback=fb, meta=meta)
                self.reset()
                return out
            return None

        return None

    def _score(self, kind: str, sig: FrameSignals, dur: float):
        # Extension: elbow angle near 160-175 is good (avoid chasing 180)
        ext = self.peak_elbow
        ext_score = max(0.0, min(1.0, (ext - 135.0) / 35.0))  # 135->170 maps to 0->1

        # Speed: peak wrist speed scaled
        sp = self.peak_speed
        sp_score = max(0.0, min(1.0, (sp - 0.02) / 0.08))

        # Guard scoring, calibration-aware
        if self.calibration:
            # compare the *worst* guard dip vs baseline minus allowance
            base_min = min(self.calibration.guard_left, self.calibration.guard_right)
            allowed = self.cfg.guard_drop_allow + 2.0 * max(self.calibration.guard_left_std, self.calibration.guard_right_std)
            guard_bad = min(self.min_guard_strike, self.min_guard_other)
            # guard_bad should stay near baseline; penalize if it drops below (base_min - allowed)
            guard_score = max(0.0, min(1.0, (guard_bad - (base_min - allowed)) / (allowed + 1e-6)))
        else:
            guard_bad = min(self.min_guard_strike, self.min_guard_other)
            guard_score = max(0.0, min(1.0, (guard_bad + 0.03) / 0.08))

        # Stability: head-over-hips x
        if self.calibration:
            tol = max(self.cfg.head_over_hips_allow, self.calibration.head_over_hips_x + 2.0*self.calibration.head_over_hips_x_std + 0.01)
        else:
            tol = self.cfg.head_over_hips_tol_x

        stab_score = 1.0 - max(0.0, min(1.0, sig.head_over_hips_x / tol))

        # Cross prefers more rotation
        rot = self.peak_rot_delta
        if kind == "cross":
            rot_score = max(0.0, min(1.0, rot / 20.0))
            # blend rotation into speed a bit
            sp_score = 0.7 * sp_score + 0.3 * rot_score

        total = (
            self.cfg.w_extension * ext_score +
            self.cfg.w_speed * sp_score +
            self.cfg.w_guard * guard_score +
            self.cfg.w_stability * stab_score
        )
        score = round(total * 100.0, 1)

        fb = []
        if guard_score < 0.5:
            fb.append("Guard dropped during the strike—keep the hands higher.")
        if ext_score < 0.6:
            fb.append("Not fully extending—snap out a bit more (without locking hard).")
        if stab_score < 0.6:
            fb.append("Balance drift—keep your head more centered over the hips.")
        if kind == "cross" and rot < 8.0:
            fb.append("Add more hip/shoulder rotation for the cross.")
        if dur > 0.55:
            fb.append("A bit slow—try a quicker extension and faster return.")

        meta = {"duration_s": dur, "peak_speed": self.peak_speed, "peak_elbow": self.peak_elbow, "rot_delta": self.peak_rot_delta}
        return score, fb[:3], meta
