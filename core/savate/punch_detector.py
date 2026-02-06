# core/savate/punch_detector.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .signals import FrameSignals
from .config import SavateConfig
from .calibration import Calibration
from .punch_classifier import PunchEvent, classify_punch


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
    mode: str = "auto"
    calibration: Optional[Calibration] = None

    # state machine
    state: str = "IDLE"
    active_side: Optional[str] = None  # "left" or "right"
    start_t: float = 0.0

    # punch window metrics
    peak_speed: float = 0.0
    peak_elbow: float = 0.0
    min_guard_other: float = 999.0
    min_guard_strike: float = 999.0

    start_rot: float = 0.0
    peak_rot_delta: float = 0.0

    # displacement tracking (from punch start)
    start_wrist_x: float = 0.0
    start_wrist_y: float = 0.0

    # store values at PEAK SPEED moment (more robust than last frame)
    peak_speed_wrist_x: float = 0.0
    peak_speed_wrist_y: float = 0.0
    peak_speed_t: float = 0.0

    peak_abs_dx: float = 0.0
    peak_abs_dy: float = 0.0

    last_dx: float = 0.0
    last_dy: float = 0.0

    # Debug info for HUD
    debug: Dict = field(default_factory=dict)

    # Optional: rep cooldown to prevent double-counting
    last_rep_t: float = -999.0

    def set_mode(self, mode: str):
        self.mode = mode
        self.reset()

    def set_calibration(self, cal: Optional[Calibration]):
        self.calibration = cal

    def get_debug(self) -> Dict:
        return dict(self.debug)

    def reset(self):
        self.state = "IDLE"
        self.active_side = None
        self.start_t = 0.0

        self.peak_speed = 0.0
        self.peak_elbow = 0.0
        self.min_guard_other = 999.0
        self.min_guard_strike = 999.0

        self.start_rot = 0.0
        self.peak_rot_delta = 0.0

        self.start_wrist_x = 0.0
        self.start_wrist_y = 0.0

        self.peak_speed_wrist_x = 0.0
        self.peak_speed_wrist_y = 0.0
        self.peak_speed_t = 0.0

        self.peak_abs_dx = 0.0
        self.peak_abs_dy = 0.0

        self.last_dx = 0.0
        self.last_dy = 0.0

        self.debug = {"state": "IDLE"}

    def _side_metrics(self, sig: FrameSignals, side: str):
        """
        Returns:
          speed, elbow_angle, guard_strike, guard_other, wrist_x, wrist_y
        """
        if side == "left":
            return (
                sig.l_wrist_speed,
                sig.l_elbow_ang,
                sig.guard_left,
                sig.guard_right,
                sig.l_wrist_x,
                sig.l_wrist_y,
            )
        else:
            return (
                sig.r_wrist_speed,
                sig.r_elbow_ang,
                sig.guard_right,
                sig.guard_left,
                sig.r_wrist_x,
                sig.r_wrist_y,
            )

    def update(self, sig: FrameSignals) -> Optional[RepResult]:
        # Cooldown to prevent double counting (helps with jitter)
        if (sig.t - self.last_rep_t) < 0.18:
            self.debug = {"state": self.state, "cooldown": True}
            return None

        # Which side is moving more THIS frame (shadow)
        target_side = "left" if sig.l_wrist_speed >= sig.r_wrist_speed else "right"
        side = self.active_side or target_side

        speed, elbow, guard_strike, guard_other, wx, wy = self._side_metrics(sig, side)

        # Dynamic thresholds (auto-calibration)
        speed_start = self.calibration.punch_speed_start if self.calibration else self.cfg.wrist_speed_start
        speed_end = self.calibration.punch_speed_end if self.calibration else self.cfg.wrist_speed_end

        # Always keep debug fresh (HUD uses this)
        self.debug = {
            "state": self.state,
            "target_side": target_side,
            "active_side": self.active_side,
            "side_used": side,
            "speed": float(speed),
            "speed_start": float(speed_start),
            "speed_end": float(speed_end),
            "elbow": float(elbow),
            "guard_strike": float(guard_strike),
            "guard_other": float(guard_other),
            "wx": float(wx),
            "wy": float(wy),
        }

        # --------------------
        # IDLE -> EXTEND
        # --------------------
        if self.state == "IDLE":
            if speed > speed_start:
                self.state = "EXTEND"
                self.active_side = side
                self.start_t = sig.t

                self.start_rot = sig.torso_rotation_proxy
                self.peak_rot_delta = 0.0

                self.peak_speed = speed
                self.peak_elbow = elbow
                self.min_guard_strike = guard_strike
                self.min_guard_other = guard_other

                # displacement baseline
                self.start_wrist_x = wx
                self.start_wrist_y = wy

                # peak speed snapshot starts here
                self.peak_speed_wrist_x = wx
                self.peak_speed_wrist_y = wy
                self.peak_speed_t = sig.t

                self.peak_abs_dx = 0.0
                self.peak_abs_dy = 0.0
                self.last_dx = 0.0
                self.last_dy = 0.0

                self.debug["transition"] = "IDLE->EXTEND"
            return None

        # Track metrics while active
        if speed > self.peak_speed:
            self.peak_speed = speed
            self.peak_speed_wrist_x = wx
            self.peak_speed_wrist_y = wy
            self.peak_speed_t = sig.t

        self.peak_elbow = max(self.peak_elbow, elbow)
        self.min_guard_strike = min(self.min_guard_strike, guard_strike)
        self.min_guard_other = min(self.min_guard_other, guard_other)
        self.peak_rot_delta = max(self.peak_rot_delta, abs(sig.torso_rotation_proxy - self.start_rot))

        dx = wx - self.start_wrist_x
        dy = wy - self.start_wrist_y
        self.last_dx = dx
        self.last_dy = dy
        self.peak_abs_dx = max(self.peak_abs_dx, abs(dx))
        self.peak_abs_dy = max(self.peak_abs_dy, abs(dy))

        self.debug.update({
            "elapsed": float(sig.t - self.start_t),
            "rot_delta": float(self.peak_rot_delta),
            "last_dx": float(self.last_dx),
            "last_dy": float(self.last_dy),
            "peak_speed": float(self.peak_speed),
            "peak_elbow": float(self.peak_elbow),
            "min_guard_strike": float(self.min_guard_strike),
            "min_guard_other": float(self.min_guard_other),
        })

        # Safety timeout (prevents stuck states)
        elapsed = sig.t - self.start_t
        if elapsed > 1.25:
            self.debug["timeout"] = True
            self.reset()
            return None

        # --------------------
        # EXTEND -> RETURN
        # --------------------
        if self.state == "EXTEND":
            # transition if elbow is "extended enough" OR speed drops after peak
            if elbow >= self.cfg.elbow_extended_deg or speed < (speed_start * 0.85):
                self.state = "RETURN"
                self.debug["transition"] = "EXTEND->RETURN"
            return None

        # --------------------
        # RETURN -> DONE
        # --------------------
        if self.state == "RETURN":
            # "back in guard" check
            if self.calibration:
                baseline = self.calibration.guard_left if side == "left" else self.calibration.guard_right
                allowed = self.cfg.guard_drop_allow + 2.0 * max(
                    self.calibration.guard_left_std, self.calibration.guard_right_std
                )
                in_guard = guard_strike >= (baseline - allowed)
            else:
                # fallback
                in_guard = guard_strike >= -self.cfg.guard_tol

            self.debug["in_guard"] = bool(in_guard)

            if speed < speed_end and in_guard:
                dur = sig.t - self.start_t

                # Use displacement at PEAK SPEED moment for classification (more robust)
                peak_dx = self.peak_speed_wrist_x - self.start_wrist_x
                peak_dy = self.peak_speed_wrist_y - self.start_wrist_y

                ev = PunchEvent(
                    side=side,
                    peak_speed=self.peak_speed,
                    peak_elbow=self.peak_elbow,
                    rot_delta=self.peak_rot_delta,
                    wrist_dx=peak_dx,
                    wrist_dy=peak_dy,
                    duration_s=dur,
                )
                kind = classify_punch(ev)

                score, fb, meta = self._score(kind, sig, dur, ev)
                out = RepResult(kind=kind, side=side, score=score, feedback=fb, meta=meta)

                self.last_rep_t = sig.t
                self.debug.update({
                    "rep_kind": kind,
                    "rep_score": float(score),
                    "rep_dur": float(dur),
                    "rep_dx": float(peak_dx),
                    "rep_dy": float(peak_dy),
                    "transition": "RETURN->DONE",
                })

                self.reset()
                return out

            return None

        return None

    def _score(self, kind: str, sig: FrameSignals, dur: float, ev: PunchEvent):
        # Extension scoring differs by punch type:
        # - Jab/Cross: prefer more extension
        # - Hook/Uppercut: prefer bent elbow, but still some “drive”
        if kind in ("jab", "cross"):
            ext_score = max(0.0, min(1.0, (self.peak_elbow - 135.0) / 35.0))  # 135->170
        else:
            elbow = self.peak_elbow
            dist = abs(elbow - 115.0)  # 95-135 is “good”
            ext_score = max(0.0, min(1.0, 1.0 - (dist / 35.0)))

        # Speed score (normalized)
        sp_score = max(0.0, min(1.0, (self.peak_speed - 0.02) / 0.08))

        # Guard score (calibration-aware)
        if self.calibration:
            base_min = min(self.calibration.guard_left, self.calibration.guard_right)
            allowed = self.cfg.guard_drop_allow + 2.0 * max(
                self.calibration.guard_left_std, self.calibration.guard_right_std
            )
            guard_bad = min(self.min_guard_strike, self.min_guard_other)
            guard_score = max(0.0, min(1.0, (guard_bad - (base_min - allowed)) / (allowed + 1e-6)))
        else:
            guard_bad = min(self.min_guard_strike, self.min_guard_other)
            guard_score = max(0.0, min(1.0, (guard_bad + 0.03) / 0.08))

        # Stability score
        if self.calibration:
            tol = max(
                self.cfg.head_over_hips_allow,
                self.calibration.head_over_hips_x + 2.0 * self.calibration.head_over_hips_x_std + 0.01
            )
        else:
            tol = self.cfg.head_over_hips_tol_x
        stab_score = 1.0 - max(0.0, min(1.0, sig.head_over_hips_x / tol))

        # Rotation preference:
        rot = self.peak_rot_delta
        if kind in ("cross", "hook"):
            rot_score = max(0.0, min(1.0, rot / 18.0))
            sp_score = 0.7 * sp_score + 0.3 * rot_score

        total = (
            self.cfg.w_extension * ext_score +
            self.cfg.w_speed * sp_score +
            self.cfg.w_guard * guard_score +
            self.cfg.w_stability * stab_score
        )
        score = round(total * 100.0, 1)

        fb: List[str] = []
        if guard_score < 0.5:
            fb.append("Guard dropped, recover hands between strikes.")
        if stab_score < 0.6:
            fb.append("Balance drift, stay tall and centered.")
        if kind == "hook":
            if ev.rot_delta < 6.0:
                fb.append("Hook: add hip/shoulder rotation through the arc.")
            if self.peak_elbow > 145:
                fb.append("Hook: keep it tighter,don't turn it into a swingy straight.")
        if kind == "uppercut":
            if self.peak_elbow > 145:
                fb.append("Uppercut: keep the elbow bent and drive upward.")
        if dur > 0.60:
            fb.append("A bit slow, snap out and return faster.")

        meta = {
            "duration_s": dur,
            "peak_speed": self.peak_speed,
            "peak_elbow": self.peak_elbow,
            "rot_delta": self.peak_rot_delta,
            "dx": ev.wrist_dx,
            "dy": ev.wrist_dy,
        }
        return score, fb[:3], meta
