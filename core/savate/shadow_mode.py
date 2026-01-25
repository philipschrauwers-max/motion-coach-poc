from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from .calibration import Calibration
from .punch_detector import RepResult


@dataclass
class ShadowSummary:
    duration_s: float
    reps_total: int
    reps_by_kind: Dict[str, int]
    avg_score_by_kind: Dict[str, float]
    best_rep: Optional[Dict[str, Any]]
    guard_down_pct: float
    balance_drift_pct: float
    notes: List[str]


@dataclass
class ShadowModeTracker:
    duration_s: float = 60.0
    calibration: Optional[Calibration] = None

    active: bool = False
    start_t: float = 0.0

    reps_by_kind: Dict[str, int] = field(default_factory=lambda: {"jab": 0, "cross": 0, "fouette": 0})
    scores_sum_by_kind: Dict[str, float] = field(default_factory=lambda: {"jab": 0.0, "cross": 0.0, "fouette": 0.0})
    scores_cnt_by_kind: Dict[str, int] = field(default_factory=lambda: {"jab": 0, "cross": 0, "fouette": 0})

    best_rep: Optional[Dict[str, Any]] = None

    # frame-level discipline stats
    frames: int = 0
    guard_down_frames: int = 0
    balance_drift_frames: int = 0

    def start(self, duration_s: Optional[float] = None):
        if duration_s is not None:
            self.duration_s = float(duration_s)
        self.active = True
        self.start_t = time.time()
        self.reps_by_kind = {"jab": 0, "cross": 0, "fouette": 0}
        self.scores_sum_by_kind = {"jab": 0.0, "cross": 0.0, "fouette": 0.0}
        self.scores_cnt_by_kind = {"jab": 0, "cross": 0, "fouette": 0}
        self.best_rep = None
        self.frames = 0
        self.guard_down_frames = 0
        self.balance_drift_frames = 0

    def stop(self) -> ShadowSummary:
        self.active = False
        return self._summarize()

    def time_left(self) -> float:
        if not self.active:
            return 0.0
        return max(0.0, self.duration_s - (time.time() - self.start_t))

    def update_frame(self, sig) -> Optional[ShadowSummary]:
        """
        Call every frame while you have pose signals.
        Returns a ShadowSummary when the timer ends, else None.
        """
        if not self.active:
            return None

        self.frames += 1

        # Guard down logic (calibration-aware)
        if self.calibration:
            base_min = min(self.calibration.guard_left, self.calibration.guard_right)
            allowed = 0.03 + 2.0 * max(self.calibration.guard_left_std, self.calibration.guard_right_std)
            # If either hand dips below (baseline - allowed), count as guard-down
            if (sig.guard_left < (base_min - allowed)) or (sig.guard_right < (base_min - allowed)):
                self.guard_down_frames += 1
        else:
            # fallback: if either guard value is quite negative
            if (sig.guard_left < -0.03) or (sig.guard_right < -0.03):
                self.guard_down_frames += 1

        # Balance drift logic
        if self.calibration:
            tol = max(0.06, self.calibration.head_over_hips_x + 2.0 * self.calibration.head_over_hips_x_std + 0.01)
        else:
            tol = 0.06
        if sig.head_over_hips_x > tol:
            self.balance_drift_frames += 1

        # End condition
        if (time.time() - self.start_t) >= self.duration_s:
            self.active = False
            return self._summarize()

        return None

    def add_rep(self, rep: RepResult):
        if not self.active:
            return
        kind = rep.kind
        if kind not in self.reps_by_kind:
            self.reps_by_kind[kind] = 0
            self.scores_sum_by_kind[kind] = 0.0
            self.scores_cnt_by_kind[kind] = 0

        self.reps_by_kind[kind] += 1
        self.scores_sum_by_kind[kind] += float(rep.score)
        self.scores_cnt_by_kind[kind] += 1

        if (self.best_rep is None) or (rep.score > self.best_rep["score"]):
            self.best_rep = {"kind": rep.kind, "side": rep.side, "score": rep.score, "feedback": rep.feedback, "meta": rep.meta}

    def _summarize(self) -> ShadowSummary:
        reps_total = sum(self.reps_by_kind.values())

        avg_score_by_kind = {}
        for k, cnt in self.scores_cnt_by_kind.items():
            avg_score_by_kind[k] = round(self.scores_sum_by_kind[k] / cnt, 1) if cnt > 0 else 0.0

        guard_down_pct = (self.guard_down_frames / max(1, self.frames)) * 100.0
        balance_drift_pct = (self.balance_drift_frames / max(1, self.frames)) * 100.0

        notes = []
        # Simple coaching rules
        if reps_total < max(6, int(self.duration_s / 10)):
            notes.append("Work rate was low, try steady output (even light techniques).")
        if guard_down_pct > 20:
            notes.append("Guard dropped often, keep hands higher between strikes.")
        if balance_drift_pct > 15:
            notes.append("Balance drift showed up, stay taller and re-center over hips.")
        if avg_score_by_kind.get("fouette", 0) and avg_score_by_kind["fouette"] < 65:
            notes.append("Fouetté quality: focus on chamber → whip → quick recoil.")
        if avg_score_by_kind.get("cross", 0) and avg_score_by_kind["cross"] < 65:
            notes.append("Cross: add hip/shoulder rotation while keeping guard disciplined.")

        if not notes:
            notes.append("Solid round — keep the same discipline and gradually increase pace.")

        return ShadowSummary(
            duration_s=self.duration_s,
            reps_total=reps_total,
            reps_by_kind=self.reps_by_kind,
            avg_score_by_kind=avg_score_by_kind,
            best_rep=self.best_rep,
            guard_down_pct=round(guard_down_pct, 1),
            balance_drift_pct=round(balance_drift_pct, 1),
            notes=notes[:6],
        )