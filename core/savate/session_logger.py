import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calibration import Calibration
from .punch_detector import RepResult


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class SessionLogger:
    def __init__(self, base_dir: str = "data/sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = _now_tag()
        self.started_t = time.time()
        self.calibration: Optional[Calibration] = None
        self.reps: List[Dict[str, Any]] = []

    def set_calibration(self, cal: Calibration):
        self.calibration = cal

    def add_rep(self, rep: RepResult):
        entry = {
            "t": time.time(),
            "kind": rep.kind,
            "side": rep.side,
            "score": rep.score,
            "feedback": rep.feedback,
            "meta": rep.meta,
        }
        self.reps.append(entry)

    def export_json(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"session-{self.session_id}.json"
        path = self.base_dir / filename

        payload = {
            "meta": {
                "session_id": self.session_id,
                "started_t": self.started_t,
                "exported_t": time.time(),
            },
            "calibration": (self.calibration.to_dict() if self.calibration else None),
            "reps": self.reps,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def export_csv(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"session-{self.session_id}.csv"
        path = self.base_dir / filename

        # Flatten common meta fields (keep the CSV simple)
        fieldnames = [
            "t", "kind", "side", "score",
            "duration_s",
            "peak_speed", "peak_elbow", "rot_delta",
            "peak_ankle_speed", "peak_knee_ang",
            "feedback",
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.reps:
                meta = r.get("meta", {}) or {}
                row = {
                    "t": r.get("t"),
                    "kind": r.get("kind"),
                    "side": r.get("side"),
                    "score": r.get("score"),
                    "duration_s": meta.get("duration_s"),
                    "peak_speed": meta.get("peak_speed"),
                    "peak_elbow": meta.get("peak_elbow"),
                    "rot_delta": meta.get("rot_delta"),
                    "peak_ankle_speed": meta.get("peak_ankle_speed"),
                    "peak_knee_ang": meta.get("peak_knee_ang"),
                    "feedback": " | ".join(r.get("feedback", [])),
                }
                w.writerow(row)

        return path
