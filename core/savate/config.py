from dataclasses import dataclass

@dataclass
class SavateConfig:
    # Which side is lead in orthodox? (left lead)
    lead_hand: str = "left"
    rear_hand: str = "right"

    # Visibility gate
    min_vis: float = 0.5

    # Punch detection thresholds (tune as you go)
    wrist_speed_start: float = 0.015   # normalized units per second-ish (depends on dt)
    wrist_speed_end: float = 0.010
    elbow_extended_deg: float = 155.0
    guard_tol: float = 0.03            # wrists should be near nose height (normalized y)

    # Kick detection thresholds
    ankle_speed_start: float = 0.020
    ankle_speed_end: float = 0.012
    knee_chamber_y_tol: float = 0.02   # knee above hip by this much (y smaller is higher)
    knee_extended_deg: float = 160.0

    # Stability/balance
    head_over_hips_tol_x: float = 0.06

    # Scoring weights (keep simple)
    w_guard: float = 0.25
    w_speed: float = 0.25
    w_extension: float = 0.30
    w_stability: float = 0.20

    # Calibration-based tolerances (multipliers)
    guard_drop_allow: float = 0.03          # how far below baseline guard before penalizing
    stance_return_allow: float = 0.12       # fraction of baseline stance width
    head_over_hips_allow: float = 0.06      # absolute fallback, overridden by calibration when available