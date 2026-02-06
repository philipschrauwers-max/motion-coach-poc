from dataclasses import dataclass
from typing import Literal

PunchKind = Literal["jab", "cross", "hook", "uppercut"]

@dataclass
class PunchEvent:
    side: str                 # "left" or "right"
    peak_speed: float
    peak_elbow: float         # degrees
    rot_delta: float          # proxy units (your existing torso_rotation_proxy delta)
    wrist_dx: float           # normalized (from pose x), signed
    wrist_dy: float           # normalized (from pose y), signed (down is + if using image coords)
    duration_s: float


def classify_punch(ev: PunchEvent) -> PunchKind:
    """
    Simple rule-based classification:
    - Jab/Cross: straighter (more extended elbow)
    - Hook: bent elbow + more lateral (dx) travel + more rotation
    - Uppercut: bent elbow + more vertical (dy) travel (upwards in image is negative dy)
    """
    side = ev.side
    elbow = ev.peak_elbow
    dx = abs(ev.wrist_dx)
    dy = abs(ev.wrist_dy)
    rot = ev.rot_delta

    # Ratios to decide motion direction (avoid division by zero)
    lateral_ratio = dx / (dy + 1e-6)
    vertical_ratio = dy / (dx + 1e-6)

    # Heuristics (tune later)
    is_extended = elbow >= 150.0           # straight-ish punch
    is_bent = elbow <= 135.0              # hook/uppercut range

    # Uppercut is primarily vertical motion.
    # Note: if your wrist_dy sign is inverted, this still works because we use abs()
    is_uppercut_like = is_bent and vertical_ratio > 1.25

    # Hook is primarily lateral motion + rotation.
    is_hook_like = is_bent and lateral_ratio > 1.25 and rot > 6.0

    if is_uppercut_like:
        return "uppercut"
    if is_hook_like:
        return "hook"

    # Default to straight punches based on side (orthodox assumption)
    # left = jab family, right = cross family
    if is_extended:
        return "jab" if side == "left" else "cross"

    # If not clearly extended/bent, fall back by side:
    return "jab" if side == "left" else "cross"