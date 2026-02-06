# core/savate/signals.py

from dataclasses import dataclass
from typing import Optional, List
import math


@dataclass
class FrameSignals:
    t: float

    # Wrist speeds
    l_wrist_speed: float
    r_wrist_speed: float

    # Elbow angles (deg)
    l_elbow_ang: float
    r_elbow_ang: float

    # Guard: shoulder.y - wrist.y (higher = better guard)
    guard_left: float
    guard_right: float

    # Wrist positions (0..1)
    l_wrist_x: float
    l_wrist_y: float
    r_wrist_x: float
    r_wrist_y: float

    # Balance
    head_over_hips_x: float

    # Rotation proxy
    torso_rotation_proxy: float

    # --- Added for calibration / kicks ---
    stance_width: float          # horizontal distance between ankles (normalized)
    l_knee_ang: float            # deg
    r_knee_ang: float            # deg
    l_ankle_speed: float         # normalized / sec
    r_ankle_speed: float         # normalized / sec
    l_ankle_x: float
    l_ankle_y: float
    r_ankle_x: float
    r_ankle_y: float


def _angle(a, b, c) -> float:
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if mag_ba * mag_bc < 1e-12:
        return 0.0

    cosang = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosang))


def compute_signals(landmarks: List, t: float, prev: Optional[FrameSignals]) -> FrameSignals:
    """
    IMPORTANT: pass MediaPipe landmarks directly:
      compute_signals(res.pose_landmarks.landmark, t=now, prev=prev_sig)
    """

    # MediaPipe indices
    head = landmarks[0]          # nose
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_el = landmarks[13]
    r_el = landmarks[14]
    l_wr = landmarks[15]
    r_wr = landmarks[16]

    l_hp = landmarks[23]
    r_hp = landmarks[24]

    l_knee = landmarks[25]
    r_knee = landmarks[26]
    l_ank = landmarks[27]
    r_ank = landmarks[28]

    # Elbow angles
    l_elbow_ang = _angle(l_sh, l_el, l_wr)
    r_elbow_ang = _angle(r_sh, r_el, r_wr)

    # Knee angles
    # (hip-knee-ankle)
    l_knee_ang = _angle(l_hp, l_knee, l_ank)
    r_knee_ang = _angle(r_hp, r_knee, r_ank)

    # Guard
    guard_left = l_sh.y - l_wr.y
    guard_right = r_sh.y - r_wr.y

    # Wrist positions
    l_wrist_x, l_wrist_y = l_wr.x, l_wr.y
    r_wrist_x, r_wrist_y = r_wr.x, r_wr.y

    # Ankle positions
    l_ankle_x, l_ankle_y = l_ank.x, l_ank.y
    r_ankle_x, r_ankle_y = r_ank.x, r_ank.y

    # Wrist speed
    if prev:
        dt = max(1e-6, t - prev.t)
        l_wrist_speed = math.hypot(l_wrist_x - prev.l_wrist_x, l_wrist_y - prev.l_wrist_y) / dt
        r_wrist_speed = math.hypot(r_wrist_x - prev.r_wrist_x, r_wrist_y - prev.r_wrist_y) / dt
        l_ankle_speed = math.hypot(l_ankle_x - prev.l_ankle_x, l_ankle_y - prev.l_ankle_y) / dt
        r_ankle_speed = math.hypot(r_ankle_x - prev.r_ankle_x, r_ankle_y - prev.r_ankle_y) / dt
    else:
        l_wrist_speed = r_wrist_speed = 0.0
        l_ankle_speed = r_ankle_speed = 0.0

    # Balance: head over hips drift (x)
    hips_x = 0.5 * (l_hp.x + r_hp.x)
    head_over_hips_x = abs(head.x - hips_x)

    # Torso rotation proxy (shoulder line in x/z plane)
    dx = r_sh.x - l_sh.x
    dz = r_sh.z - l_sh.z
    torso_rotation_proxy = math.degrees(math.atan2(dz, dx))

    # Stance width (ankle separation in x)
    stance_width = abs(l_ankle_x - r_ankle_x)

    return FrameSignals(
        t=t,

        l_wrist_speed=l_wrist_speed,
        r_wrist_speed=r_wrist_speed,

        l_elbow_ang=l_elbow_ang,
        r_elbow_ang=r_elbow_ang,

        guard_left=guard_left,
        guard_right=guard_right,

        l_wrist_x=l_wrist_x,
        l_wrist_y=l_wrist_y,
        r_wrist_x=r_wrist_x,
        r_wrist_y=r_wrist_y,

        head_over_hips_x=head_over_hips_x,
        torso_rotation_proxy=torso_rotation_proxy,

        stance_width=stance_width,
        l_knee_ang=l_knee_ang,
        r_knee_ang=r_knee_ang,
        l_ankle_speed=l_ankle_speed,
        r_ankle_speed=r_ankle_speed,
        l_ankle_x=l_ankle_x,
        l_ankle_y=l_ankle_y,
        r_ankle_x=r_ankle_x,
        r_ankle_y=r_ankle_y,
    )