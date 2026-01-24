from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# MediaPipe indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

def xy(lms, idx):
    x, y, _, _ = lms[idx]
    return np.array([x, y], dtype=np.float32)

def vis(lms, idx):
    return float(lms[idx][3])

def angle_deg(a, b, c) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

@dataclass
class FrameSignals:
    t: float
    dt: float
    # key points
    nose: np.ndarray
    l_wrist: np.ndarray
    r_wrist: np.ndarray
    l_elbow_ang: float
    r_elbow_ang: float
    l_ankle: np.ndarray
    r_ankle: np.ndarray
    l_knee_ang: float
    r_knee_ang: float
    # derived
    guard_left: float
    guard_right: float
    head_over_hips_x: float
    torso_rotation_proxy: float
    stance_width: float
    # velocities (normalized)
    l_wrist_speed: float
    r_wrist_speed: float
    l_ankle_speed: float
    r_ankle_speed: float

def compute_signals(
    lms: List[Tuple[float, float, float, float]],
    t: float,
    prev: Optional[FrameSignals],
) -> FrameSignals:
    nose = xy(lms, NOSE)
    lw, rw = xy(lms, L_WRIST), xy(lms, R_WRIST)
    ls, rs = xy(lms, L_SHOULDER), xy(lms, R_SHOULDER)
    le, re = xy(lms, L_ELBOW), xy(lms, R_ELBOW)
    lh, rh = xy(lms, L_HIP), xy(lms, R_HIP)
    lk, rk = xy(lms, L_KNEE), xy(lms, R_KNEE)
    la, ra = xy(lms, L_ANKLE), xy(lms, R_ANKLE)

    l_elbow_ang = angle_deg(ls, le, lw)
    r_elbow_ang = angle_deg(rs, re, rw)
    l_knee_ang = angle_deg(lh, lk, la)
    r_knee_ang = angle_deg(rh, rk, ra)

    # guard: positive means wrist above nose line (good); negative means below (bad)
    guard_left = float(nose[1] - lw[1])
    guard_right = float(nose[1] - rw[1])

    hips_mid = (lh + rh) / 2.0
    head_over_hips_x = float(abs(nose[0] - hips_mid[0]))

    shoulder_vec = rs - ls
    hip_vec = rh - lh
    shoulder_ang = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    hip_ang = float(np.degrees(np.arctan2(hip_vec[1], hip_vec[0])))
    torso_rotation_proxy = shoulder_ang - hip_ang

    hip_width = float(np.linalg.norm(lh - rh) + 1e-9)
    stance_width = float(np.linalg.norm(la - ra) / hip_width)

    if prev is None:
        dt = 1/30.0
        l_wrist_speed = r_wrist_speed = 0.0
        l_ankle_speed = r_ankle_speed = 0.0
    else:
        dt = max(t - prev.t, 1e-3)
        l_wrist_speed = float(np.linalg.norm(lw - prev.l_wrist) / dt)
        r_wrist_speed = float(np.linalg.norm(rw - prev.r_wrist) / dt)
        l_ankle_speed = float(np.linalg.norm(la - prev.l_ankle) / dt)
        r_ankle_speed = float(np.linalg.norm(ra - prev.r_ankle) / dt)

    return FrameSignals(
        t=t, dt=dt,
        nose=nose, l_wrist=lw, r_wrist=rw,
        l_elbow_ang=l_elbow_ang, r_elbow_ang=r_elbow_ang,
        l_ankle=la, r_ankle=ra,
        l_knee_ang=l_knee_ang, r_knee_ang=r_knee_ang,
        guard_left=guard_left, guard_right=guard_right,
        head_over_hips_x=head_over_hips_x,
        torso_rotation_proxy=torso_rotation_proxy,
        stance_width=stance_width,
        l_wrist_speed=l_wrist_speed, r_wrist_speed=r_wrist_speed,
        l_ankle_speed=l_ankle_speed, r_ankle_speed=r_ankle_speed,
    )
