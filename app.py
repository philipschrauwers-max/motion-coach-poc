# app.py
import time
import os
import cv2
import mediapipe as mp

from core.savate.config import SavateConfig
from core.savate.signals import compute_signals
from core.savate.punch_detector import PunchDetector
from core.savate.kick_detector import KickDetector

from core.savate.calibration import CalibrationManager
from core.savate.session_logger import SessionLogger
from core.savate.shadow_mode import ShadowModeTracker


WINDOW_NAME = "Savate Motion Coach POC"


def _ui_scale(frame_w: int) -> float:
    # Scales UI with resolution (tuned for 720p–4K)
    return max(0.8, min(1.8, frame_w / 1200.0))


def put_text_rel(img, text: str, x_frac: float, y_frac: float, scale_mult: float = 1.0):
    """Draw text at a relative position in the frame."""
    h, w = img.shape[:2]
    ui = _ui_scale(w) * scale_mult
    font_scale = 0.55 * ui
    thickness = int(max(1, round(2 * ui)))
    x = int(x_frac * w)
    y = int(y_frac * h)

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(img, x0_frac: float, y0_frac: float, x1_frac: float, y1_frac: float, alpha: float = 0.35):
    """Semi-transparent panel to improve text readability."""
    h, w = img.shape[:2]
    x0, y0 = int(x0_frac * w), int(y0_frac * h)
    x1, y1 = int(x1_frac * w), int(y1_frac * h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def main():
    print("CWD:", os.getcwd())

    cfg = SavateConfig()

    punch = PunchDetector(cfg=cfg, mode="auto")
    kick = KickDetector(cfg=cfg, mode="auto")

    # Modes: auto / jab / cross / fouette / shadow
    mode = "auto"

    # Gate: don’t track reps until calibrated + mode selected
    selected_mode = None        # None until user presses 1/2/3/4 (or 0 if you allow auto)
    tracking_enabled = False    # True only after calibration + mode selection

    last_rep = None
    rep_counts = {"jab": 0, "cross": 0, "fouette": 0}

    # Calibration + session logging
    cal_mgr = CalibrationManager(duration_s=2.0)
    logger = SessionLogger()
    cal = None

    # Shadow mode
    shadow = ShadowModeTracker(duration_s=60.0, calibration=None)
    shadow_summary = None
    shadow_summary_until = 0.0

    # Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different camera index (0,1,2...).")

    # Request a decent capture resolution (camera may ignore)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # MediaPipe Pose
    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils

    prev_sig = None
    last_time = time.time()

    # Initialize detectors once
    punch.set_mode("auto")
    kick.set_mode("auto")

    # Make window resizable and start at a sensible size
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 850)

    print(
        "Hotkeys:\n"
        "  k=calibrate (required)\n"
        "  1=jab  2=cross  3=fouette  4=shadow  (pick a mode to enable tracking)\n"
        "  s=start shadow round (only in shadow mode)\n"
        "  +/- change shadow duration\n"
        "  e=export session\n"
        "  q=quit"
    )

    def set_mode(new_mode: str):
        nonlocal mode, selected_mode, tracking_enabled
        mode = new_mode
        selected_mode = new_mode

        # Update detectors ONLY here (never inside the frame loop)
        if new_mode == "jab":
            punch.set_mode("jab")
            kick.set_mode("auto")
        elif new_mode == "cross":
            punch.set_mode("cross")
            kick.set_mode("auto")
        elif new_mode == "fouette":
            kick.set_mode("fouette")
            punch.set_mode("auto")
        elif new_mode == "shadow":
            punch.set_mode("auto")
            kick.set_mode("auto")
        else:
            punch.set_mode("auto")
            kick.set_mode("auto")

        # Tracking turns on only if calibrated
        tracking_enabled = (cal is not None)

        # Reset last rep display when switching modes
        # (optional, but makes the UI feel less “stale”)
        # nonlocal last_rep
        # last_rep = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)
        rgb.flags.writeable = True

        # Panels: top-left HUD + top-right feedback
        draw_panel(frame, 0.01, 0.01, 0.52, 0.34, alpha=0.35)
        draw_panel(frame, 0.55, 0.01, 0.99, 0.40, alpha=0.35)

        if res.pose_landmarks:
            drawing.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            lms = [(lm.x, lm.y, lm.z, lm.visibility) for lm in res.pose_landmarks.landmark]

            sig = compute_signals(lms, t=now, prev=prev_sig)
            prev_sig = sig

            # Calibration update (if active)
            if cal_mgr.collecting:
                maybe = cal_mgr.update(sig)
                if maybe:
                    cal = maybe
                    logger.set_calibration(cal)
                    punch.set_calibration(cal)
                    kick.set_calibration(cal)
                    shadow.calibration = cal

                    # Only enable tracking if a mode was already chosen
                    tracking_enabled = (selected_mode is not None)

            # Shadow frame tracking only when round is active and tracking is enabled
            if mode == "shadow" and tracking_enabled and shadow.active:
                finished = shadow.update_frame(sig)
                if finished:
                    shadow_summary = finished
                    shadow_summary_until = time.time() + 8.0

            # ---- Rep detection (GATED) ----
            rep = None
            if tracking_enabled:
                if mode == "jab":
                    rep = punch.update(sig)
                elif mode == "cross":
                    rep = punch.update(sig)
                elif mode == "fouette":
                    rep = kick.update(sig)
                elif mode == "shadow":
                    # Only detect reps during an active shadow round
                    if shadow.active:
                        rep = punch.update(sig) or kick.update(sig)
                else:
                    # No "auto" tracking unless you explicitly want it
                    rep = None

            if rep:
                last_rep = rep
                rep_counts[rep.kind] = rep_counts.get(rep.kind, 0) + 1
                logger.add_rep(rep)
                if mode == "shadow" and shadow.active:
                    shadow.add_rep(rep)

            # ---- Left HUD ----
            put_text_rel(frame, f"Mode: {mode.upper()}   FPS: {fps:.1f}", 0.02, 0.05, 1.1)
            put_text_rel(
                frame,
                f"Jab:{rep_counts['jab']}  Cross:{rep_counts['cross']}  Fouette:{rep_counts['fouette']}",
                0.02,
                0.095,
            )

            # Gate instructions / status
            if cal_mgr.collecting:
                put_text_rel(frame, f"Calibrating... {int(cal_mgr.progress() * 100)}%", 0.02, 0.145, 1.05)
            elif cal is None:
                put_text_rel(frame, "Step 1: Press K to calibrate (stand in guard)", 0.02, 0.145, 1.05)
            elif selected_mode is None:
                put_text_rel(frame, "Step 2: Pick a mode: 1=Jab 2=Cross 3=Fouette 4=Shadow", 0.02, 0.145, 1.05)
            elif mode == "shadow" and not shadow.active:
                put_text_rel(frame, "Step 3: Press S to start shadow round", 0.02, 0.145, 1.05)
            else:
                put_text_rel(frame, "TRACKING: ON", 0.02, 0.145, 1.05)

            # Shadow HUD
            if mode == "shadow":
                if shadow.active:
                    put_text_rel(frame, f"SHADOW: {shadow.time_left():.0f}s left", 0.02, 0.195, 1.1)
                else:
                    put_text_rel(frame, f"SHADOW READY: {shadow.duration_s:.0f}s (+/- to adjust)", 0.02, 0.195, 1.1)

            # Debug signals (optional)
            put_text_rel(frame, f"Guard L:{sig.guard_left:+.3f}  R:{sig.guard_right:+.3f}", 0.02, 0.245)
            put_text_rel(frame, f"WristSpd L:{sig.l_wrist_speed:.3f}  R:{sig.r_wrist_speed:.3f}", 0.02, 0.285)

            # ---- Right Feedback panel (only when tracking is enabled) ----
            if tracking_enabled and last_rep:
                put_text_rel(
                    frame,
                    f"Last: {last_rep.kind.upper()} ({last_rep.side})  Score: {last_rep.score}",
                    0.57,
                    0.05,
                    1.1,
                )
                y = 0.11
                for line in last_rep.feedback:
                    put_text_rel(frame, f"- {line}", 0.57, y)
                    y += 0.05
            else:
                put_text_rel(frame, "Feedback will appear after calibration + mode selection.", 0.57, 0.06)

            # Shadow summary overlay
            if shadow_summary and time.time() < shadow_summary_until:
                draw_panel(frame, 0.12, 0.42, 0.88, 0.88, alpha=0.45)
                put_text_rel(frame, f"ROUND SUMMARY ({shadow_summary.duration_s:.0f}s)", 0.15, 0.48, 1.2)
                put_text_rel(frame, f"Reps: {shadow_summary.reps_total}  {shadow_summary.reps_by_kind}", 0.15, 0.54)
                put_text_rel(frame, f"Avg: {shadow_summary.avg_score_by_kind}", 0.15, 0.59)
                put_text_rel(frame, f"Guard down: {shadow_summary.guard_down_pct}%", 0.15, 0.64)
                put_text_rel(frame, f"Balance drift: {shadow_summary.balance_drift_pct}%", 0.15, 0.69)

                yy = 0.76
                for note in shadow_summary.notes[:4]:
                    put_text_rel(frame, f"- {note}", 0.15, yy)
                    yy += 0.055

        else:
            put_text_rel(frame, f"Mode: {mode.upper()}   FPS: {fps:.1f}", 0.02, 0.05, 1.1)
            put_text_rel(frame, "No pose detected", 0.02, 0.10)
            if cal_mgr.collecting:
                put_text_rel(frame, f"Calibrating... {int(cal_mgr.progress() * 100)}%", 0.02, 0.15)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Calibration (always allowed)
        elif key == ord("k"):
            cal_mgr.start()
            print("Calibration started: stand in guard for ~2 seconds.")

        # Mode selection (does NOT enable tracking unless calibrated)
        elif key == ord("1"):
            set_mode("jab")
        elif key == ord("2"):
            set_mode("cross")
        elif key == ord("3"):
            set_mode("fouette")
        elif key == ord("4"):
            set_mode("shadow")
            print("Shadow mode selected. Press S to start a timed round.")

        # Start shadow round (only if calibrated + shadow mode)
        elif key == ord("s"):
            if mode == "shadow" and tracking_enabled:
                shadow.start()
                shadow_summary = None
                print(f"Shadow round started: {shadow.duration_s:.0f}s")
            elif mode == "shadow" and not tracking_enabled:
                print("Shadow round requires calibration first (press K).")

        # Adjust shadow duration
        elif key == ord("+") or key == ord("="):
            shadow.duration_s = min(300.0, shadow.duration_s + 15.0)
            print(f"Shadow duration: {shadow.duration_s:.0f}s")

        elif key == ord("-") or key == ord("_"):
            shadow.duration_s = max(15.0, shadow.duration_s - 15.0)
            print(f"Shadow duration: {shadow.duration_s:.0f}s")

        # Export session (always allowed, but may be empty)
        elif key == ord("e"):
            j = logger.export_json()
            c = logger.export_csv()
            print(f"Exported JSON: {j.resolve()}")
            print(f"Exported CSV : {c.resolve()}")

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
