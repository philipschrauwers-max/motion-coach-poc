# app.py
import time
import cv2
import mediapipe as mp

from core.savate.config import SavateConfig
from core.savate.signals import compute_signals
from core.savate.punch_detector import PunchDetector
from core.savate.kick_detector import KickDetector

from core.savate.calibration import CalibrationManager
from core.savate.session_logger import SessionLogger


WINDOW_NAME = "Savate Motion Coach POC"


def _ui_scale(frame_w: int) -> float:
    # Scales UI with resolution (tuned for 720p–4K)
    return max(0.8, min(1.8, frame_w / 1200.0))


def put_text_rel(img, text: str, x_frac: float, y_frac: float, scale_mult: float = 1.0):
    """
    Draw text at a relative position in the frame.
    x_frac/y_frac are in [0..1], relative to frame width/height.
    """
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
    cfg = SavateConfig()

    punch = PunchDetector(cfg=cfg, mode="auto")
    kick = KickDetector(cfg=cfg, mode="auto")

    # App mode: auto / jab / cross / fouette
    mode = "auto"

    last_rep = None
    rep_counts = {"jab": 0, "cross": 0, "fouette": 0}

    # Calibration + session logging
    cal_mgr = CalibrationManager(duration_s=2.0)
    logger = SessionLogger()
    cal = None

    # Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; safe if it falls back
    if not cap.isOpened():
        # fallback
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

    # Set initial detector modes ONCE (don’t reset every frame)
    punch.set_mode("auto")
    kick.set_mode("auto")

    # Make window resizable and start at a sensible size
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 850)

    print("Hotkeys: 0=auto 1=jab 2=cross 3=fouette  k=calibrate  e=export  q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for training
        frame = cv2.flip(frame, 1)

        # FPS estimate
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)
        rgb.flags.writeable = True

        # Panels: left HUD + right feedback
        draw_panel(frame, 0.01, 0.01, 0.42, 0.26, alpha=0.35)  # top-left
        draw_panel(frame, 0.58, 0.01, 0.99, 0.36, alpha=0.35)  # top-right feedback

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

            # Update detectors (based on current mode)
            rep = None
            if mode == "jab":
                rep = punch.update(sig)
            elif mode == "cross":
                rep = punch.update(sig)
            elif mode == "fouette":
                rep = kick.update(sig)
            else:
                rep = punch.update(sig) or kick.update(sig)

            if rep:
                last_rep = rep
                rep_counts[rep.kind] = rep_counts.get(rep.kind, 0) + 1
                logger.add_rep(rep)

            # ---- Left HUD (relative layout) ----
            put_text_rel(frame, f"Mode: {mode.upper()}   FPS: {fps:.1f}", 0.02, 0.05, 1.1)
            put_text_rel(
                frame,
                f"Jab:{rep_counts['jab']}  Cross:{rep_counts['cross']}  Fouette:{rep_counts['fouette']}",
                0.02,
                0.095,
            )

            if cal_mgr.collecting:
                put_text_rel(frame, f"Calibrating... {int(cal_mgr.progress() * 100)}%", 0.02, 0.14)
            elif cal:
                put_text_rel(frame, "Calibrated: YES", 0.02, 0.14)
            else:
                put_text_rel(frame, "Calibrated: NO (press K)", 0.02, 0.14)

            # Live debug signals (optional but helpful)
            put_text_rel(frame, f"Guard L:{sig.guard_left:+.3f}  R:{sig.guard_right:+.3f}", 0.02, 0.185)
            put_text_rel(frame, f"WristSpd L:{sig.l_wrist_speed:.3f}  R:{sig.r_wrist_speed:.3f}", 0.02, 0.225)
            put_text_rel(frame, f"AnkleSpd L:{sig.l_ankle_speed:.3f}  R:{sig.r_ankle_speed:.3f}", 0.02, 0.265)

            # ---- Right Feedback panel ----
            if last_rep:
                put_text_rel(
                    frame,
                    f"Last: {last_rep.kind.upper()} ({last_rep.side})  Score: {last_rep.score}",
                    0.60,
                    0.05,
                    1.1,
                )
                y = 0.10
                for line in last_rep.feedback:
                    put_text_rel(frame, f"- {line}", 0.60, y)
                    y += 0.045

        else:
            put_text_rel(frame, f"Mode: {mode.upper()}   FPS: {fps:.1f}", 0.02, 0.05, 1.1)
            put_text_rel(frame, "No pose detected", 0.02, 0.10)
            if cal_mgr.collecting:
                put_text_rel(frame, f"Calibrating... {int(cal_mgr.progress() * 100)}%", 0.02, 0.145)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Modes (set detector modes ONLY on keypress)
        elif key == ord("0"):
            mode = "auto"
            punch.set_mode("auto")
            kick.set_mode("auto")

        elif key == ord("1"):
            mode = "jab"
            punch.set_mode("jab")      # orthodox: left jab
            kick.set_mode("auto")

        elif key == ord("2"):
            mode = "cross"
            punch.set_mode("cross")    # orthodox: right cross
            kick.set_mode("auto")

        elif key == ord("3"):
            mode = "fouette"
            kick.set_mode("fouette")
            punch.set_mode("auto")

        # Calibration
        elif key == ord("k"):
            cal_mgr.start()
            print("Calibration started: stand in guard for ~2 seconds.")

        # Export session
        elif key == ord("e"):
            j = logger.export_json()
            c = logger.export_csv()
            print(f"Exported:\n  {j}\n  {c}")

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()