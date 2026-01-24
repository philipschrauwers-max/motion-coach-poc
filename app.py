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


def put_text(img, text, x, y):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different camera index (0,1,2...).")

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

            # HUD
            put_text(frame, f"Mode: {mode.upper()}   FPS: {fps:.1f}", 10, 25)
            put_text(
                frame,
                f"Jab:{rep_counts['jab']}  Cross:{rep_counts['cross']}  Fouette:{rep_counts['fouette']}",
                10,
                50,
            )

            if cal_mgr.collecting:
                put_text(frame, f"Calibrating... {int(cal_mgr.progress() * 100)}%", 10, 75)
            elif cal:
                put_text(frame, "Calibrated: YES", 10, 75)
            else:
                put_text(frame, "Calibrated: NO (press K)", 10, 75)

            y = 105
            put_text(frame, f"Guard L:{sig.guard_left:+.3f}  R:{sig.guard_right:+.3f}", 10, y)
            y += 22
            put_text(frame, f"WristSpd L:{sig.l_wrist_speed:.3f}  R:{sig.r_wrist_speed:.3f}", 10, y)
            y += 22
            put_text(frame, f"AnkleSpd L:{sig.l_ankle_speed:.3f}  R:{sig.r_ankle_speed:.3f}", 10, y)
            y += 22

            if last_rep:
                y += 10
                put_text(
                    frame,
                    f"Last: {last_rep.kind.upper()} ({last_rep.side})  Score: {last_rep.score}",
                    10,
                    y,
                )
                y += 22
                for line in last_rep.feedback:
                    put_text(frame, f"- {line}", 10, y)
                    y += 20

        else:
            put_text(frame, f"Mode: {mode.upper()}   FPS: {fps:.1f}", 10, 25)
            put_text(frame, "No pose detected", 10, 60)
            if cal_mgr.collecting:
                put_text(frame, f"Calibrating... {int(cal_mgr.progress() * 100)}%", 10, 85)

        cv2.imshow("Savate Motion Coach POC", frame)
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
