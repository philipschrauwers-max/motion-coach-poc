# app.py  (Punch-only Shadow Mode + Auto-threshold Calibration)

import time
import cv2
import mediapipe as mp

from core.savate.config import SavateConfig
from core.savate.signals import compute_signals
from core.savate.punch_detector import PunchDetector

from core.savate.calibration import CalibrationManager
from core.savate.session_logger import SessionLogger
from core.savate.shadow_mode import ShadowModeTracker

WINDOW_NAME = "Savate Motion Coach POC"


def _ui_scale(frame_w: int) -> float:
    return max(0.8, min(1.8, frame_w / 1200.0))


def put_text_rel(img, text: str, x_frac: float, y_frac: float, scale_mult: float = 1.0):
    h, w = img.shape[:2]
    ui = _ui_scale(w) * scale_mult
    font_scale = 0.55 * ui
    thickness = int(max(1, round(2 * ui)))
    cv2.putText(
        img,
        text,
        (int(x_frac * w), int(y_frac * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(img, x0f, y0f, x1f, y1f, alpha=0.35):
    h, w = img.shape[:2]
    x0, y0 = int(x0f * w), int(y0f * h)
    x1, y1 = int(x1f * w), int(y1f * h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def main():
    cfg = SavateConfig()
    punch = PunchDetector(cfg=cfg, mode="auto")

    # Auto-threshold calibration (guard + 5 test punches)
    cal_mgr = CalibrationManager(duration_guard_s=2.0, max_test_s=6.0, target_peaks=5)

    logger = SessionLogger()
    cal = None

    shadow = ShadowModeTracker(duration_s=60.0, calibration=None)
    shadow_summary = None
    shadow_summary_until = 0.0

    tracking_ready = False  # becomes True after calibration

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # You can drop resolution if FPS is low
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 850)

    prev_sig = None
    last_time = time.time()
    last_rep = None
    rep_counts = {}

    print("Hotkeys: k=calibrate  s=start round  +/- duration  e=export  q=quit")
    print("Calibration flow: hold guard 2s → throw 5 straight punches (jab/cross) at ~60%")

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

        draw_panel(frame, 0.01, 0.01, 0.52, 0.34, alpha=0.35)
        draw_panel(frame, 0.55, 0.01, 0.99, 0.40, alpha=0.35)

        if res.pose_landmarks:
            drawing.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            sig = compute_signals(res.pose_landmarks.landmark, t=now, prev=prev_sig)
            prev_sig = sig
            pdbg = punch.get_debug()

            # Calibration (two-phase)
            if cal_mgr.collecting:
                maybe = cal_mgr.update(sig)
                if maybe:
                    cal = maybe
                    logger.set_calibration(cal)
                    punch.set_calibration(cal)
                    shadow.calibration = cal
                    tracking_ready = True

                    print(
                        f"Calibration complete. "
                        f"idle_p95={cal.punch_idle_p95:.4f}, peak_med={cal.punch_peak_median:.4f}, "
                        f"start={cal.punch_speed_start:.4f}, end={cal.punch_speed_end:.4f}"
                    )

            # Rep detection ONLY during active round
            rep = None
            if tracking_ready and shadow.active:
                finished = shadow.update_frame(sig)
                if finished:
                    shadow_summary = finished
                    shadow_summary_until = time.time() + 8.0

                rep = punch.update(sig)

                if rep:
                    last_rep = rep
                    rep_counts[rep.kind] = rep_counts.get(rep.kind, 0) + 1
                    logger.add_rep(rep)
                    shadow.add_rep(rep)

            # HUD
            put_text_rel(frame, f"SHADOW MODE (PUNCHES)   FPS: {fps:.1f}", 0.02, 0.05, 1.1)

            if cal_mgr.collecting:
                put_text_rel(frame, cal_mgr.status_text(), 0.02, 0.12, 1.1)
                put_text_rel(frame, f"Progress: {int(cal_mgr.progress() * 100)}%", 0.02, 0.19, 1.0)
            elif not tracking_ready:
                put_text_rel(frame, "Step 1: Press K to calibrate", 0.02, 0.12, 1.1)
                put_text_rel(frame, "Hold guard 2s → throw 5 straight punches at ~60%", 0.02, 0.19, 1.0)
            else:
                if shadow.active:
                    put_text_rel(frame, f"ROUND: {shadow.time_left():.0f}s left", 0.02, 0.12, 1.1)
                else:
                    put_text_rel(frame, f"READY: {shadow.duration_s:.0f}s round (press S)", 0.02, 0.12, 1.1)

                # show thresholds for debugging
                if cal:
                    put_text_rel(
                        frame,
                        f"Auto thresholds: start={cal.punch_speed_start:.3f} end={cal.punch_speed_end:.3f}",
                        0.02,
                        0.19,
                        1.0,
                    )

            put_text_rel(frame, f"Counts: {rep_counts}", 0.02, 0.26)

            # Extra debug (optional): speeds/angles
            put_text_rel(
                frame,
                f"WS L:{sig.l_wrist_speed:.3f} R:{sig.r_wrist_speed:.3f}  EL:{sig.l_elbow_ang:.0f} ER:{sig.r_elbow_ang:.0f}",
                0.02,
                0.32,
                0.95,
            )


             # --- Punch detector debug ---
            pdbg = punch.get_debug()
            put_text_rel(frame, "PUNCH DEBUG", 0.57, 0.48, 1.0)
            put_text_rel(
                frame,
                f"State:{pdbg.get('state')}  Side:{pdbg.get('side_used')}  Target:{pdbg.get('target_side')}",
                0.57,
                0.53,
            )
            put_text_rel(
                frame,
                f"Speed:{pdbg.get('speed',0):.4f}  start>{pdbg.get('speed_start',0):.4f}  "
                f"end<{pdbg.get('speed_end',0):.4f}",
                0.57,
                0.58,
            )
            put_text_rel(
                frame,
                f"Elbow:{pdbg.get('elbow',0):.1f}  GuardS:{pdbg.get('guard_strike',0):.3f}  "
                f"GuardO:{pdbg.get('guard_other',0):.3f}",
                0.57,
                0.63,
            )

            if last_rep:
                put_text_rel(
                    frame,
                    f"Last: {last_rep.kind.upper()} ({last_rep.side}) score:{last_rep.score}",
                    0.57,
                    0.70,
                )
            

            if tracking_ready and last_rep:
                put_text_rel(
                    frame,
                    f"Last: {last_rep.kind.upper()} ({last_rep.side})  {last_rep.score}",
                    0.57,
                    0.05,
                    1.1,
                )
                y = 0.11
                for line in last_rep.feedback:
                    put_text_rel(frame, f"- {line}", 0.57, y)
                    y += 0.05
            else:
                put_text_rel(frame, "Feedback appears after a round starts.", 0.57, 0.06)

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
            put_text_rel(frame, f"SHADOW MODE (PUNCHES)   FPS: {fps:.1f}", 0.02, 0.05, 1.1)
            put_text_rel(frame, "No pose detected", 0.02, 0.12)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("k"):
            cal_mgr.start()
            tracking_ready = False
            cal = None
            shadow.calibration = None
            punch.set_calibration(None)
            print("Calibration started: hold guard ~2s, then throw 5 straight punches at ~60%.")

        elif key == ord("s"):
            if tracking_ready:
                shadow.start()
                punch.reset()
                shadow_summary = None
                rep_counts = {}
                last_rep = None
                print(f"Shadow round started: {shadow.duration_s:.0f}s")
            else:
                print("Calibrate first (press K).")

        elif key == ord("+") or key == ord("="):
            shadow.duration_s = min(300.0, shadow.duration_s + 15.0)
            print(f"Shadow duration: {shadow.duration_s:.0f}s")

        elif key == ord("-") or key == ord("_"):
            shadow.duration_s = max(15.0, shadow.duration_s - 15.0)
            print(f"Shadow duration: {shadow.duration_s:.0f}s")

        elif key == ord("e"):
            j = logger.export_json("latest.json")
            c = logger.export_csv("latest.csv")
            print(f"Exported JSON: {j.resolve()}")
            print(f"Exported CSV : {c.resolve()}")

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()