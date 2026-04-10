#!/usr/local/bin/python3
"""
Plank Posture Coach – main entry point.

Controls (OpenCV window must be focused)
  q / ESC  Quit
  d        Toggle skeleton debug overlay
  p        Pause / resume
  1        Sensitivity: low
  2        Sensitivity: medium  (default)
  3        Sensitivity: high
"""

import sys
import time

import cv2
import numpy as np

from config import CAMERA_INDEX, GOOD_FORM_INTERVAL, SENSITIVITY_PRESETS
from pose_detector import PoseDetector
from posture_analyzer import PlankAnalyzer
from session_logger import SessionLogger
from voice_coach import VoiceCoach

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_C = {
    "green":  (50,  210,  50),
    "yellow": (30,  210, 230),
    "red":    (40,   50, 220),
    "blue":   (220, 120,  30),
    "white":  (255, 255, 255),
    "black":  (0,     0,   0),
    "gray":   (120, 120, 120),
    "panel":  (20,   20,  20),
}

_ISSUE_LABEL = {
    "hip_high":    "Hips too high",
    "hip_low":     "Hips sagging",
    "head_up":     "Head up",
    "head_down":   "Head down",
    "not_in_plank":"Not in plank",
    "good_form":   "Good form",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_px(point: tuple, w: int, h: int) -> tuple[int, int]:
    return (int(point[0] * w), int(point[1] * h))


def _severity_color(severity: float) -> tuple:
    if severity < 1.5:
        return _C["yellow"]
    return _C["red"]


def _draw_panel(frame, x: int, y: int, pw: int, ph: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), _C["panel"], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_text(
    frame,
    text: str,
    pos: tuple[int, int],
    color: tuple,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, _C["black"], thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Visual overlay
# ---------------------------------------------------------------------------

def draw_overlay(
    frame,
    result: dict,
    issue_counters: dict[str, int],
    preset_frames: int,
    logger: SessionLogger,
    sensitivity: str,
    fps: float,
    paused: bool,
    debug: bool,
) -> None:
    h, w = frame.shape[:2]
    metrics = result["metrics"]
    in_plank = result["in_plank"]
    issues = result["issues"]

    # ---- Ideal plank line & hip indicator ----------------------------------
    if in_plank:
        shoulder_px = _to_px(metrics["shoulder"], w, h)
        ankle_px    = _to_px(metrics["ankle"],    w, h)
        hip_px      = _to_px(metrics["hip"],      w, h)

        # Reference line (shoulder → ankle)
        cv2.line(frame, shoulder_px, ankle_px, _C["blue"], 2, cv2.LINE_AA)

        # Hip dot coloured by deviation
        hip_dev = metrics.get("hip_deviation") or 0.0
        thr = SENSITIVITY_PRESETS[sensitivity]["hip_threshold"]
        abs_dev = abs(hip_dev)
        if abs_dev < thr:
            hc = _C["green"]
        elif abs_dev < thr * 2:
            hc = _C["yellow"]
        else:
            hc = _C["red"]
        cv2.circle(frame, hip_px, 12, hc, -1, cv2.LINE_AA)
        cv2.circle(frame, hip_px, 12, _C["white"], 1, cv2.LINE_AA)

        # Arrow showing direction of deviation
        if abs(hip_dev) > thr * 0.5:
            arrow_dy = int(np.sign(hip_dev) * 25)  # positive = down = sagging
            cv2.arrowedLine(
                frame, hip_px,
                (hip_px[0], hip_px[1] + arrow_dy),
                _C["red"] if abs_dev > thr else _C["yellow"],
                2, cv2.LINE_AA, tipLength=0.4,
            )

        if debug:
            # Show nose position
            nose_px = _to_px(metrics["nose"], w, h)
            cv2.circle(frame, nose_px, 6, _C["yellow"], -1, cv2.LINE_AA)

    # ---- Top status bar ----------------------------------------------------
    bar_h = 46
    _draw_panel(frame, 0, 0, w, bar_h)

    dur = logger.duration
    mins, secs = int(dur // 60), int(dur % 60)
    status = (
        f"PLANK COACH  |  {mins:02d}:{secs:02d}"
        f"  |  Corrections: {logger.total_corrections}"
        f"  |  {sensitivity.upper()}"
        f"  |  {fps:.0f} fps"
    )
    if paused:
        status += "  |  [PAUSED]"
    _put_text(frame, status, (10, 30), _C["white"], scale=0.55)

    # ---- Good-form banner (centre of frame) --------------------------------
    good_form_active = (
        in_plank
        and len(issues) == 1
        and issues[0]["key"] == "good_form"
        and issue_counters.get("good_form", 0) >= preset_frames
    )
    if good_form_active:
        banner = "PERFECT FORM!"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        bx = (w - bw) // 2
        by = h // 2 + bh // 2
        # Semi-transparent dark pill behind text
        pad = 18
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - pad, by - bh - pad), (bx + bw + pad, by + pad),
                      (20, 20, 20), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        # Outlined green text
        cv2.putText(frame, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, _C["black"], 6, cv2.LINE_AA)
        cv2.putText(frame, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, _C["green"], 3, cv2.LINE_AA)

    # ---- Issue panel (bottom) ----------------------------------------------
    row_h = 32
    panel_h = len(issues) * row_h + 12
    panel_y = h - panel_h
    _draw_panel(frame, 0, panel_y, w, panel_h, alpha=0.6)

    bar_max_w = 180

    for i, issue in enumerate(issues):
        key = issue["key"]
        row_y = panel_y + 8 + i * row_h

        # Progress bar (how close to triggering the alert)
        count = issue_counters.get(key, 0)
        pct = min(count / max(preset_frames, 1), 1.0)
        fill_w = int(bar_max_w * pct)

        if key == "good_form":
            bar_color = _C["green"]
            label_color = _C["green"]
            prefix = "+"
        elif key == "not_in_plank":
            bar_color = _C["blue"]
            label_color = _C["blue"]
            prefix = "o"
        else:
            bar_color = _severity_color(issue["severity"])
            label_color = bar_color
            prefix = "!"

        cv2.rectangle(frame, (10, row_y + 4), (10 + fill_w, row_y + 20), bar_color, -1)
        cv2.rectangle(frame, (10, row_y + 4), (10 + bar_max_w, row_y + 20), _C["gray"], 1)

        label = f"{prefix}  {_ISSUE_LABEL.get(key, key.replace('_', ' ').title())}"
        _put_text(frame, label, (bar_max_w + 20, row_y + 20), label_color, scale=0.55)

    # ---- Debug metrics (bottom-right) --------------------------------------
    if debug and in_plank:
        hip_dev = metrics.get("hip_deviation")
        head_r  = metrics.get("head_rise")
        angle   = metrics.get("body_angle")
        lines = [
            f"Body angle : {angle:.1f} deg",
            f"Hip dev    : {hip_dev:+.4f}" if hip_dev is not None else "Hip dev    : --",
            f"Head rise  : {head_r:+.4f}" if head_r  is not None else "Head rise  : --",
        ]
        for j, line in enumerate(lines):
            _put_text(frame, line, (w - 260, panel_y - 20 - j * 20), _C["gray"], scale=0.42)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX in config.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    detector = PoseDetector()
    analyzer = PlankAnalyzer()
    coach    = VoiceCoach()
    logger   = SessionLogger()

    sensitivity = "medium"
    issue_counters: dict[str, int] = {}
    debug  = False
    paused = False

    fps   = 30.0
    prev  = time.monotonic()

    print("=" * 60)
    print("  Plank Posture Coach")
    print("=" * 60)
    print("  Controls (focus the preview window first):")
    print("    q / ESC  → quit")
    print("    d        → toggle skeleton / debug overlay")
    print("    p        → pause / resume")
    print("    1        → sensitivity: low")
    print("    2        → sensitivity: medium")
    print("    3        → sensitivity: high")
    print()
    print("  TIP: Place the camera at torso height to your side")
    print("       for the most accurate hip-alignment detection.")
    print("=" * 60)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame – retrying…")
                continue

            frame = cv2.flip(frame, 1)   # mirror for natural interaction
            h, w  = frame.shape[:2]

            # FPS (exponential moving average)
            now  = time.monotonic()
            fps  = 0.9 * fps + 0.1 / max(now - prev, 1e-4)
            prev = now

            if paused:
                _put_text(frame, "PAUSED – press P to resume",
                          (w // 2 - 180, h // 2), _C["yellow"], scale=0.8, thickness=2)
                cv2.imshow("Plank Posture Coach", frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == ord('p'):
                    paused = False
                continue

            # ----- Pose detection -------------------------------------------
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.pose_landmarks:
                lm_list = results.pose_landmarks[0]   # single-person detection

                if debug:
                    detector.draw_landmarks(frame, lm_list)

                preset = SENSITIVITY_PRESETS[sensitivity]
                result = analyzer.analyze(detector, lm_list)

                # Update per-issue consecutive-frame counters
                active = {iss["key"] for iss in result["issues"]}
                for key in list(issue_counters):
                    if key not in active:
                        issue_counters[key] = 0
                for iss in result["issues"]:
                    key = iss["key"]
                    issue_counters[key] = issue_counters.get(key, 0) + 1

                # Trigger voice alerts once threshold is met
                for iss in result["issues"]:
                    key   = iss["key"]
                    count = issue_counters.get(key, 0)
                    if count >= preset["consecutive_frames"]:
                        cd = GOOD_FORM_INTERVAL if key == "good_form" else preset["cooldown"]
                        fired = coach.alert(key, iss["message"], cooldown=cd)
                        if fired:
                            if key not in ("good_form", "not_in_plank"):
                                logger.record_correction(key)
                            elif key == "good_form":
                                logger.record_compliment()

                draw_overlay(
                    frame, result, issue_counters,
                    preset["consecutive_frames"],
                    logger, sensitivity, fps, paused, debug,
                )

            else:
                _put_text(
                    frame,
                    "No pose detected – step in front of the camera",
                    (20, h // 2),
                    _C["white"], scale=0.7, thickness=2,
                )

            cv2.imshow("Plank Posture Coach", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('d'):
                debug = not debug
                print(f"Debug overlay: {'ON' if debug else 'OFF'}")
            elif k == ord('p'):
                paused = True
                print("Paused.")
            elif k == ord('1'):
                sensitivity = "low"
                print(f"Sensitivity → {sensitivity}")
            elif k == ord('2'):
                sensitivity = "medium"
                print(f"Sensitivity → {sensitivity}")
            elif k == ord('3'):
                sensitivity = "high"
                print(f"Sensitivity → {sensitivity}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        coach.stop()
        logger.save(sensitivity)


if __name__ == "__main__":
    main()
