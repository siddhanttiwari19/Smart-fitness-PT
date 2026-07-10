#!/usr/bin/env python3
"""
AI Squat Coach — main entry point.

Architecture
------------
Pure OpenCV application.  No Streamlit, no WebRTC, no async event loops.
One thread owns the camera + MediaPipe + rendering (the main loop).
A second daemon thread handles voice output (VoiceCoach).

Controls (focus the OpenCV window first)
  q / ESC   Quit
  p         Pause / resume
  r         Reset session (reps, scores, calibration)
  v         Toggle voice feedback
  d         Toggle debug overlay (raw angles)

Run
  python3 main.py
  or
  bash run.sh
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CONSECUTIVE_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GOOD_REP_VOICE_COOLDOWN,
    INFERENCE_EVERY,
    VOICE_COOLDOWN_SEC,
)
from pose_detector import PoseDetector
from rep_counter import RepCounter, SquatPhase
from session_logger import SessionLogger
from squat_analyzer import SquatAnalyzer
from utils import MovingAverage, clamp, draw_color_for_score
from voice_coach import VoiceCoach


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_C = {
    "green":  (50, 210, 50),
    "yellow": (30, 210, 230),
    "orange": (30, 140, 240),
    "red":    (40, 50, 220),
    "blue":   (220, 120, 30),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "gray":   (120, 120, 120),
    "panel":  (20, 20, 20),
}


_PHASE_COLOR = {
    SquatPhase.STANDING:    _C["green"],
    SquatPhase.DESCENDING:  _C["orange"],
    SquatPhase.BOTTOM:      _C["red"],
    SquatPhase.ASCENDING:   _C["yellow"],
    SquatPhase.CALIBRATING: _C["gray"],
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _draw_panel(frame, x: int, y: int, pw: int, ph: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), _C["panel"], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_text(
    frame, text: str, pos: tuple[int, int], color: tuple,
    scale: float = 0.55, thickness: int = 1,
) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _C["black"], thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame,
    phase: SquatPhase,
    reps: int,
    score: float,
    fps: float,
    issues: list[str],
    issue_counters: dict[str, int],
    feedback_text: str,
    feedback_color,
    voice_on: bool,
    paused: bool,
    debug: bool,
    debug_metrics: dict,
    recent_scores: list[float],
) -> None:
    h, w = frame.shape[:2]

    # ───── Top status bar ─────────────────────────────────────────────
    bar_h = 54
    _draw_panel(frame, 0, 0, w, bar_h)

    status = (
        f"SQUAT COACH  |  Reps: {reps}  |  Score: {score:.0f}  "
        f"|  {fps:.0f} fps  |  Voice: {'ON' if voice_on else 'OFF'}"
    )
    if paused:
        status += "  |  [PAUSED]"
    _put_text(frame, status, (14, 24), _C["white"], scale=0.55)

    # Phase badge
    phase_color = _PHASE_COLOR.get(phase, _C["white"])
    _put_text(frame, f"Phase: {phase.value}", (14, 46), phase_color, scale=0.55)

    # Score bar (top-right)
    bar_w_px = int(w * 0.32)
    bx, by   = w - bar_w_px - 14, 18
    bh       = 18
    fill_px  = int(bar_w_px * clamp(score, 0, 100) / 100)
    cv2.rectangle(frame, (bx, by), (bx + bar_w_px, by + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + fill_px, by + bh),
                  draw_color_for_score(score), -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w_px, by + bh), _C["gray"], 1)
    _put_text(frame, f"FORM {score:.0f}", (bx + 6, by + 14), _C["white"], scale=0.45)

    # ───── Bottom feedback panel ──────────────────────────────────────
    row_h    = 28
    n_rows   = max(1, len(issues))
    panel_h  = n_rows * row_h + 46
    panel_y  = h - panel_h
    _draw_panel(frame, 0, panel_y, w, panel_h, alpha=0.65)

    # Main coaching line
    _put_text(frame, feedback_text, (14, panel_y + 26), feedback_color,
              scale=0.72, thickness=2)

    # Per-issue progress bars
    bar_max_w = 180
    for i, key in enumerate(issues):
        row_y = panel_y + 40 + i * row_h
        count = issue_counters.get(key, 0)
        pct   = min(count / max(CONSECUTIVE_FRAMES, 1), 1.0)
        fill_w = int(bar_max_w * pct)

        color = _C["red"] if pct >= 1.0 else _C["yellow"]
        cv2.rectangle(frame, (14, row_y + 2), (14 + fill_w, row_y + 16), color, -1)
        cv2.rectangle(frame, (14, row_y + 2), (14 + bar_max_w, row_y + 16),
                      _C["gray"], 1)
        _put_text(frame, f"!  {key}", (bar_max_w + 28, row_y + 16), color, scale=0.5)

    # ───── Mini sparkline of recent scores (bottom-right) ─────────────
    if recent_scores:
        spark_w, spark_h = 220, 40
        sx = w - spark_w - 20
        sy = panel_y - spark_h - 10
        cv2.rectangle(frame, (sx - 6, sy - 6), (sx + spark_w + 6, sy + spark_h + 6),
                      (30, 30, 30), -1)
        pts = []
        max_n = 30
        vals = recent_scores[-max_n:]
        for i, v in enumerate(vals):
            px = sx + int(i * spark_w / max(len(vals) - 1, 1))
            py = sy + spark_h - int(clamp(v, 0, 100) / 100 * spark_h)
            pts.append((px, py))
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32)], False,
                          _C["green"], 2, cv2.LINE_AA)
        _put_text(frame, "Recent reps", (sx, sy - 8), _C["gray"], scale=0.4)

    # ───── Debug metrics (middle-right) ───────────────────────────────
    if debug and debug_metrics:
        lines = [
            f"Back angle  : {debug_metrics.get('back_angle', 0):.1f} deg",
            f"Knee L      : {debug_metrics.get('knee_l', 0):.1f} deg",
            f"Knee R      : {debug_metrics.get('knee_r', 0):.1f} deg",
            f"Hip Y       : {debug_metrics.get('hip_y', 0):.0f} px",
        ]
        for j, line in enumerate(lines):
            _put_text(frame, line, (w - 260, 80 + j * 20), _C["gray"], scale=0.45)


# ---------------------------------------------------------------------------
# Banner drawn when the user is calibrating or a rep just completed
# ---------------------------------------------------------------------------
def draw_banner(frame, text: str, color) -> None:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    bx = (w - tw) // 2
    by = h // 2 + th // 2
    pad = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - pad, by - th - pad), (bx + tw + pad, by + pad),
                  (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, _C["black"], 6, cv2.LINE_AA)
    cv2.putText(frame, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, color, 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    # ─── Camera ──────────────────────────────────────────────────────────
    # On macOS, CAP_AVFOUNDATION yields cleanest low-latency capture.
    import platform
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}. "
              "Close any app using the webcam and grant camera permission.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # latest frame only, no backlog

    # ─── Pipeline objects ────────────────────────────────────────────────
    detector = PoseDetector(smoothing_window=5)
    counter  = RepCounter(smoothing_window=7)
    analyzer = SquatAnalyzer()
    coach    = VoiceCoach(enabled=True)
    logger   = SessionLogger()

    voice_on = coach.available
    paused   = False
    debug    = False

    # Per-issue consecutive-frame counters (for on-screen progress bars
    # and for gating voice alerts until an issue is truly sustained)
    issue_counters: dict[str, int] = {}

    # Per-frame state kept between iterations for frame-skipping
    last_kp       = None
    last_analysis = None
    frame_count   = 0

    fps_smoother = MovingAverage(window=12)
    prev_t       = time.monotonic()

    # Wire rep-complete callback so we can finalize + log each rep
    def _on_rep_done(record) -> None:
        score, issues, depth_ok = analyzer.finalize_rep()
        counter.record_rep_score(score, issues, depth_ok)
        logger.record_rep(record.rep_number, score, issues, depth_ok)
        for iss in issues:
            logger.record_correction(iss)
        if voice_on:
            key = "Good rep" if score >= 85 else "Rep complete"
            coach.alert(key, cooldown=GOOD_REP_VOICE_COOLDOWN)

    counter.on_rep_complete(_on_rep_done)

    # ─── Banner ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  AI Squat Coach")
    print("=" * 60)
    print("  Controls (focus preview window):")
    print("    q / ESC → quit            p → pause / resume")
    print("    r       → reset session   v → toggle voice")
    print("    d       → toggle debug overlay")
    print()
    print("  TIP: Stand side-on to the camera ~1.5–2 m away,")
    print("       full body in frame. The coach calibrates for ~1 s")
    print("       while you stand still, then start squatting.")
    print("=" * 60)

    window = "AI Squat Coach"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

    # ─── Main loop ───────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Don't spin; just try again
                if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                    break
                continue

            frame = cv2.flip(frame, 1)   # mirror for natural UX
            h, w  = frame.shape[:2]

            # FPS (exponential moving average)
            now = time.monotonic()
            dt  = max(now - prev_t, 1e-4)
            fps = fps_smoother.update(1.0 / dt)
            prev_t = now

            # ── Pause handling ─────────────────────────────────────────
            if paused:
                draw_banner(frame, "PAUSED — press P to resume", _C["yellow"])
                cv2.imshow(window, frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == ord('p'):
                    paused = False
                    print("Resumed.")
                continue

            # ── MediaPipe inference (every N frames) ───────────────────
            frame_count += 1
            run_inference = (frame_count % INFERENCE_EVERY == 0)

            if run_inference:
                kp = detector.process(frame)
                last_kp = kp
            else:
                detector.draw_cached(frame)
                kp = last_kp

            # ── Squat analysis + rep counter ──────────────────────────
            phase = SquatPhase.CALIBRATING
            score = 100.0
            current_issues: list[str] = []
            feedback_text  = "Get into view — stand side-on, full body"
            feedback_color = _C["white"]
            debug_metrics: dict = {}

            if kp is not None and kp.valid:
                l_hip = kp.get("left_hip")
                r_hip = kp.get("right_hip")
                hip_y = float(np.mean([l_hip[1], r_hip[1]])) \
                    if (l_hip is not None and r_hip is not None) else 0.0

                phase = counter.update(hip_y, h)

                if run_inference:
                    analysis = analyzer.analyse(kp, phase)
                    last_analysis = analysis
                else:
                    analysis = last_analysis

                if analysis is not None and analysis.analysed:
                    score          = analysis.score
                    current_issues = list(analysis.issues)

                    debug_metrics = {
                        "back_angle": analysis.back_angle,
                        "knee_l":     analysis.knee_angle_left,
                        "knee_r":     analysis.knee_angle_right,
                        "hip_y":      hip_y,
                    }

                # Update per-issue streak counters
                active_set = set(current_issues)
                for key in list(issue_counters):
                    if key not in active_set:
                        issue_counters[key] = 0
                for key in current_issues:
                    issue_counters[key] = issue_counters.get(key, 0) + 1

                # Fire voice alerts for sustained issues
                if voice_on and current_issues:
                    # Pick the highest-count issue that has crossed the threshold
                    sustained = [
                        k for k in current_issues
                        if issue_counters.get(k, 0) >= CONSECUTIVE_FRAMES
                    ]
                    if sustained:
                        # Sort by priority (first in list = highest)
                        priority = ["Go deeper", "Push knees out",
                                    "Knees too far forward", "Keep chest up"]
                        sustained.sort(key=lambda k: priority.index(k)
                                       if k in priority else 99)
                        coach.alert(sustained[0], cooldown=VOICE_COOLDOWN_SEC)

                # Choose on-screen feedback
                if current_issues:
                    feedback_text  = current_issues[0]
                    feedback_color = _C["red"]
                elif phase == SquatPhase.CALIBRATING:
                    feedback_text  = "Calibrating — stand still"
                    feedback_color = _C["yellow"]
                elif phase == SquatPhase.STANDING:
                    feedback_text  = "Ready — squat when you like"
                    feedback_color = _C["green"]
                else:
                    feedback_text  = f"{phase.value} — form looks good"
                    feedback_color = _C["green"]

            else:
                # No valid pose
                draw_banner(frame, "Step into frame", _C["white"])

            # ── Render overlay ─────────────────────────────────────────
            recent_scores = [r.score for r in counter.rep_history if r.score > 0]
            draw_overlay(
                frame=frame,
                phase=phase,
                reps=counter.rep_count,
                score=score,
                fps=fps,
                issues=current_issues,
                issue_counters=issue_counters,
                feedback_text=feedback_text,
                feedback_color=feedback_color,
                voice_on=voice_on,
                paused=paused,
                debug=debug,
                debug_metrics=debug_metrics,
                recent_scores=recent_scores,
            )

            cv2.imshow(window, frame)

            # ── Keyboard input ─────────────────────────────────────────
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('p'):
                paused = True
                print("Paused.")
            elif k == ord('r'):
                counter.reset()
                analyzer.reset()
                issue_counters.clear()
                last_kp = last_analysis = None
                print("Session reset.")
            elif k == ord('v'):
                voice_on = not voice_on and coach.available
                coach.set_enabled(voice_on)
                print(f"Voice: {'ON' if voice_on else 'OFF'}")
            elif k == ord('d'):
                debug = not debug
                print(f"Debug overlay: {'ON' if debug else 'OFF'}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        coach.stop()
        logger.save()


if __name__ == "__main__":
    main()
