"""
squat_analyzer.py — Biomechanical form analysis and per rep scoring.

Checks performed each frame:
  1. Squat depth     — hip must drop below knee level
  2. Knee over toe   — knee X should not overshoot ankle X beyond threshold
  3. Knee valgus     — knees must not collapse inward relative to ankles
  4. Back angle      — torso lean must not be excessive

A rep score starts at 100 and deductions are applied per fault.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from utils import calculate_angle, clamp, MovingAverage
from pose_detector import PoseKeypoints
from rep_counter import SquatPhase


# ---------------------------------------------------------------------------
# Deduction magnitudes (points off per fault per evaluation)
# ---------------------------------------------------------------------------
DEDUCT_DEPTH      = 25
DEDUCT_KNEE_TOE   = 15
DEDUCT_VALGUS     = 20
DEDUCT_BACK       = 15

# Biomechanical thresholds
KNEE_TOE_SLACK_FRACTION   = 0.08   # knee may extend up to 8 % beyond ankle X
VALGUS_SLACK_FRACTION     = 0.04   # knees may collapse inward up to 4 % of hip width
BACK_LEAN_MIN_ANGLE       = 40.0   # degrees — torso-vertical angle below this = excessive lean
DEPTH_HIP_BELOW_KNEE_PX   = -5     # hip Y must be >= knee Y - 5px (in image coords, Y increases down)


@dataclass
class FormAnalysis:
    """Result of analysing one frame."""
    score: float = 100.0
    depth_ok: bool = True
    knee_ok: bool = True
    valgus_ok: bool = True
    back_ok: bool = True
    issues: List[str] = field(default_factory=list)
    knee_angle_left: float = 0.0
    knee_angle_right: float = 0.0
    back_angle: float = 0.0
    # True if we had enough landmarks to run the analysis
    analysed: bool = False


class SquatAnalyzer:
    """
    Stateful form analyser.

    Call `analyse(keypoints, phase)` each frame.
    At BOTTOM and during ASCENDING the analysis is sharpest.
    At end of rep call `finalize_rep()` to get a locked score.
    """

    def __init__(self):
        # Rolling best / worst scores within current rep
        self._rep_scores: List[float] = []
        self._rep_issues: List[str] = []
        self._rep_depth_ok: bool = True

        # Smoothed angles for trend display
        self._back_smoother   = MovingAverage(window=5)
        self._knee_l_smoother = MovingAverage(window=5)
        self._knee_r_smoother = MovingAverage(window=5)

        # Running session accumulators
        self.session_scores: List[float] = []
        self.all_issues: List[str] = []

    # ------------------------------------------------------------------
    # Per-frame analysis
    # ------------------------------------------------------------------

    def analyse(self, kp: PoseKeypoints, phase: SquatPhase) -> FormAnalysis:
        result = FormAnalysis()

        if not kp.valid:
            return result

        # ---- Resolve bilateral landmarks → use the side with better visibility ----
        side = self._dominant_side(kp)
        shoulder = kp.get(f"{side}_shoulder")
        hip      = kp.get(f"{side}_hip")
        knee     = kp.get(f"{side}_knee")
        ankle    = kp.get(f"{side}_ankle")

        if any(v is None for v in [shoulder, hip, knee, ankle]):
            return result

        result.analysed = True

        # ---- Knee angles ----
        l_knee = kp.get("left_knee")
        l_hip  = kp.get("left_hip")
        l_ankle = kp.get("left_ankle")
        r_knee = kp.get("right_knee")
        r_hip  = kp.get("right_hip")
        r_ankle = kp.get("right_ankle")

        if all(v is not None for v in [l_hip, l_knee, l_ankle]):
            result.knee_angle_left = self._knee_l_smoother.update(
                calculate_angle(l_hip, l_knee, l_ankle)
            )
        if all(v is not None for v in [r_hip, r_knee, r_ankle]):
            result.knee_angle_right = self._knee_r_smoother.update(
                calculate_angle(r_hip, r_knee, r_ankle)
            )

        # ---- Back angle (shoulder–hip vertical) ----
        # We define back angle as the angle between the torso line (shoulder→hip)
        # and the vertical axis. 0° = perfectly upright, 90° = horizontal.
        torso_vec = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])
        vertical  = np.array([0.0, 1.0])
        norm_t = np.linalg.norm(torso_vec)
        if norm_t > 1e-6:
            cos_a = np.dot(torso_vec, vertical) / norm_t
            back_angle_from_vertical = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        else:
            back_angle_from_vertical = 0.0

        result.back_angle = self._back_smoother.update(back_angle_from_vertical)

        # Only apply form checks during meaningful phases
        active = phase in (SquatPhase.DESCENDING, SquatPhase.BOTTOM, SquatPhase.ASCENDING)
        if not active:
            return result

        score = 100.0

        # ---- 1. Depth check ----
        # In image coordinates Y increases downward.
        # Hip Y >= Knee Y means hip is at or below knee (good).
        if hip[1] < knee[1] - DEPTH_HIP_BELOW_KNEE_PX:
            result.depth_ok = False
            result.issues.append("Go deeper")
            score -= DEDUCT_DEPTH
        else:
            self._rep_depth_ok = True

        # ---- 2. Knee over toe ----
        # Compute hip width ONCE (not per side).  Fall back to the single-side
        # hip x for both endpoints if one hip is missing — that gives 0 width,
        # and the `max(..., 10)` below still yields a sane slack value.
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if lh is not None and rh is not None:
            hip_width = float(abs(lh[0] - rh[0]))
        else:
            hip_width = 0.0

        knee_toe_fault = False
        for s in ("left", "right"):
            k = kp.get(f"{s}_knee")
            a = kp.get(f"{s}_ankle")
            h = kp.get(f"{s}_hip")
            if k is None or a is None or h is None:
                continue
            slack = max(KNEE_TOE_SLACK_FRACTION * hip_width, 10.0)
            # In image coords, a knee "overshooting" the ankle means it sits
            # further forward along the facing direction.  Side-on to the
            # camera the facing axis is x, so we compare knee.x to ankle.x.
            overshoot = k[0] - a[0] if s == "left" else a[0] - k[0]
            if overshoot > slack:
                knee_toe_fault = True

        if knee_toe_fault:
            result.knee_ok = False
            result.issues.append("Knees too far forward")
            score -= DEDUCT_KNEE_TOE

        # ---- 3. Knee valgus ----
        l_k = kp.get("left_knee")
        r_k = kp.get("right_knee")
        l_a = kp.get("left_ankle")
        r_a = kp.get("right_ankle")
        if all(v is not None for v in [l_k, r_k, l_a, r_a]):
            knee_width  = abs(l_k[0] - r_k[0])
            ankle_width = abs(l_a[0] - r_a[0])
            if ankle_width > 1e-4:
                # Valgus: knees collapse inward → knee_width < ankle_width by more than slack
                valgus_slack = VALGUS_SLACK_FRACTION * ankle_width
                if (ankle_width - knee_width) > valgus_slack:
                    result.valgus_ok = False
                    result.issues.append("Push knees out")
                    score -= DEDUCT_VALGUS

        # ---- 4. Back angle ----
        if result.back_angle > (90.0 - BACK_LEAN_MIN_ANGLE):
            result.back_ok = False
            result.issues.append("Keep chest up")
            score -= DEDUCT_BACK

        result.score = clamp(score, 0.0, 100.0)

        # Accumulate within rep
        self._rep_scores.append(result.score)
        self._rep_issues.extend(result.issues)
        if not result.depth_ok:
            self._rep_depth_ok = False

        return result

    # ------------------------------------------------------------------
    # Rep finalization
    # ------------------------------------------------------------------

    def finalize_rep(self) -> Tuple[float, List[str], bool]:
        """
        Called when a rep completes. Returns (score, unique_issues, depth_ok).
        Resets per-rep accumulators.
        """
        if self._rep_scores:
            # Use the worst score captured during the rep
            final_score = min(self._rep_scores)
        else:
            final_score = 0.0

        from collections import Counter
        issue_counts = Counter(self._rep_issues)
        # Return issues that appeared at least 3 frames
        unique_issues = [iss for iss, cnt in issue_counts.items() if cnt >= 3]

        depth_ok = self._rep_depth_ok

        self.session_scores.append(final_score)
        self.all_issues.extend(unique_issues)

        # Reset
        self._rep_scores.clear()
        self._rep_issues.clear()
        self._rep_depth_ok = True

        return final_score, unique_issues, depth_ok

    # ------------------------------------------------------------------
    # Session summary helpers
    # ------------------------------------------------------------------

    @property
    def session_average_score(self) -> float:
        if not self.session_scores:
            return 0.0
        return sum(self.session_scores) / len(self.session_scores)

    @property
    def most_frequent_issue(self) -> Optional[str]:
        from collections import Counter
        if not self.all_issues:
            return None
        return Counter(self.all_issues).most_common(1)[0][0]

    def reset(self):
        self._rep_scores.clear()
        self._rep_issues.clear()
        self._rep_depth_ok = True
        self.session_scores.clear()
        self.all_issues.clear()
        self._back_smoother.reset()
        self._knee_l_smoother.reset()
        self._knee_r_smoother.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dominant_side(self, kp: PoseKeypoints) -> str:
        """Return the side ('left' or 'right') with better average visibility."""
        left_vis = sum(
            kp.visibility.get(f"left_{p}", 0) for p in ("shoulder", "hip", "knee", "ankle")
        )
        right_vis = sum(
            kp.visibility.get(f"right_{p}", 0) for p in ("shoulder", "hip", "knee", "ankle")
        )
        return "left" if left_vis >= right_vis else "right"
