"""
pose_detector.py — MediaPipe Pose wrapper with smoothing and confidence filtering.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
from utils import PointSmoother

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Pose)
# ---------------------------------------------------------------------------
LM = mp.solutions.pose.PoseLandmark

KEY_LANDMARKS = {
    "left_shoulder":  LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_hip":       LM.LEFT_HIP,
    "right_hip":      LM.RIGHT_HIP,
    "left_knee":      LM.LEFT_KNEE,
    "right_knee":     LM.RIGHT_KNEE,
    "left_ankle":     LM.LEFT_ANKLE,
    "right_ankle":    LM.RIGHT_ANKLE,
    "left_ear":       LM.LEFT_EAR,
    "right_ear":      LM.RIGHT_EAR,
}

MIN_VISIBILITY = 0.5   # Landmark must exceed this to be considered detected


@dataclass
class PoseKeypoints:
    """Holds the resolved, smoothed landmark positions for one frame."""
    landmarks: Dict[str, np.ndarray] = field(default_factory=dict)
    visibility: Dict[str, float] = field(default_factory=dict)
    valid: bool = False

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.landmarks.get(name)

    def has(self, *names: str) -> bool:
        """Return True only if ALL named landmarks are present."""
        return all(n in self.landmarks for n in names)


class PoseDetector:
    """
    Wraps MediaPipe Pose.

    Key design decisions
    --------------------
    model_complexity=0   Fast-path model (~5–15 ms/frame on modern CPU).
                         Sufficient for lower-body squat analysis.
    smoothing_window=5   Per-landmark 3-axis moving average dampens jitter
                         without introducing noticeable lag.
    _last_results        Stored so the caller can redraw the skeleton on
                         frames where MediaPipe inference was skipped.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 0,          # ← was 1; 0 is ~3× faster
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_window: int = 5,
    ):
        self._mp_pose   = mp.solutions.pose
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self.pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,          # ← 0
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Per-landmark smoothers (x, y, z)
        self._smoothers: Dict[str, PointSmoother] = {
            name: PointSmoother(window=smoothing_window, dims=3)
            for name in KEY_LANDMARKS
        }

        # Cache last mediapipe result so draw_cached() can replay it cheaply
        self._last_results = None

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> PoseKeypoints:
        """
        Run MediaPipe inference on `frame` (BGR), draw skeleton in-place,
        return smoothed PoseKeypoints.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        # Always cache — even a None result clears stale skeleton data
        self._last_results = results

        kp = PoseKeypoints()

        if results.pose_landmarks is None:
            return kp

        # Draw skeleton on the provided frame
        self._mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
        )

        lm_list = results.pose_landmarks.landmark
        for name, idx in KEY_LANDMARKS.items():
            lm  = lm_list[idx]
            vis = lm.visibility if lm.visibility is not None else 0.0
            kp.visibility[name] = float(vis)

            if vis >= MIN_VISIBILITY:
                raw      = np.array([lm.x * w, lm.y * h, lm.z * w], dtype=float)
                smoothed = self._smoothers[name].update(raw)
                kp.landmarks[name] = smoothed

        critical = [
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]
        kp.valid = all(n in kp.landmarks for n in critical)
        return kp

    # ------------------------------------------------------------------
    def draw_cached(self, frame: np.ndarray) -> None:
        """
        Replay the last known skeleton onto `frame` without re-running
        MediaPipe inference.  Called on frames where inference was skipped
        to keep the visual overlay smooth even at reduced inference rate.
        """
        if self._last_results is None:
            return
        if self._last_results.pose_landmarks is None:
            return
        self._mp_draw.draw_landmarks(
            frame,
            self._last_results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
        )

    # ------------------------------------------------------------------
    def reset_smoothers(self) -> None:
        for s in self._smoothers.values():
            s.reset()
        self._last_results = None

    def release(self) -> None:
        self.pose.close()
