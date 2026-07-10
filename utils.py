import numpy as np
from collections import deque
from typing import Optional, Tuple, List


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle at point b, formed by vectors b->a and b->c.
    Returns angle in degrees [0, 180].
    """
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return midpoint between two 2D/3D points."""
    return (np.array(a) + np.array(b)) / 2.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


class MovingAverage:
    """
    Smooths a scalar signal using a sliding window mean.
    Handles cold-start by averaging over available samples.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._buf: deque = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf.clear()

    @property
    def value(self) -> Optional[float]:
        if not self._buf:
            return None
        return float(np.mean(self._buf))


class PointSmoother:
    """
    Smooths a 2D/3D point (x, y) or (x, y, z) using per-axis moving averages.
    """

    def __init__(self, window: int = 5, dims: int = 3):
        self._smoothers = [MovingAverage(window) for _ in range(dims)]
        self.dims = dims

    def update(self, point: np.ndarray) -> np.ndarray:
        arr = np.array(point)
        return np.array([s.update(arr[i]) for i, s in enumerate(self._smoothers)])

    def reset(self):
        for s in self._smoothers:
            s.reset()


def normalize_y(y: float, frame_height: int) -> float:
    """Convert raw pixel Y to [0,1] where 0=top, 1=bottom."""
    if frame_height == 0:
        return 0.0
    return y / frame_height


def safe_divide(num: float, denom: float, fallback: float = 0.0) -> float:
    if abs(denom) < 1e-9:
        return fallback
    return num / denom


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def draw_color_for_score(score: float) -> Tuple[int, int, int]:
    """
    Returns BGR color: green for high score, red for low.
    Score range [0, 100].
    """
    score = clamp(score, 0, 100)
    ratio = score / 100.0
    r = int((1 - ratio) * 255)
    g = int(ratio * 255)
    return (0, g, r)  # BGR
